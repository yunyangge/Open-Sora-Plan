import time
import traceback

try:
    import torch_npu
    from opensora.npu_config import npu_config
except:
    torch_npu = None
    npu_config = None
import glob
import json
import pickle
import os, io, csv, math, random
import numpy as np
import torchvision
from einops import rearrange
from os.path import join as opj
from collections import Counter

import cv2
import pandas as pd
import time
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from PIL import Image
import gc
import decord

from opensora.utils.dataset_utils import DecordInit
from opensora.utils.utils import text_preprocessing
from opensora.dataset.transform import get_params, maxhwresize, add_masking_notice, calculate_statistics, \
    add_aesthetic_notice_image, add_aesthetic_notice_video
from opensora.utils.mask_utils import MaskProcessor, STR_TO_TYPE
from opensora.dataset.t2v_dataset import T2V_dataset, DataSetProg, DecordDecoder, filter_resolution, find_closest_y
from opensora.utils.custom_logger import get_logger

logger = get_logger(os.path.relpath(__file__))

dataset_prog = DataSetProg()

def type_ratio_normalize(mask_type_ratio_dict):
    for k, v in mask_type_ratio_dict.items():
        assert v >= 0, f"mask_type_ratio_dict[{k}] should be non-negative, but got {v}"
    total = sum(mask_type_ratio_dict.values())
    length = len(mask_type_ratio_dict)
    if total == 0:
        return {k: 1.0 / length for k in mask_type_ratio_dict.keys()}
    return {k: v / total for k, v in mask_type_ratio_dict.items()}

class Transition_dataset(T2V_dataset):
    def __init__(self, args, resize_transform, transform, temporal_sample, tokenizer_1, tokenizer_2):
        super().__init__(
            args=args, 
            transform=transform,  
            temporal_sample=temporal_sample, 
            tokenizer_1=tokenizer_1, 
            tokenizer_2=tokenizer_2
        )

        self.resize_transform = resize_transform

        if self.num_frames != 1:
            self.mask_type_ratio_dict_video = args.mask_type_ratio_dict_video if args.mask_type_ratio_dict_video is not None else {'random_temporal': 1.0}
            self.mask_type_ratio_dict_video = {STR_TO_TYPE[k]: v for k, v in self.mask_type_ratio_dict_video.items()}
            self.mask_type_ratio_dict_video = type_ratio_normalize(self.mask_type_ratio_dict_video)
                
        self.mask_type_ratio_dict_image = args.mask_type_ratio_dict_image if args.mask_type_ratio_dict_image is not None else {'random_spatial': 1.0}
        self.mask_type_ratio_dict_image = {STR_TO_TYPE[k]: v for k, v in self.mask_type_ratio_dict_image.items()}
        self.mask_type_ratio_dict_image = type_ratio_normalize(self.mask_type_ratio_dict_image)

        print(f"mask_type_ratio_dict_video: {self.mask_type_ratio_dict_video}")
        print(f"mask_type_ratio_dict_image: {self.mask_type_ratio_dict_image}")

        self.mask_processor = MaskProcessor(
            max_height=args.max_height,
            max_width=args.max_width,
            min_clear_ratio=args.min_clear_ratio,
            max_clear_ratio=args.max_clear_ratio,
        )

        self.default_text_ratio = args.default_text_ratio

    def __getitem__(self, idx):
        try:
            return self.get_data(idx)
        except Exception as e:
            print(f'Error with {e}')
            index_cand = self.shape_idx_dict[self.sample_size[idx]]  # pick same shape
            return self.__getitem__(random.choice(index_cand))
    
    def define_frame_index(self, data):
        shape_idx_dict = {}
        new_cap_list = []
        sample_size = []
        
        data_dir = data
        json_path = os.path.join(data_dir, 'transition.json')
        
        with open(json_path, 'r') as f:
            sub_list = json.load(f)
        ### for debug
        sub_list = sub_list * 64
        ###
        logger.info(f'Start to build transition.json, including {len(sub_list)} items in total.')
        for index, i in enumerate(tqdm(sub_list)):
            # get path
            i['video_dir'] = os.path.join(data_dir, i['video_id'])
            # resize resolution
            height, width = i['resolution']['height'], i['resolution']['width']
            
            if not self.force_resolution:
                tr_h, tr_w = maxhwresize(height, width, self.max_hxw)
                _, _, sample_h, sample_w = get_params(tr_h, tr_w, self.hw_stride)
                if sample_h <= 0 or sample_w <= 0 or sample_h * sample_w < self.min_hxw:
                    continue
                # filter aspect
                is_pick = filter_resolution(
                    sample_h, sample_w, max_h_div_w_ratio=self.hw_aspect_thr, min_h_div_w_ratio=1/self.hw_aspect_thr
                    )
                if not is_pick:
                    continue
                i['resolution'].update(dict(sample_height=sample_h, sample_width=sample_w))
            else:
                aspect = self.max_height / self.max_width
                is_pick = filter_resolution(
                    height, width, max_h_div_w_ratio=self.hw_aspect_thr*aspect, min_h_div_w_ratio=1/self.hw_aspect_thr*aspect
                    )
                if not is_pick:
                    continue
                sample_h, sample_w = self.max_height, self.max_width
                i['resolution'].update(dict(sample_height=sample_h, sample_width=sample_w))

            # resample frames
            i['fps'] = 24
            closest_num_frames = find_closest_y(
                i['num_frames'], vae_stride_t=self.ae_stride_t, model_ds_t=self.sp_size
            )
            margin = (i['num_frames'] - closest_num_frames) // 2
            i['start_frame_idx'] = margin
            frame_indices = np.arange(i['start_frame_idx'], i['start_frame_idx']+closest_num_frames).astype(int)
            i['sample_frame_index'] = frame_indices.tolist()

            new_cap_list.append(i)
            pre_define_shape = f"{len(i['sample_frame_index'])}x{sample_h}x{sample_w}"
            sample_size.append(pre_define_shape)
            
        counter_sample_size = Counter(sample_size)
        
        if not self.force_resolution and self.max_hxw is not None and self.min_hxw is not None:
            assert all([np.prod(np.array(k.split('x')[1:]).astype(np.int32)) <= self.max_hxw for k in counter_sample_size.keys()])
            assert all([np.prod(np.array(k.split('x')[1:]).astype(np.int32)) >= self.min_hxw for k in counter_sample_size.keys()])
        
        filter_major_num = 4 * self.total_batch_size
        new_cap_list, sample_size = zip(*[[i, j] for i, j in zip(new_cap_list, sample_size) if counter_sample_size[j] >= filter_major_num])
        for idx, shape in enumerate(sample_size):
            if shape_idx_dict.get(shape, None) is None:
                shape_idx_dict[shape] = [idx]
            else:
                shape_idx_dict[shape].append(idx)
        logger.info(f'Finish to build transition.json, including {len(new_cap_list)} valid items in total')
        return new_cap_list, sample_size, shape_idx_dict
    

    def get_data(self, idx):
        video_data = dataset_prog.cap_list[idx]
        video_path = os.path.dir(video_data['video_dir'], 'video.mp4')
        if not os.path.exists(video_path):
            logger.warning(f"file {video_path} do not exist, random choice a new one with same shape!")
            index_cand = self.shape_idx_dict[self.sample_size[idx]]
            return self.__getitem__(random.choice(index_cand))
        
        key_frame_selected_index = random.randint(0, len(video_data['key_idx']))

        key_frame_path = os.path.dir(video_data['video_dir'], f'key_frame_{key_frame_selected_index}.jpg')
        key_frame_edge_path = os.path.dir(video_data['video_dir'], f'key_frame_{key_frame_selected_index}_edge.jpg')
        key_frame_cap = video_data['key_frame_caption'][key_frame_selected_index]

        sample_h = video_data['resolution']['sample_height']
        sample_w = video_data['resolution']['sample_width']

        if self.video_reader == 'opencv':
            logger.warning('Not support OpenCV, using Decord to read video.')
        
        frame_indice = video_data['sample_frame_index']
        start_frame_idx = video_data['start_frame_idx']
        clip_total_frames = video_data['num_frames']
        fps = video_data['fps']
        
        decord_vr = DecordDecoder(video_path)
        video_data = decord_vr.get_batch(frame_indice)
        if video_data is not None:
            video_data = video_data.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T C H W)
        else:
            raise ValueError(f'Get video_data {video_data}')
        # TODO
        return video_data


    def drop(self, text, is_video=True):
        rand_num = random.random()
        rand_num_text = random.random()

        if rand_num < self.cfg:
            if rand_num_text < self.default_text_ratio:
                if not is_video:
                    text = "The image showcases a scene with coherent and clear visuals." 
                else:
                    text = "The video showcases a scene with coherent and clear visuals." 
            else:
                text = ''

        return dict(text=text)
    
    
    def get_video(self, idx):
        video_data = dataset_prog.cap_list[idx]
        video_path = video_data['path']
        # assert os.path.exists(video_path), f"file {video_path} do not exist!"
        sample_h = video_data['resolution']['sample_height']
        sample_w = video_data['resolution']['sample_width']
        
        if self.video_reader == 'decord':
            video = self.decord_read(video_data)
        elif self.video_reader == 'opencv':
            video = self.opencv_read(video_data)
        else:
            NotImplementedError(f'Found {self.video_reader}, but support decord or opencv')
        # import ipdb;ipdb.set_trace()

        video = self.resize_transform(video)  # T C H W -> T C H W
        assert video.shape[2] == sample_h and video.shape[3] == sample_w, f'sample_h ({sample_h}), sample_w ({sample_w}), video ({video.shape}), video_path ({video_path})'

        inpaint_cond_data = self.mask_processor(video, mask_type_ratio_dict=self.mask_type_ratio_dict_video)
        mask, masked_video = inpaint_cond_data['mask'], inpaint_cond_data['masked_pixel_values']

        video = self.transform(video)  # T C H W -> T C H W
        masked_video = self.transform(masked_video)  # T C H W -> T C H W

        video = torch.cat([video, masked_video, mask], dim=1)  # T 2C+1 H W

        video = video.transpose(0, 1)  # T C H W -> C T H W
        text = video_data['cap']
        if not isinstance(text, list):
            text = [text]
        text = [random.choice(text)]
        if video_data.get('aesthetic', None) is not None or video_data.get('aes', None) is not None:
            aes = video_data.get('aesthetic', None) or video_data.get('aes', None)
            text = [add_aesthetic_notice_video(text[0], aes)]

        text = self.drop(text, is_video=True)['text']

        text_tokens_and_mask_1 = self.tokenizer_1(
            text,
            max_length=self.model_max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        input_ids_1 = text_tokens_and_mask_1['input_ids']
        cond_mask_1 = text_tokens_and_mask_1['attention_mask']
        
        input_ids_2, cond_mask_2 = None, None
        if self.tokenizer_2 is not None:
            text_tokens_and_mask_2 = self.tokenizer_2(
                text,
                max_length=self.tokenizer_2.model_max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors='pt'
            )
            input_ids_2 = text_tokens_and_mask_2['input_ids']
            cond_mask_2 = text_tokens_and_mask_2['attention_mask']

        return dict(
            pixel_values=video, input_ids_1=input_ids_1, cond_mask_1=cond_mask_1, 
            input_ids_2=input_ids_2, cond_mask_2=cond_mask_2,
            )

    def get_image(self, idx):
        image_data = dataset_prog.cap_list[idx]  # [{'path': path, 'cap': cap}, ...]
        sample_h = image_data['resolution']['sample_height']
        sample_w = image_data['resolution']['sample_width']

        image = Image.open(image_data['path']).convert('RGB')  # [h, w, c]
        image = torch.from_numpy(np.array(image))  # [h, w, c]
        image = rearrange(image, 'h w c -> c h w').unsqueeze(0)  #  [1 c h w]

        image = self.resize_transform(image)  # [1 c h w]
        assert image.shape[2] == sample_h, image.shape[3] == sample_w

        inpaint_cond_data = self.mask_processor(image, mask_type_ratio_dict=self.mask_type_ratio_dict_image)
        mask, masked_image = inpaint_cond_data['mask'], inpaint_cond_data['masked_pixel_values']   

        image = self.transform(image)
        masked_image = self.transform(masked_image)

        image = torch.cat([image, masked_image, mask], dim=1)  #  [1 2C+1 H W]
        # image = [torch.rand(1, 3, 480, 640) for i in image_data]
        image = image.transpose(0, 1)  # [1 C H W] -> [C 1 H W]

        caps = image_data['cap'] if isinstance(image_data['cap'], list) else [image_data['cap']]
        caps = [random.choice(caps)]
        # caps = [caps[0]]
        if '/sam/' in image_data['path']:
            caps = [add_masking_notice(caps[0])]
        if image_data.get('aesthetic', None) is not None or image_data.get('aes', None) is not None:
            aes = image_data.get('aesthetic', None) or image_data.get('aes', None)
            caps = [add_aesthetic_notice_image(caps[0], aes)]
        text = text_preprocessing(caps, support_Chinese=self.support_Chinese)
        text = self.drop(text, is_video=False)['text']

        text_tokens_and_mask_1 = self.tokenizer_1(
            text,
            max_length=self.model_max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        input_ids_1 = text_tokens_and_mask_1['input_ids']  # 1, l
        cond_mask_1 = text_tokens_and_mask_1['attention_mask']  # 1, l
        
        input_ids_2, cond_mask_2 = None, None
        if self.tokenizer_2 is not None:
            text_tokens_and_mask_2 = self.tokenizer_2(
                text,
                max_length=self.tokenizer_2.model_max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors='pt'
            )
            input_ids_2 = text_tokens_and_mask_2['input_ids']  # 1, l
            cond_mask_2 = text_tokens_and_mask_2['attention_mask']  # 1, l

        return dict(
            pixel_values=image, input_ids_1=input_ids_1, cond_mask_1=cond_mask_1, motion_score=None, 
            input_ids_2=input_ids_2, cond_mask_2=cond_mask_2
            )
