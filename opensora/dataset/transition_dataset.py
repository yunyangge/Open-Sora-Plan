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

from opensora.utils.utils import text_preprocessing
from opensora.dataset.transform import get_params, maxhwresize
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
    def __init__(self, args, transform, condition_transform, temporal_sample, tokenizer_1, tokenizer_2):
        super().__init__(
            args=args, 
            transform=transform,  
            temporal_sample=temporal_sample, 
            tokenizer_1=tokenizer_1, 
            tokenizer_2=tokenizer_2
        )

        self.condition_transform = condition_transform

    def __getitem__(self, idx):
        try:
            return self.get_data(idx)
        except Exception as e:
            logger.warning(f'Error with {e}')
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
        # ### for debug
        # sub_list = sub_list*64
        # ###
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
        
        new_cap_list, sample_size = zip(*[[i, j] for i, j in zip(new_cap_list, sample_size)])
        for idx, shape in enumerate(sample_size):
            if shape_idx_dict.get(shape, None) is None:
                shape_idx_dict[shape] = [idx]
            else:
                shape_idx_dict[shape].append(idx)
        logger.info(f'Finish to build transition.json, including {len(new_cap_list)} valid items in total')
        return new_cap_list, sample_size, shape_idx_dict
    

    def get_data(self, idx):
        video_data = dataset_prog.cap_list[idx]
        video_path = os.path.join(video_data['video_dir'], 'video.mp4')
        if not os.path.exists(video_path):
            logger.warning(f"file {video_path} do not exist, random choice a new one with same shape!")
            index_cand = self.shape_idx_dict[self.sample_size[idx]]
            return self.__getitem__(random.choice(index_cand))
        
        key_frame_selected_index = random.randint(0, len(video_data['key_idx'])-1)

        key_frame_path = os.path.join(video_data['video_dir'], f'key_frame_{key_frame_selected_index}.jpg')
        key_frame_edge_path = os.path.join(video_data['video_dir'], f'key_frame_{key_frame_selected_index}_edge.jpg')
        key_frame_cap = video_data['key_frame_caption'][key_frame_selected_index]
        key_frame_idx = video_data['key_idx'][key_frame_selected_index] - video_data['start_frame_idx']

        text = [key_frame_cap]
        text = text_preprocessing(text, support_Chinese=self.support_Chinese)

        sample_h = video_data['resolution']['sample_height']
        sample_w = video_data['resolution']['sample_width']

        if self.video_reader == 'opencv':
            logger.warning('Not support OpenCV, using Decord to read video.')
        
        frame_indice = video_data['sample_frame_index']
        decord_vr = DecordDecoder(video_path)
        frames = decord_vr.get_batch(frame_indice)
        
        if frames is not None:
            frames = frames.permute(0, 3, 1, 2)  # (T H W C) -> (T C H W)
        else:
            raise ValueError(f'Get video frames {frames}')
        
        video = self.transform(frames)  # T C H W -> T C H W
        
        transition_invisible_mask = torch.ones_like(video, device=video.device, dtype=video.dtype)[:, :1] 
        transition_invisible_mask[0, ...] = 0
        transition_invisible_mask[-1, ...] = 0  # [T, 1, H, W]

        masked_video = video * (1-transition_invisible_mask)
        # # load key_frame
        # key_frame = Image.open(key_frame_path).convert('RGB')  # [h, w, c]
        # key_frame = torch.from_numpy(np.array(key_frame))  # [h, w, c]
        # key_frame = rearrange(key_frame, 'h w c -> c h w')[None, ...]  #  [1 c h w]

        # key_frame = self.resize_transform(key_frame)  # [1 c h w]
        # assert key_frame.shape[2] == sample_h, key_frame.shape[3] == sample_w
        # key_frame = self.transform(key_frame).transpose(0, 1)  # [1 C H W] -> [C 1 H W]

        # load key_frame_edge
        key_frame_edge = Image.open(key_frame_edge_path).convert('L')  # [h, w]
        key_frame_edge = torch.from_numpy(np.array(key_frame_edge))[None, None, ...] # [1, 1, h, w]
        key_frame_edge = self.condition_transform(key_frame_edge)
        
        transition_invisible_mask[key_frame_idx, 0, ...] = key_frame_edge[0, 0, ...] # [T, 1, H, W]

        video = torch.cat([video, masked_video, transition_invisible_mask], dim=1)  # T 2C+1 H W
        video = video.transpose(0, 1)  # T C H W -> C T H W

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

        return {
            "pixel_values": video,
            "input_ids_1": input_ids_1,
            "cond_mask_1": cond_mask_1,
            "input_ids_2": input_ids_2,
            "cond_mask_2": cond_mask_2
        }


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
    