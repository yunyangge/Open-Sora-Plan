# import torch

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
# import torch

import torch.nn.functional as F
from torchvision import transforms as T
import decord
import glob, os, json
import traceback
import argparse
from decord import cpu
import math

import cv2

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from pidinet import create_pidinet

from PIL import Image
import imageio
import numpy as np

import uuid
import tqdm
import pickle

import time
from torch.utils.data import DataLoader, Dataset


# from torch_npu.contrib import transfer_to_npu
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class ProcessDataset(Dataset):
    def __init__(self, pkl, slice_len, actual_slice_len, video_path):
        super().__init__()
        self.pkl = pkl
        self.slice_len = slice_len
        self.actual_slice_len = actual_slice_len
        self.video_path = video_path
        print(f'part video nums: {len(video_paths_alloc)}!')
        self.get_dino_transform()
        
    def __len__(self):
        return len(self.pkl)
    
    def get_dino_transform(self, dino_h=224, dino_w=224, dino_patch_size=14):
        self.dino_transform = T.Compose([
            T.ToTensor(),
            T.Resize((dino_h//dino_patch_size*dino_patch_size, dino_w//dino_patch_size*dino_patch_size)),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def load_videos(self, video_path, video_info):
        crop_info = video_info["crop"]
        res_info = video_info["resolution"]

        if video_info['num_frames'] < self.actual_slice_len:
            return None
        if not (crop_info[0]==0 and crop_info[2]==0 and crop_info[1]==res_info['width'] and crop_info[3]==res_info['height']):
            return None
        if res_info['height']/res_info['width']<0.5 or res_info['height']/res_info['width']>0.75:
            return None
        
        video = decord.VideoReader(video_path, ctx=cpu(0))
        speed_up_times = self.actual_slice_len / self.slice_len
        frames = video.get_batch(np.arange(video_info['cut'][0], video_info['cut'][1], speed_up_times)).asnumpy()
        return frames
    
    def wrapper(self, index):
        video_info = self.pkl[index]
        video_path = os.path.join(self.video_path, video_info['path'])
        
        ori_frames = self.load_videos(video_path, video_info)
        frames = []
        for i in range(ori_frames.shape[0]):
            frames.append(self.dino_transform(ori_frames[i]))
        dino_frames = torch.stack(frames, dim=0)

        return ori_frames, dino_frames

    def __getitem__(self, index):
        try:
            ori_frames, dino_frames = self.wrapper(index)
            return ori_frames, dino_frames
        except Exception as e:
            return None, None
        # ori_frames, dino_frames = self.wrapper(index)
        # return ori_frames, dino_frames

def collate_fn(batch):
    ori_frames, dino_frames = zip(*batch)
    if ori_frames is None:
        return None, None
    return ori_frames, dino_frames

def get_args():
    parser = argparse.ArgumentParser(description='Use of argparse')
    parser.add_argument('--part_nums',type=int,default=2)
    parser.add_argument('--part_idx',type=int,default=1)
    parser.add_argument('--pkl_path',type=str,default='/storage/anno_pkl/vid_nocn_res160_pkl/panda_part1_final_1193638.pkl')
    parser.add_argument('--video_path',type=str,default='/storage/dataset/panda70m')
    args = parser.parse_args()
    return args

# /home/obs_data/open-sora-plan/datasets_sucai,/home/save_dir/captions/10.12_hwpkl_movie_sucai/sucai_final_3880570.pkl

def load_models():
    Qwen2VL = Qwen2VLForConditionalGeneration.from_pretrained(
        "checkpoints/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/51c47430f97dd7c74aa1fa6825e68a813478097f", 
        torch_dtype="auto", device_map="auto", 
        local_files_only=True
    )

    Qwen2VL_processor = AutoProcessor.from_pretrained(
        "checkpoints/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/51c47430f97dd7c74aa1fa6825e68a813478097f", 
        min_pixels=256*28*28, max_pixels=1280*28*28,
        local_files_only=True, 
    )

    dinov2 = torch.hub.load(
        'checkpoints/facebookresearch_dinov2_main', 
        model='dinov2_vitb14_reg', 
        source='local'
    ).to(Qwen2VL.device)
    dinov2.eval()

    pidinet = create_pidinet(
        'pidinet/trained_models/table5_pidinet.pth'
    ).to(Qwen2VL.device)
    pidinet.eval()

    return Qwen2VL, Qwen2VL_processor, dinov2, pidinet


def captioner(image, model, processor, instruct='Describe this image in one sentence.'):
    start = time.time()

    describe_prefix = [
            'The image depicts ', 
            'The image captures ', 
            'In the image, ', 
            'The image showcases ', 
            'The image features ', 
            'The image is ', 
            'The image appears to be ', 
            'The image shows ', 
            'The image begins with ', 
            'The image displays ', 
            'The image begins in ', 
            'The image consists of ', 
            'The image opens with ', 
            'The image opens on ', 
            'The image appears to capture ', 
            'The image appears to show ', 
            "The image appears to depict ", 
            "The image opens in ", 
            "The image appears to focus closely on ", 
            "The image starts with ", 
            "The image begins inside ", 
            "The image presents ", 
            "The image takes place in ", 
            "The image appears to showcase ", 
            "The image appears to display ", 
            "The image appears to focus on ", 
            "The image appears to feature "
    ]

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    # "image": image_path,
                },
                {"type": "text", "text": instruct},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=image,
        videos=None,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    for prefix in describe_prefix:
        if output_text[0].startswith(prefix):
            text = output_text[0][len(prefix):]
            output_text[0] = text[0].upper() + text[1:]
            break
    # print(f"Caption Time: {time.time()-start}")
    return output_text


def get_dino_features(frames, model):
    start = time.time()
    
    with torch.no_grad():
        features_dict = model.forward_features(frames)
    cls_token, patch_tokens = features_dict['x_norm_clstoken'][:, None], features_dict['x_norm_patchtokens']
    frame_features = torch.cat([cls_token, patch_tokens], dim=1)
    # print(f"Dino Time: {time.time()-start}")
    return frame_features

def get_edge_image(frame, model):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    image = transform(frame)[None, ...].cuda()
    with torch.no_grad():
        result = model(image)
    edge_image = torch.squeeze(result[-1]).detach().cpu().numpy()
    edge_image = Image.fromarray((edge_image * 255).astype(np.uint8))
    return edge_image


if __name__ == "__main__":
    args = get_args()
    with open(args.pkl_path, 'rb') as f:
        data = pickle.load(f)
    video_paths = data

    pkl_path = args.pkl_path
    prefix = pkl_path.split('/')[-1].split('.')[0]
    save_dir = f'{prefix}_num{args.part_nums}_idx{args.part_idx}'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    Qwen2VL, Qwen2VL_processor, dinov2, pidinet = load_models()
    
    nums_per_part = len(video_paths)//args.part_nums
    video_paths_alloc = video_paths[args.part_idx*nums_per_part : (args.part_idx+1)*nums_per_part]
    
    slice_len = 93
    actual_slice_len = 144

    frame_interval=12
    sem_thres=0.3
    sem_thres_key=0.7
    fps=24
    
    data = ProcessDataset(video_paths_alloc, slice_len=slice_len, actual_slice_len=actual_slice_len, video_path=args.video_path)

    loader = DataLoader(data, batch_size=1, 
                        num_workers=8, 
                        pin_memory=False, 
                        prefetch_factor=4, 
                        shuffle=False, drop_last=False, collate_fn=collate_fn)

    for ori_frames, dino_frames in tqdm.tqdm(loader):
        # import pdb; pdb.set_trace()
        ori_frames, dino_frames = ori_frames[0], dino_frames[0]
        if ori_frames is None:
            continue
        dino_frames = dino_frames.cuda()
        # ori_frames, dino_frames = ori_frames.to(Qwen2VL.device), dino_frames.to(Qwen2VL.device)
        f, h, w, c = ori_frames.shape
        # text detection
        
        frame_features = get_dino_features(dino_frames, model=dinov2)
        cos_sim = F.cosine_similarity(frame_features[:, 0, :].unsqueeze(1), frame_features[:, 0, :].unsqueeze(0), dim=-1)
        
        cur_frame = 0
        while True:
            end_frame = cur_frame + slice_len - 1
            if end_frame >= f:
                break

            candidate_end_sim = cos_sim[cur_frame, end_frame]
            end_val, end_idx = torch.min(candidate_end_sim, dim=-1)

            if end_val >= sem_thres:
                cur_frame = end_frame
                continue

            # key frame = from small to big: max(sim(s, k), sim(k, e))
            candidate_key_sim = torch.max(cos_sim[cur_frame+1:end_frame-1, cur_frame], cos_sim[cur_frame+1:end_frame-1, end_frame])
            key_val, key_idx = torch.sort(candidate_key_sim)
            # the start point of candidate key is cur_frame+1
            key_idx = cur_frame + 1 + key_idx
            end_val, end_idx, key_val, key_idx = end_val.item(), end_idx.item(), key_val.tolist(), key_idx.tolist()

            candidate_key_val, candidate_key_idx = [key_val[0]], [key_idx[0]]
            for val, idx in zip(key_val[1:], key_idx[1:]):
                if val > sem_thres_key: # unselect key frame with similar semantic
                    break
                available = True
                for candidate_idx in candidate_key_idx:
                    if abs(idx-candidate_idx) < frame_interval:
                        available = False
                if available:
                    candidate_key_val.append(val)
                    candidate_key_idx.append(idx)

            # create dir
            slice_id = str(uuid.uuid4().hex)

            sub_dir = os.path.join(save_dir, slice_id)
            if not os.path.exists(sub_dir):
                os.mkdir(sub_dir)
            
            captions = []
            captions.append(captioner(image=ori_frames[cur_frame], model=Qwen2VL, processor=Qwen2VL_processor)[0])
            captions.append(captioner(image=ori_frames[end_frame], model=Qwen2VL, processor=Qwen2VL_processor)[0])
            
            cnt = 0
            for val, idx in zip(candidate_key_val, candidate_key_idx):
                key_frame = Image.fromarray(ori_frames[idx])
                key_frame.save(os.path.join(sub_dir, f'key_frame_{cnt}.jpg'))

                key_frame_edge = get_edge_image(ori_frames[idx], model=pidinet)
                key_frame_edge.save(os.path.join(sub_dir, f'key_frame_{cnt}_edge.jpg'))

                captions.append(captioner(image=key_frame, model=Qwen2VL, processor=Qwen2VL_processor)[0])
                cnt += 1

            slice_frames = ori_frames[cur_frame:end_frame+1]
            slice_images = [slice_frames[i] for i in range(slice_frames.shape[0])]
            imageio.mimwrite(os.path.join(sub_dir, 'video.mp4'), slice_images, fps=fps)

            candidate_key_idx = [i - cur_frame for i in candidate_key_idx]

            slice_data = {
                'video_id': slice_id,
                'num_frames': slice_len,
                'end_frame_sim': end_val,
                'fps': fps,
                'key_idx': candidate_key_idx,
                'key_val': candidate_key_val,
                'start_caption': captions[0],
                'end_caption': captions[1],
                'key_captions': captions[2:],
                'resolution': {
                    "height": h,
                    "width": w
                }
            }
            with open(os.path.join(sub_dir, 'info.json'), "w") as file:
                json.dump(slice_data, file)
            
            cur_frame += slice_len
            # torch.cuda.empty_cache()