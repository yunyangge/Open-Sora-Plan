# import torch

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu

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

# from torch_npu.contrib import transfer_to_npu
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def get_args():
    parser = argparse.ArgumentParser(description='Use of argparse')
    parser.add_argument('--part_nums',type=int,default=2)
    parser.add_argument('--part_idx',type=int,default=1)
    parser.add_argument('--pkl_path',type=str,default='/home/save_dir/captions/10.12_hwpkl_panda/panda_part1_final_1193638_hw.pkl')
    parser.add_argument('--video_path',type=str,default='/home/obs_data/20240426')
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
    return output_text


def get_dino_features(frames, model, h=224, w=224, patch_size=14):
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((h//patch_size*patch_size, w//patch_size*patch_size)),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    images = []
    for i in range(frames.shape[0]):
        images.append(transform(frames[i]))
    img_tensor = torch.stack(images, dim=0).to('cuda')
 
    with torch.no_grad():
        features_dict = model.forward_features(img_tensor)
    cls_token, patch_tokens = features_dict['x_norm_clstoken'][:, None], features_dict['x_norm_patchtokens']
    frame_features = torch.cat([cls_token, patch_tokens], dim=1)
    return frame_features

def get_edge_image(frame, model):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    image = transform(frame)[None, ...].to('cuda')
    with torch.no_grad():
        result = model(image)
    edge_image = torch.squeeze(result[-1]).detach().cpu().numpy()
    edge_image = Image.fromarray((edge_image * 255).astype(np.uint8))
    return edge_image

def preprocess(video_info, dinov2, pidinet, Qwen2VL, Qwen2VL_processor, 
               slice_len=[93], frame_interval=12, 
               sem_thres=0.3, sem_thres_key=0.7, save_dir='test', fps=24):
    video_path = video_info['path']
    crop_info = video_info["crop"]
    res_info = video_info["resolution"]
    
    if video_info['num_frames'] < slice_len[0]:
        return
    if not (crop_info[0]==0 and crop_info[2]==0 and crop_info[1]==res_info['width'] and crop_info[3]==res_info['height']):
        return
    
    video = decord.VideoReader(video_path, ctx=cpu(0))
    # frames: [f, h, w, c]
    frames = video.get_batch(range(video_info['cut'][0], video_info['cut'][1], 1)).asnumpy()
    f, h, w, c = frames.shape
    # text detection
    
    frame_features = get_dino_features(frames, model=dinov2)
    cos_sim = F.cosine_similarity(frame_features[:, 0, :].unsqueeze(1), frame_features[:, 0, :].unsqueeze(0), dim=-1)

    cur_frame = 0
    vaild_frame_range = video_info['num_frames']

    while cur_frame < vaild_frame_range:
        candidate_end_frames = [cur_frame + i for i in slice_len]
        # del out of range index
        candidate_end_frames = list(filter(lambda x : x < vaild_frame_range, candidate_end_frames))
        if not candidate_end_frames:
            break
        candidate_end_sim = cos_sim[cur_frame, candidate_end_frames]
        end_val, end_idx = torch.min(candidate_end_sim, dim=-1)

        end_frame = cur_frame + slice_len[end_idx]

        if end_val >= sem_thres:
            cur_frame += slice_len[-1]
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
        captions.append(captioner(image=frames[cur_frame], model=Qwen2VL, processor=Qwen2VL_processor)[0])
        captions.append(captioner(image=frames[end_frame], model=Qwen2VL, processor=Qwen2VL_processor)[0])
        
        cnt = 0
        for val, idx in zip(candidate_key_val, candidate_key_idx):
            key_frame = Image.fromarray(frames[idx])
            key_frame.save(os.path.join(sub_dir, f'key_frame_{cnt}.jpg'))

            key_frame_edge = get_edge_image(frames[idx], model=pidinet)
            key_frame_edge.save(os.path.join(sub_dir, f'key_frame_{cnt}_edge.jpg'))

            captions.append(captioner(image=key_frame, model=Qwen2VL, processor=Qwen2VL_processor)[0])
            cnt += 1

        slice_frames = frames[cur_frame:end_frame]
        slice_images = [slice_frames[i] for i in range(slice_frames.shape[0])]
        imageio.mimwrite(os.path.join(sub_dir, 'video.mp4'), slice_images, fps=fps)

        slice_data = {
            'video_id': slice_id,
            'num_frames': slice_len[end_idx],
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
        
        cur_frame += slice_len[end_idx]
        torch.cuda.empty_cache()

    return

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

    print(f'Video information is loaded!, video number: {len(video_paths)}!')
    
    Qwen2VL, Qwen2VL_processor, dinov2, pidinet = load_models()
    
    nums_per_part = len(video_paths)//args.part_nums
    video_paths_alloc = video_paths[args.part_idx*nums_per_part : (args.part_idx+1)*nums_per_part]
    
    print(f'video_paths_alloc nums: {len(video_paths_alloc)}!')

    for item in tqdm.tqdm(video_paths_alloc):
        item['path'] = os.path.join(args.video_path, item['path'])
        try:
            preprocess(item, dinov2=dinov2, pidinet=pidinet, Qwen2VL=Qwen2VL, Qwen2VL_processor=Qwen2VL_processor, save_dir=save_dir)

        except Exception as e:
            traceback.print_exc()
            