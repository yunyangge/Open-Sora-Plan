

import inspect
from typing import Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import torch
import os
from PIL import Image
import decord
from einops import rearrange
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, CLIPImageProcessor, MT5Tokenizer, T5EncoderModel
from torchvision.transforms import Lambda, Compose

from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, HunyuanDiT2DModel
from diffusers.models.embeddings import get_2d_rotary_pos_embed
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import DDPMScheduler
from diffusers.utils import logging, BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from opensora.dataset.transform import ToTensorVideo, TemporalRandomCrop, MaxHWStrideResizeVideo, CenterCropResizeVideo, LongSideResizeVideo, SpatialStrideCropVideo, NormalizeVideo, ToTensorAfterResize
from opensora.models.diffusion.opensora_v1_3.modeling_opensora import OpenSoraT2V_v1_3
from opensora.schedulers.scheduling_flow_match_euler import FlowMatchEulerScheduler
from opensora.schedulers.inversion import flow_matching_inversion
from opensora.schedulers.sigma_schedule import opensora_linear_quadratic_schedule
from opensora.sample.pipeline_opensora import OpenSoraPipeline, OpenSoraPipelineOutput, rescale_noise_cfg
try:
    import torch_npu
    from opensora.npu_config import npu_config
    from opensora.acceleration.parallel_states import get_sequence_parallel_state, hccl_info
except:
    torch_npu = None
    npu_config = None
    from opensora.utils.parallel_states import get_sequence_parallel_state, nccl_info

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def is_video_file(file_path):
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.mpeg', '.mpg', '.3gp'}
    file_extension = os.path.splitext(file_path)[1].lower()
    return file_extension in video_extensions

def is_image_file(file_path):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    file_extension = os.path.splitext(file_path)[1].lower()
    return file_extension in image_extensions

def open_image(file_path):
    image = Image.open(file_path).convert("RGB")
    return image

def open_video(file_path, start_frame_idx, num_frames, frame_interval=1):

    decord_vr = decord.VideoReader(file_path, ctx=decord.cpu(0), num_threads=1)

    total_frames = len(decord_vr)
    frame_indices = list(range(start_frame_idx, min(start_frame_idx + num_frames * frame_interval, total_frames), frame_interval))

    if len(frame_indices) == 0:
        raise ValueError("No frames selected. Check your start_frame_idx and num_frames.")
    
    if len(frame_indices) < num_frames:
        raise ValueError(f"Requested {num_frames} frames but only {len(frame_indices)} frames are available, please adjust the start_frame_idx and num_frames or decrease the frame_interval.")
        
    video_data = decord_vr.get_batch(frame_indices).asnumpy()
    video_data = torch.from_numpy(video_data)
    video_data = video_data.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T C H W)
    return video_data


def get_pixel_values(file_path, num_frames):
    if is_image_file(file_path[0]):
        pixel_values = [open_image(path) for path in file_path]
        pixel_values = [torch.from_numpy(np.array(image)) for image in pixel_values]
        pixel_values = [rearrange(image, 'h w c -> c h w').unsqueeze(0) for image in pixel_values]
    elif is_video_file(file_path[0]):
        pixel_values = [open_video(video_path, 0, num_frames) for video_path in file_path]
    return pixel_values

class OpenSoraInversionPipeline(OpenSoraPipeline):

    def get_transform(self, max_height, max_width):
        norm_fun = Lambda(lambda x: 2. * x - 1.)
        resize = [CenterCropResizeVideo((max_height, max_width))]
        transform = Compose([
            ToTensorVideo(),
            *resize, 
            norm_fun
        ])
        return transform
    
    @torch.no_grad()
    def __call__(
        self,
        image_path: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        prompt: Union[str, List[str]] = None,
        prompt_target: Union[str, List[str]] = None,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = 50,
        use_linear_quadratic_schedule: bool = False,
        num_inverse_steps: Optional[int] = None,
        timesteps: List[int] = None,
        guidance_scale: Optional[float] = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_samples_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_2: Optional[torch.Tensor] = None,
        prompt_embeds_target: Optional[torch.Tensor] = None,
        prompt_embeds_2_target: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_2: Optional[torch.Tensor] = None,
        negative_prompt_embeds_target: Optional[torch.Tensor] = None,
        negative_prompt_embeds_2_target: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        prompt_attention_mask_2: Optional[torch.Tensor] = None,
        prompt_attention_mask_target: Optional[torch.Tensor] = None,
        prompt_attention_mask_2_target: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask_2: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask_target: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask_2_target: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        guidance_rescale: float = 0.0,
        max_sequence_length: int = 512,
        device = None, 
    ):
        
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. default height and width
        num_frames = num_frames or (self.transformer.config.sample_size_t - 1) * self.vae.vae_scale_factor[0] + 1
        height = height or self.transformer.config.sample_size[0] * self.vae.vae_scale_factor[1]
        width = width or self.transformer.config.sample_size[1] * self.vae.vae_scale_factor[2]

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            num_frames, 
            height,
            width,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
            prompt_embeds_2,
            negative_prompt_embeds_2,
            prompt_attention_mask_2,
            negative_prompt_attention_mask_2,
            callback_on_step_end_tensor_inputs,
        )
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = device or getattr(self, '_execution_device', None) or getattr(self, 'device', None) or torch.device('cuda')


        # 3. Encode input prompt

        (
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            device=device,
            dtype=self.transformer.dtype,
            num_samples_per_prompt=num_samples_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
            text_encoder_index=0,
        )

        if prompt_target is not None:
            (
                prompt_embeds_target,
                negative_prompt_embeds_target,
                prompt_attention_mask_target,
                negative_prompt_attention_mask_target,
            ) = self.encode_prompt(
                prompt=prompt_target,
                device=device,
                dtype=self.transformer.dtype,
                num_samples_per_prompt=num_samples_per_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                prompt_embeds=prompt_embeds_target,
                negative_prompt_embeds=negative_prompt_embeds_target,
                prompt_attention_mask=prompt_attention_mask_target,
                negative_prompt_attention_mask=negative_prompt_attention_mask_target,
                max_sequence_length=max_sequence_length,
                text_encoder_index=0,
            )
        else:
            prompt_embeds_target = None
            negative_prompt_embeds_target = None
            prompt_attention_mask_target = None
            negative_prompt_attention_mask_target = None

        if self.tokenizer_2 is not None:
            (
                prompt_embeds_2,
                negative_prompt_embeds_2,
                prompt_attention_mask_2,
                negative_prompt_attention_mask_2,
            ) = self.encode_prompt(
                prompt=prompt,
                device=device,
                dtype=self.transformer.dtype,
                num_samples_per_prompt=num_samples_per_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                prompt_embeds=prompt_embeds_2,
                negative_prompt_embeds=negative_prompt_embeds_2,
                prompt_attention_mask=prompt_attention_mask_2,
                negative_prompt_attention_mask=negative_prompt_attention_mask_2,
                max_sequence_length=77,
                text_encoder_index=1,
            )

            if prompt_target is not None:
                (
                    prompt_embeds_2_target,
                    negative_prompt_embeds_2_target,
                    prompt_attention_mask_2_target,
                    negative_prompt_attention_mask_2_target,
                ) = self.encode_prompt(
                    prompt=prompt_target,
                    device=device,
                    dtype=self.transformer.dtype,
                    num_samples_per_prompt=num_samples_per_prompt,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    negative_prompt=negative_prompt,
                    prompt_embeds=prompt_embeds_2_target,
                    negative_prompt_embeds=negative_prompt_embeds_2_target,
                    prompt_attention_mask=prompt_attention_mask_2_target,
                    negative_prompt_attention_mask=negative_prompt_attention_mask_2_target,
                    max_sequence_length=77,
                    text_encoder_index=1,
                )
        else:
            prompt_embeds_2 = None
            negative_prompt_embeds_2 = None
            prompt_attention_mask_2 = None
            negative_prompt_attention_mask_2 = None

            prompt_embeds_2_target = None
            negative_prompt_embeds_2_target = None
            prompt_attention_mask_2_target = None
            negative_prompt_attention_mask_2_target = None


        # 4. Prepare timesteps
        if not isinstance(self.scheduler, FlowMatchEulerScheduler):
            raise ValueError("Only FlowMatchEulerScheduler is supported for inversion.")
        
        sigmas = None
        if use_linear_quadratic_schedule:
            sigmas = opensora_linear_quadratic_schedule(num_inference_steps=num_inference_steps, approximate_steps=min(num_inference_steps * 10, 1000))
            sigmas = np.array(sigmas)
            print(f"use linear quadratic schedule, sigmas: {sigmas}, approximate_steps: {min(num_inference_steps * 10, 1000)}")

        # 5. Prepare latent variables
        world_size = None
        if get_sequence_parallel_state():
            world_size = hccl_info.world_size if torch_npu is not None else nccl_info.world_size
        num_channels_latents = self.transformer.config.in_channels

        # ==================================== prepare image ====================================   
        image = open_image(image_path)
        image = torch.from_numpy(np.array(image))  # [h, w, c]
        image = rearrange(image, 'h w c -> c h w').unsqueeze(1)  #  [c 1 h w]
        transform = self.get_transform(height, width)
        image = transform(image).unsqueeze(0)
        image = image.to(dtype=self.vae.vae.dtype, device=device)
        latents = self.vae.encode(image)
        print('latents shape:', latents.shape)
        # ==================================== prepare image ====================================

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        if not isinstance(self.scheduler, FlowMatchEulerScheduler):
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        else:
            extra_step_kwargs = {}
        # 7 create image_rotary_emb, style embedding & time ids
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask])
            if self.tokenizer_2 is not None:
                prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2])
                prompt_attention_mask_2 = torch.cat([negative_prompt_attention_mask_2, prompt_attention_mask_2])

            if prompt_embeds_target is not None:
                prompt_embeds_target = torch.cat([negative_prompt_embeds_target, prompt_embeds_target])
                prompt_attention_mask_target = torch.cat([negative_prompt_attention_mask_target, prompt_attention_mask_target])
                if self.tokenizer_2 is not None:
                    prompt_embeds_2_target = torch.cat([negative_prompt_embeds_2_target, prompt_embeds_2_target])
                    prompt_attention_mask_2_target = torch.cat([negative_prompt_attention_mask_2_target, prompt_attention_mask_2_target])

        prompt_embeds = prompt_embeds.to(device=device)
        prompt_attention_mask = prompt_attention_mask.to(device=device)
        if prompt_embeds_target is not None:
            prompt_embeds_target = prompt_embeds_target.to(device=device)
            prompt_attention_mask_target = prompt_attention_mask_target.to(device=device)

        if self.tokenizer_2 is not None:
            prompt_embeds_2 = prompt_embeds_2.to(device=device)
            prompt_attention_mask_2 = prompt_attention_mask_2.to(device=device)
            if prompt_embeds_2_target is not None:
                prompt_embeds_2_target = prompt_embeds_2_target.to(device=device)
                prompt_attention_mask_2_target = prompt_attention_mask_2_target.to(device=device)


        # ==================make sp=====================================
        if get_sequence_parallel_state():
            prompt_embeds = rearrange(
                prompt_embeds, 
                'b (n x) h -> b n x h', 
                n=world_size,
                x=prompt_embeds.shape[1] // world_size
                ).contiguous()
            rank = hccl_info.rank if torch_npu is not None else nccl_info.rank
            prompt_embeds = prompt_embeds[:, rank, :, :]
        # ==================make sp=====================================

        print('Start flow matching inverse...')
        print(f'input image path: {image_path}')
        print(f'prompt: {prompt}')
        print(f'use linear quadratic schedule: {use_linear_quadratic_schedule}')
        print(f'num inverse step: {num_inverse_steps}')
        # 8. Denoising loop

        # ==================prepare my shape=====================================
        # predict the noise residual
        if prompt_embeds.ndim == 3:
            prompt_embeds = prompt_embeds.unsqueeze(1)  # b l d -> b 1 l d
        if prompt_attention_mask.ndim == 2:
            prompt_attention_mask = prompt_attention_mask.unsqueeze(1)  # b l -> b 1 l
        if prompt_embeds_2 is not None and prompt_embeds_2.ndim == 2:
            prompt_embeds = prompt_embeds.unsqueeze(1)  # b d -> b 1 d
        
        attention_mask = torch.ones_like(latents)[:, 0].to(device=device)
        # ==================prepare my shape=====================================
        if self.do_classifier_free_guidance:
            attention_mask = torch.cat([attention_mask] * 2)
        # ==================make sp=====================================
        if get_sequence_parallel_state():
            attention_mask = attention_mask.repeat(1, world_size, 1, 1)
        # ==================make sp=====================================

        # inversion process
        latents = flow_matching_inversion(
            self.transformer,
            self.scheduler,
            latents=latents,
            attention_mask=attention_mask,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            pooled_projections=prompt_embeds_2,
            encoder_hidden_states_target=prompt_embeds_target,
            encoder_attention_mask_target=prompt_attention_mask_target,
            pooled_projections_target=prompt_embeds_2_target,
            sigmas=sigmas,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            guidance_scale=self._guidance_scale,
            num_inference_steps=num_inference_steps,
            num_inverse_steps=num_inverse_steps,
            resample=True,
        )

        # ==================make sp=====================================
        if get_sequence_parallel_state():
            latents_shape = list(latents.shape)  # b c t//sp h w
            full_shape = [latents_shape[0] * world_size] + latents_shape[1:]  # # b*sp c t//sp h w
            all_latents = torch.zeros(full_shape, dtype=latents.dtype, device=latents.device)
            torch.distributed.all_gather_into_tensor(all_latents, latents)
            latents_list = list(all_latents.chunk(world_size, dim=0))
            latents = torch.cat(latents_list, dim=2)
        # ==================make sp=====================================

        if not output_type == "latent":
            videos = self.decode_latents(latents)
            videos = videos[:, :num_frames, :height, :width]
        else:
            videos = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (videos, )

        return OpenSoraPipelineOutput(videos=videos)

if __name__ == "__main__":

    import math
    from transformers import T5EncoderModel, T5Tokenizer, AutoTokenizer, MT5EncoderModel, CLIPTextModelWithProjection
    from torchvision.utils import save_image
    from opensora.models.causalvideovae import WFVAEModelWrapper
    from opensora.models.diffusion.opensora_v1_5.modeling_opensora import OpenSoraT2V_v1_5
    weight_dtype = torch.float16
    device = torch.device('cuda')

    class Args:
        ae_path = '/home/save_dir/lzj/Middle888'
        model_path = '/home/save_dir/runs/t2v_1_5_dit_bs16x8x32_lr1e-4_256x256_192x192_new6b_14kpretrained/checkpoint-96000/model_ema'
        text_encoder_name_1 = '/home/save_dir/pretrained/t5/t5-v1_1-xl'
        text_encoder_name_2 = '/home/save_dir/pretrained/clip/models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k/snapshots/bc7788f151930d91b58474715fdce5524ad9a189'
        input_image_path = '/home/save_dir/projects/gyy/mmdit/Open-Sora-Plan/validation_dir/i2v_0011.png'
        prompt = 'A coffee cup with "anytext" foam floating on it.'
        negative_prompt = ''
        num_inference_steps = 100
        num_inverse_steps = 80
        use_linear_quadratic_schedule = False
        height = 256
        width = 256
        num_frames = 1
        guidance_scale = 7.0
        num_samples_per_prompt = 1
        max_sequence_length = 512
        save_img_path = './test_inversion'


    args = Args()

    vae = WFVAEModelWrapper(args.ae_path)
    vae.vae = vae.vae.to(device=device, dtype=weight_dtype).eval()
    vae.vae_scale_factor = [8, 8, 8]

    text_encoder_1 = T5EncoderModel.from_pretrained(args.text_encoder_name_1, torch_dtype=weight_dtype).eval()
    tokenizer_1 = AutoTokenizer.from_pretrained(args.text_encoder_name_1)

    if args.text_encoder_name_2 is not None:
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.text_encoder_name_2, torch_dtype=weight_dtype).eval()
        tokenizer_2 = AutoTokenizer.from_pretrained(args.text_encoder_name_2)
    else:
        text_encoder_2, tokenizer_2 = None, None

    transformer_model = OpenSoraT2V_v1_5.from_pretrained(args.model_path, torch_dtype=weight_dtype).eval()
    scheduler = FlowMatchEulerScheduler()

    pipeline = OpenSoraInversionPipeline(
        vae=vae,
        transformer=transformer_model,
        scheduler=scheduler,
        text_encoder=text_encoder_1,
        tokenizer=tokenizer_1,
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
    ).to(device=device)

    videos = pipeline(
        image_path=args.input_image_path,
        prompt=args.prompt,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        num_inverse_steps=args.num_inverse_steps,
        use_linear_quadratic_schedule=args.use_linear_quadratic_schedule,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt,
        num_samples_per_prompt=args.num_samples_per_prompt,
        max_sequence_length=args.max_sequence_length,
    ).videos

    videos = rearrange(videos, 'b t h w c -> (b t) c h w')
    os.makedirs(args.save_img_path, exist_ok=True)

    save_image(
        videos / 255.0, 
        os.path.join(args.save_img_path, 'test.jpg'),
        nrow=math.ceil(math.sqrt(videos.shape[0])), 
        normalize=True, 
        value_range=(0, 1)
    )  # b c h w