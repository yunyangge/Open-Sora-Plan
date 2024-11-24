

import inspect
from typing import Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import torch
import os
from PIL import Image
from einops import rearrange
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, CLIPImageProcessor, MT5Tokenizer, T5EncoderModel
from torchvision.transforms import Lambda, Compose

from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL, HunyuanDiT2DModel
from diffusers.models.embeddings import get_2d_rotary_pos_embed
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import DDPMScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.utils import logging, BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from opensora.dataset.transform import ToTensorVideo, TemporalRandomCrop, MaxHWStrideResizeVideo, CenterCropResizeVideo, LongSideResizeVideo, SpatialStrideCropVideo, NormalizeVideo, ToTensorAfterResize

from opensora.models.diffusion.opensora_v1_3.modeling_opensora import OpenSoraT2V_v1_3
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

def is_image_file(file_path):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    file_extension = os.path.splitext(file_path)[1].lower()
    return file_extension in image_extensions

def open_image(file_path):
    image = Image.open(file_path).convert("RGB")
    return image

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

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
    
    def sample(
        self, 
        latents,
        prompt_embeds,
        prompt_embeds_2,
        prompt_attention_mask,
        timesteps,
        num_warmup_steps,
        guidance_scale,
        guidance_rescale,
        extra_step_kwargs,
        callback_on_step_end,
        callback_on_step_end_tensor_inputs,
        world_size,
        device,
        progress_bar,
    ):
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            if not isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # expand scalar t to 1-D tensor to match the 1st dim of latent_model_input
            if not isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
                timestep = torch.tensor([t] * latent_model_input.shape[0], device=device).to(
                    dtype=latent_model_input.dtype
                )
            else:
                timestep = t.expand(latent_model_input.shape[0])

            # ==================prepare my shape=====================================
            # predict the noise residual
            if prompt_embeds.ndim == 3:
                prompt_embeds = prompt_embeds.unsqueeze(1)  # b l d -> b 1 l d
            if prompt_attention_mask.ndim == 2:
                prompt_attention_mask = prompt_attention_mask.unsqueeze(1)  # b l -> b 1 l
            if prompt_embeds_2 is not None and prompt_embeds_2.ndim == 2:
                prompt_embeds = prompt_embeds.unsqueeze(1)  # b d -> b 1 d
            
            attention_mask = torch.ones_like(latent_model_input)[:, 0].to(device=device)
            # ==================prepare my shape=====================================

            # ==================make sp=====================================
            if get_sequence_parallel_state():
                attention_mask = attention_mask.repeat(1, world_size, 1, 1)
            # ==================make sp=====================================

            noise_pred = self.transformer(
                latent_model_input,
                attention_mask=attention_mask, 
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                timestep=timestep,
                pooled_projections=prompt_embeds_2,
                return_dict=False,
            )[0]
            assert not torch.any(torch.isnan(noise_pred))
            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if self.do_classifier_free_guidance and guidance_rescale > 0.0 and not isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                prompt_embeds_2 = callback_outputs.pop("prompt_embeds_2", prompt_embeds_2)
                negative_prompt_embeds_2 = callback_outputs.pop(
                    "negative_prompt_embeds_2", negative_prompt_embeds_2
                )
            
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

        return latents

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
        if not isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps
            inverse_timesteps = reversed(timesteps)
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            self._num_timesteps = len(timesteps)
        else:
            sigmas = None
            if use_linear_quadratic_schedule:
                sigmas = opensora_linear_quadratic_schedule(num_inference_steps=num_inference_steps, approximate_steps=min(num_inference_steps * 10, 1000))
                sigmas = np.array(sigmas)
                print(f"use linear quadratic schedule, sigmas: {sigmas}, approximate_steps: {min(num_inference_steps * 10, 1000)}")
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps, sigmas)
            timesteps = self.scheduler.sigmas[:num_inference_steps] * self.scheduler.num_train_timesteps
            self.scheduler.sigmas = reversed(self.scheduler.sigmas)
            print(f"self.scheduler.sigmas: {self.scheduler.sigmas}")
            inverse_timesteps = self.scheduler.sigmas[:num_inference_steps] * self.scheduler.num_train_timesteps
            self.scheduler.timesteps = inverse_timesteps
            print(f"self.scheduler.timesteps: {self.scheduler.timesteps}")
            num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
            self._num_timesteps = len(timesteps)
        

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
        if not isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
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
        self.scheduler._step_index = 0
        if num_inverse_steps is not None:
            assert num_inverse_steps <= num_inference_steps
            inverse_t = inverse_timesteps[:num_inverse_steps]
            t = reversed(inverse_t)
        else:
            num_inverse_steps = num_inference_steps
        with self.progress_bar(total=num_inverse_steps) as progress_bar:
            noised_latents = self.sample(
                latents=latents,
                prompt_embeds=prompt_embeds,
                prompt_embeds_2=prompt_embeds_2,
                prompt_attention_mask=prompt_attention_mask,
                timesteps=inverse_t,
                num_warmup_steps=num_warmup_steps,
                guidance_scale=guidance_scale,
                guidance_rescale=guidance_rescale,
                extra_step_kwargs=extra_step_kwargs,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                world_size=world_size,
                device=device,
                progress_bar=progress_bar,
            )
        print('Start flow matching sample...')
        prompt_embeds_target = prompt_embeds_target if prompt_embeds_target is not None else prompt_embeds
        prompt_embeds_2_target = prompt_embeds_2_target if prompt_embeds_2_target is not None else prompt_embeds_2
        prompt_attention_mask_target = prompt_attention_mask_target if prompt_attention_mask_target is not None else prompt_attention_mask
        self.scheduler._step_index = num_inference_steps - num_inverse_steps
        self.scheduler.sigmas = reversed(self.scheduler.sigmas)
        self.scheduler.timesteps = timesteps
        print(f"self.scheduler.sigmas: {self.scheduler.sigmas}")
        print(f"self.scheduler.timesteps: {self.scheduler.timesteps}")
        with self.progress_bar(total=min(num_inference_steps, num_inverse_steps)) as progress_bar:
            new_latents = self.sample(
                latents=noised_latents,
                prompt_embeds=prompt_embeds_target,
                prompt_embeds_2=prompt_embeds_2_target,
                prompt_attention_mask=prompt_attention_mask_target,
                timesteps=t,
                num_warmup_steps=0,
                guidance_scale=guidance_scale,
                guidance_rescale=guidance_rescale,
                extra_step_kwargs=extra_step_kwargs,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                world_size=world_size,
                device=device,
                progress_bar=progress_bar,
            )

        latents = new_latents

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
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
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
        num_inverse_steps = 40
        use_linear_quadratic_schedule = True
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
    scheduler = FlowMatchEulerDiscreteScheduler()

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
