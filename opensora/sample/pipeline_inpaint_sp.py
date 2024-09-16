# Copyright 2024 PixArt-Sigma Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import html
import inspect
import re
import urllib.parse as ul
from typing import Callable, List, Optional, Tuple, Union
import math
import torch
from transformers import T5EncoderModel, T5Tokenizer
from einops import rearrange
from diffusers.models import AutoencoderKL, Transformer2DModel
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.utils import (
    BACKENDS_MAPPING,
    deprecate,
    is_bs4_available,
    is_ftfy_available,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput

try:
    import torch_npu
    from opensora.acceleration.parallel_states import get_sequence_parallel_state, hccl_info
except:
    torch_npu = None
    from opensora.utils.parallel_states import get_sequence_parallel_state, nccl_info

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_bs4_available():
    from bs4 import BeautifulSoup

if is_ftfy_available():
    import ftfy

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

from .pipeline_inpaint import EXAMPLE_DOC_STRING, retrieve_timesteps, OpenSoraInpaintPipeline as OriginOpenSoraInpaintPipeline

class OpenSoraInpaintPipeline(OriginOpenSoraInpaintPipeline):

    @torch.no_grad()
    def __call__(
        self,
        # NOTE inpaint
        conditional_images: Optional[List[torch.FloatTensor]] = None,
        conditional_images_indices: Optional[List[int]] = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        timesteps: List[int] = None,
        guidance_scale: float = 4.5,
        motion_score: float = None, 
        num_samples_per_prompt: Optional[int] = 1,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        clean_caption: bool = True,
        use_resolution_binning: bool = True,
        max_sequence_length: int = 300,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:

        # 1. Check inputs. Raise error if not correct
        num_frames = num_frames or self.transformer.config.sample_size_t * self.vae.vae_scale_factor[0]
        height = height or self.transformer.config.sample_size[0] * self.vae.vae_scale_factor[1]
        width = width or self.transformer.config.sample_size[1] * self.vae.vae_scale_factor[2]
        self.check_inputs(
            prompt,
            num_frames, 
            height,
            width,
            negative_prompt,
            callback_steps,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
        )

        # 2. Default height and width to transformer
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        # import ipdb;ipdb.set_trace()
        device = getattr(self, '_execution_device', None) or getattr(self, 'device', None) or torch.device('cuda')

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt,
            do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            num_samples_per_prompt=num_samples_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            clean_caption=clean_caption,
            max_sequence_length=max_sequence_length,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

        # 5. Prepare latents.
        latent_channels = self.transformer.config.in_channels
        world_size = hccl_info.world_size if torch_npu is not None else nccl_info.world_size
        latents = self.prepare_latents(
            batch_size * num_samples_per_prompt,
            latent_channels,
            (num_frames + world_size - 1) // world_size if get_sequence_parallel_state() else num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        if get_sequence_parallel_state():
            prompt_embeds = rearrange(prompt_embeds, 'b (n x) h -> b n x h', n=world_size,
                                      x=prompt_embeds.shape[1] // world_size).contiguous()
            rank = hccl_info.rank if torch_npu is not None else nccl_info.rank
            prompt_embeds = prompt_embeds[:, rank, :, :]

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Prepare micro-conditions.
        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}

        masked_video, mask = self.get_masked_video_mask(
            conditional_images, 
            conditional_images_indices, 
            batch_size, 
            num_samples_per_prompt, 
            num_frames, 
            height, 
            width,
            do_classifier_free_guidance,
            latents.dtype,
            device
        )

        if get_sequence_parallel_state():
            latents_frames = latents.shape[2]
            rank = hccl_info.rank if torch_npu is not None else nccl_info.rank
            masked_video = masked_video[:, :, latents_frames * rank: latents_frames * (rank + 1)]
            mask = mask[:, :, latents_frames * rank: latents_frames * (rank + 1)]

        # 7. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                # inpaint
                latent_model_input = torch.cat([latent_model_input, masked_video, mask], dim=1)

                current_timestep = t
                if not torch.is_tensor(current_timestep):
                    # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                    # This would be a good case for the `match` statement (Python 3.10+)
                    is_mps = latent_model_input.device.type == "mps"
                    if isinstance(current_timestep, float):
                        dtype = torch.float32 if is_mps else torch.float64
                    else:
                        dtype = torch.int32 if is_mps else torch.int64
                    current_timestep = torch.tensor([current_timestep], dtype=dtype, device=latent_model_input.device)
                elif len(current_timestep.shape) == 0:
                    current_timestep = current_timestep[None].to(latent_model_input.device)
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                current_timestep = current_timestep.expand(latent_model_input.shape[0])

                # import ipdb;ipdb.set_trace()
                if prompt_embeds.ndim == 3:
                    prompt_embeds = prompt_embeds.unsqueeze(1)  # b l d -> b 1 l d
                if prompt_attention_mask.ndim == 2:
                    prompt_attention_mask = prompt_attention_mask.unsqueeze(1)  # b l -> b 1 l
                # prepare attention_mask.
                # b c t h w -> b t h w
                attention_mask = torch.ones_like(latent_model_input)[:, 0]
                if get_sequence_parallel_state():
                    attention_mask = attention_mask.repeat(1, world_size, 1, 1)
                # predict noise model_output
                noise_pred = self.transformer(
                    latent_model_input,
                    attention_mask=attention_mask, 
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    timestep=current_timestep,
                    added_cond_kwargs=added_cond_kwargs,
                    motion_score=motion_score, 
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # learned sigma
                if self.transformer.config.out_channels // 2 == latent_channels:
                    noise_pred = noise_pred.chunk(2, dim=1)[0]
                else:
                    noise_pred = noise_pred

                # compute previous image: x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                # print(f'latents_{i}_{t}', torch.max(latents), torch.min(latents), torch.mean(latents), torch.std(latents))
                
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)
        # import ipdb;ipdb.set_trace()
        # latents = latents.squeeze(2)
        world_size = hccl_info.world_size if torch_npu is not None else nccl_info.world_size
        if get_sequence_parallel_state():
            latents_shape = list(latents.shape)
            full_shape = [latents_shape[0] * world_size] + latents_shape[1:]
            all_latents = torch.zeros(full_shape, dtype=latents.dtype, device=latents.device)
            torch.distributed.all_gather_into_tensor(all_latents, latents)
            latents_list = list(all_latents.chunk(world_size, dim=0))
            latents = torch.cat(latents_list, dim=2)

        if not output_type == "latent":
            # b t h w c
            image = self.decode_latents(latents)
            image = image[:, :num_frames, :height, :width]
        else:
            image = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
    
    
    def decode_latents(self, latents):
        # print(f'before vae decode', torch.max(latents).item(), torch.min(latents).item(), torch.mean(latents).item(), torch.std(latents).item())
        video = self.vae.decode(latents.to(self.vae.vae.dtype))
        # print(f'after vae decode', torch.max(video).item(), torch.min(video).item(), torch.mean(video).item(), torch.std(video).item())
        video = ((video / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().permute(0, 1, 3, 4, 2).contiguous() # b t h w c
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        return video
