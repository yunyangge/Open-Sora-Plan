# coding=utf-8
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
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

from typing import Mapping, Any
from logging import getLogger

import torch
from torch import nn
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training import get_args
from megatron.core import mpu

from mindspeed_mm.models.common.communications import collect_tensors_across_ranks
from mindspeed_mm.utils.utils import get_dtype
from mindspeed_mm.models.sora_model import SoRAModel

from einops import rearrange, repeat

logger = getLogger(__name__)


class ResI2VModel(SoRAModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self, 
        video, 
        start_frame=None,
        video_mask=None, 
        prompt_ids=None, 
        prompt_mask=None, 
        prompt_ids_2=None, 
        prompt_mask_2=None, 
        **kwargs
    ):
        with torch.autocast("cuda", enabled=False):
            with torch.no_grad():
                if video is not None:
                    self.index = 0
                    # Visual Encode
                    if self.load_video_features:
                        latents = video
                    else:
                        video = video.to(self.ae.dtype) 
                        latents = self.ae.encode(video)
                        start_frame = start_frame.to(self.ae.dtype)
                        start_frame_latents = self.ae.encode(start_frame)
                    # Text Encode
                    if self.load_text_features:
                        prompt = prompt_ids
                        prompt_2 = prompt_ids_2
                    else:
                        B, _, L = prompt_ids.shape
                        prompt_ids = prompt_ids.view(-1, L)
                        prompt_mask = prompt_mask.view(-1, L)
                        prompt = self.text_encoder.encode(prompt_ids, prompt_mask)
                        prompt = prompt.view(B, 1, L, -1)
                        prompt_mask = prompt_mask.view(B, 1, L)

                        if self.text_encoder_2 is not None and prompt_ids_2 is not None:
                            B_, _, L_ = prompt_ids_2.shape
                            prompt_ids_2 = prompt_ids_2.view(-1, L_)
                            prompt_mask_2 = prompt_mask_2.view(-1, L_)
                            prompt_2 = self.text_encoder_2.encode(prompt_ids_2, prompt_mask_2)
                            prompt_2 = prompt_2.view(B_, 1, -1)
        # print("--------------------------shape--------------------------")
        # print(f"latent: {latents.shape}, prompt: {prompt.shape}, prompt_2: {prompt_2.shape}, video_mask: {video_mask.shape}, prompt_mask: {prompt_mask.shape}, prompt_mask_2: {prompt_mask_2.shape}")
        # print("--------------------------dtype--------------------------")
        # print(f"latent: {latents.dtype}, prompt: {prompt.dtype}, prompt_2: {prompt_2.dtype}, video_mask: {video_mask.dtype}, prompt_mask: {prompt_mask.dtype}, prompt_mask_2: {prompt_mask_2.dtype}")
        # print("--------------------------device--------------------------")
        # print(f"latent: {latents.device}, prompt: {prompt.device}, prompt_2: {prompt_2.device}, video_mask: {video_mask.device}, prompt_mask: {prompt_mask.device}, prompt_mask_2: {prompt_mask_2.device}")

        if video_mask is None:
            video_mask = torch.ones_like(latents[:, 0], dtype=latents.dtype, device=latents.device)

        # Gather the results after encoding of ae and text_encoder
        if self.enable_encoder_dp:
            if self.index == 0:
                self.init_cache(latents, start_frame_latents, video_mask, prompt, prompt_mask, prompt_2)
            latents, start_frame_latents, video_mask, prompt, prompt_mask, prompt_2 = self.get_feature_from_cache()

        start_frame_latents = repeat(start_frame_latents, 'b c 1 h w -> b c t h w', t=latents.shape[2])
        latents = latents - start_frame_latents
        output = self.diffusion.q_sample(latents, model_kwargs=kwargs)

        noised_latents = output.get('x_t', None)
        noise = output.get('noise', None)
        timesteps = output.get('timesteps', None)
        sigmas = output.get('sigmas', None)

        model_output = self.predictor(
            torch.cat([noised_latents, start_frame_latents], dim=1).to(self.weight_dtype),
            attention_mask=video_mask,
            timestep=timesteps,
            encoder_hidden_states=prompt,
            encoder_attention_mask=prompt_mask,
            pooled_projections=prompt_2,
            **kwargs,
        )
        return model_output, latents, timesteps, noise, video_mask, sigmas

    def compute_loss(
        self, model_output, latents, timesteps, noise, video_mask, sigmas
    ):
        """compute diffusion loss"""
        loss_dict = self.diffusion.training_losses(
            model_output=model_output,
            x_start=latents,
            noise=noise,
            timesteps=timesteps,
            mask=video_mask,
            sigmas=sigmas
        )
        return loss_dict
    
    def train(self, mode=True):
        self.predictor.train()

    def state_dict_for_save_checkpoint(self, prefix="", keep_vars=False):
        """Customized state_dict"""
        return self.predictor.state_dict(prefix=prefix, keep_vars=keep_vars)
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = False):
        """Customized load."""
        if not isinstance(state_dict, Mapping):
            raise TypeError(f"Expected state_dict to be dict-like, got {type(state_dict)}.")
        
        missing_keys, unexpected_keys = self.predictor.load_state_dict(state_dict, strict)

        if missing_keys is not None:
            logger.info(f"Missing keys in state_dict: {missing_keys}.")
        if unexpected_keys is not None:
            logger.info(f"Unexpected key(s) in state_dict: {unexpected_keys}.")

    def init_cache(self, latents, start_frame_latents, video_mask, prompt, prompt_mask, prompt_2):
        """Initialize cache in the first step."""
        self.cache = {}
        group = mpu.get_tensor_and_context_parallel_group()
        # gather as list
        self.cache = {
            "LATENTS": collect_tensors_across_ranks(latents, group=group),
            "START_FRAME_LATENTS": collect_tensors_across_ranks(start_frame_latents, group=group),
            "VIDEO_MASK": collect_tensors_across_ranks(video_mask, group=group),
            "PROMPT": collect_tensors_across_ranks(prompt, group=group),
            "PROMPT_MASK": collect_tensors_across_ranks(prompt_mask, group=group),
            "PROMPT2": collect_tensors_across_ranks(prompt_2, group=group),
        }

    def get_feature_from_cache(self):
        """Get from the cache"""
        latents = self.cache["LATENTS"][self.index]
        start_frame_latents = self.cache["START_FRAME_LATENTS"][self.index]
        video_mask = self.cache["VIDEO_MASK"][self.index]
        prompt = self.cache["PROMPT"][self.index]
        prompt_mask = self.cache["PROMPT_MASK"][self.index]
        prompt_2 = self.cache["PROMPT2"][self.index]

        self.index += 1
        return latents, start_frame_latents, video_mask, prompt, prompt_mask, prompt_2
