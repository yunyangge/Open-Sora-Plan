import os
import numpy as np
from torch import nn
import torch
from einops import rearrange, repeat
from typing import Any, Dict, Optional, Tuple
from torch.nn import functional as F
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import is_torch_version, deprecate
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormSingle
from diffusers.models.embeddings import PixArtAlphaTextProjection
from opensora.models.diffusion.opensora_v1_3.modules import BasicTransformerBlock, Attention
from opensora.models.diffusion.common import PatchEmbed2D
from opensora.utils.utils import to_2tuple
try:
    import torch_npu
    from opensora.npu_config import npu_config
    from opensora.acceleration.parallel_states import get_sequence_parallel_state, hccl_info
except:
    torch_npu = None
    npu_config = None
    from opensora.utils.parallel_states import get_sequence_parallel_state, nccl_info

from opensora.models.diffusion.opensora_v1_3.modeling_opensora import OpenSoraT2V_v1_3 as OpenSoraT2V
import glob

from opensora.utils.custom_logger import get_logger
logger = get_logger(os.path.relpath(__file__))

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class EdgeEmbeddingModule(nn.Module):
    """2D Image to Patch Embedding but with video"""

    def __init__(
        self,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        bias=True,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size), bias=bias
        )

    def forward(self, latent):
        b, _, _, _, _ = latent.shape
        latent = rearrange(latent, 'b c t h w -> (b t) c h w') # [b*1, ae_dim, h, w]
        latent = self.proj(latent)
        latent = rearrange(latent, '(b t) c h w -> b (t h w) c', b=b) # [b, 1*h*w, c]
        return latent


def CalculateWeights(key_idx, frame_num, frame_token_num, weighted_function='hard_function'):
    weights = torch.zeros(1, int(frame_num * frame_token_num), 1) # [b, n, c]

    if weighted_function == 'hard_function':
        weights[:, int(frame_token_num*key_idx.item()): int(frame_token_num*(key_idx.item()+1))] = 1.
    else:
        a_1, b_1 = 1/key_idx, 0
        a_2, b_2 = 1/(key_idx-frame_num), -frame_num/(key_idx-frame_num)
        def y(x, a, b):
            return a*x+b
        
    weights = weights.to(key_idx.device)
    return weights

def reconstitute_checkpoint(pretrained_checkpoint, model_state_dict):
    pretrained_keys = set(list(pretrained_checkpoint.keys()))
    model_keys = set(list(model_state_dict.keys()))
    common_keys = list(pretrained_keys & model_keys)
    checkpoint = {k: pretrained_checkpoint[k] for k in common_keys if model_state_dict[k].numel() == pretrained_checkpoint[k].numel()}
    return checkpoint

class OpenSoraTransition(OpenSoraT2V):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = True,
        sample_size_h: Optional[int] = None,
        sample_size_w: Optional[int] = None,
        sample_size_t: Optional[int] = None,
        patch_size: Optional[int] = None,
        patch_size_t: Optional[int] = None,
        activation_fn: str = "geglu",
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        caption_channels: int = None,
        interpolation_scale_h: float = 1.0,
        interpolation_scale_w: float = 1.0,
        interpolation_scale_t: float = 1.0,
        sparse1d: bool = False,
        sparse_n: int = 2,
        # inpaint
        vae_scale_factor_t: int = 4,
    ):
        super().__init__(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            cross_attention_dim=cross_attention_dim,
            attention_bias=attention_bias,
            sample_size_h=sample_size_h,
            sample_size_w=sample_size_w,
            sample_size_t=sample_size_t,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            activation_fn=activation_fn,
            only_cross_attention=only_cross_attention,
            double_self_attention=double_self_attention,
            upcast_attention=upcast_attention,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            caption_channels=caption_channels,
            interpolation_scale_h=interpolation_scale_h,
            interpolation_scale_w=interpolation_scale_w,
            interpolation_scale_t=interpolation_scale_t,
            sparse1d=sparse1d,
            sparse_n=sparse_n,
        )

        self.vae_scale_factor_t = vae_scale_factor_t
        # init masked_pixel_values and mask conv_in
        self._init_patched_inputs_for_transition()
        self.add_edge_layer_num = num_layers // 4

    def _init_patched_inputs_for_transition(self):

        self.config.sample_size = to_2tuple(self.config.sample_size)

        self.pos_embed_masked_hidden_states = nn.ModuleList(
            [
                PatchEmbed2D(
                    patch_size=self.config.patch_size,
                    in_channels=self.config.in_channels,
                    embed_dim=self.config.hidden_size,
                ),
                zero_module(nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)),
            ]
        )

        self.pos_embed_mask = nn.ModuleList(
            [
                PatchEmbed2D(
                    patch_size=self.config.patch_size,
                    in_channels=self.vae_scale_factor_t,
                    embed_dim=self.config.hidden_size,
                ),
                zero_module(nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)),
            ]
        )

        self.edge_embed_module = nn.ModuleList(
            [
                EdgeEmbeddingModule(
                    patch_size=self.config.patch_size,
                    in_channels=self.config.in_channels,
                    embed_dim=self.config.hidden_size,
                ),
                zero_module(nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)),
            ]
        )

        self.key_caption_projection = PixArtAlphaTextProjection(
            in_features=self.config.caption_channels, hidden_size=self.config.hidden_size
        )

    def _operate_on_patched_inputs(self, hidden_states, encoder_hidden_states, timestep, batch_size, frame, key_frame_edge, key_frame_idx, key_encoder_hidden_states):
        b, c, t, h, w = hidden_states.shape
        assert hidden_states.shape[1] == 2 * self.config.in_channels + self.vae_scale_factor_t
        in_channels = self.config.in_channels

        input_hidden_states, input_masked_hidden_states, input_mask = hidden_states[:, :in_channels], hidden_states[:, in_channels: 2 * in_channels], hidden_states[:, 2 * in_channels:]

        input_hidden_states = self.pos_embed(input_hidden_states.to(self.dtype))

        input_masked_hidden_states = self.pos_embed_masked_hidden_states[0](input_masked_hidden_states.to(self.dtype))
        input_masked_hidden_states = self.pos_embed_masked_hidden_states[1](input_masked_hidden_states)

        input_mask = self.pos_embed_mask[0](input_mask.to(self.dtype))
        input_mask = self.pos_embed_mask[1](input_mask)

        embedded_edge = self.edge_embed_module[0](key_frame_edge.to(self.dtype))
        embedded_edge = self.edge_embed_module[1](embedded_edge) # [b, 1*h*w, c]
        embedded_edge = torch.cat([embedded_edge for i in range(t)], dim=1) # [b, t*h*w, c]

        # print(self.config.patch_size, hidden_states.shape, input_hidden_states.shape, key_frame_edge.shape)
        weights = CalculateWeights(key_frame_idx, t, (h*w)//(self.config.patch_size*self.config.patch_size))
        
        # input_hidden_states: [b, (t, h, w), c]
        hidden_states = input_hidden_states + input_masked_hidden_states + input_mask
        
        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        timestep, embedded_timestep = self.adaln_single(
            timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=self.dtype
        )  # b 6d, b d

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)  # b, 1, l, d or b, 1, l, d
        assert encoder_hidden_states.shape[1] == 1
        encoder_hidden_states = rearrange(encoder_hidden_states, 'b 1 l d -> (b 1) l d')

        key_encoder_hidden_states = self.key_caption_projection(key_encoder_hidden_states)  # b, 1, l, d or b, 1, l, d
        assert key_encoder_hidden_states.shape[1] == 1
        key_encoder_hidden_states = rearrange(key_encoder_hidden_states, 'b 1 l d -> (b 1) l d')

        return hidden_states, encoder_hidden_states, timestep, embedded_timestep, embedded_edge, weights, key_encoder_hidden_states

    def transformer_model_custom_load_state_dict(self, pretrained_model_path):
        pretrained_model_path = os.path.join(pretrained_model_path, 'diffusion_pytorch_model.*')
        pretrained_model_path = glob.glob(pretrained_model_path)
        assert len(pretrained_model_path) > 0, f"Cannot find pretrained model in {pretrained_model_path}"
        pretrained_model_path = pretrained_model_path[0]

        logger.debug(f'Loading {self.__class__.__name__} pretrained weights...')
        logger.debug(f'Loading pretrained model from {pretrained_model_path}...')
        model_state_dict = self.state_dict()
        if 'safetensors' in pretrained_model_path:  # pixart series
            from safetensors.torch import load_file as safe_load
            # import ipdb;ipdb.set_trace()
            pretrained_checkpoint = safe_load(pretrained_model_path, device="cpu")
        else:  # latest stage training weight
            pretrained_checkpoint = torch.load(pretrained_model_path, map_location='cpu')
            if 'model' in pretrained_checkpoint:
                pretrained_checkpoint = pretrained_checkpoint['model']
        checkpoint = reconstitute_checkpoint(pretrained_checkpoint, model_state_dict)
        logger.debug(checkpoint)
        if not 'pos_embed_masked_hidden_states.0.proj' in checkpoint:
            checkpoint['pos_embed_masked_hidden_states.0.proj.weight'] = checkpoint['pos_embed.proj.weight']
            checkpoint['pos_embed_masked_hidden_states.0.proj.bias'] = checkpoint['pos_embed.proj.bias']

        missing_keys, unexpected_keys = self.load_state_dict(checkpoint, strict=False)
        logger.debug(f'missing_keys {len(missing_keys)} {missing_keys}, unexpected_keys {len(unexpected_keys)}')
        logger.debug(f'Successfully load {len(self.state_dict()) - len(missing_keys)}/{len(model_state_dict)} keys from {pretrained_model_path}!')

    def custom_load_state_dict(self, pretrained_model_path):
        assert isinstance(pretrained_model_path, dict), "pretrained_model_path must be a dict"

        pretrained_transformer_model_path = pretrained_model_path.get('transformer_model', None)

        self.transformer_model_custom_load_state_dict(pretrained_transformer_model_path)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        key_encoder_hidden_states: Optional[torch.Tensor] = None,
        key_encoder_attention_mask: Optional[torch.Tensor] = None,
        key_frame_edge: Optional[torch.Tensor] = None,
        key_frame_idx: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs, 
    ):
        batch_size, c, frame, h, w = hidden_states.shape
        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 4:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #   (keep = +0,     discard = -10000.0)
            # b, frame, h, w -> a video
            # b, 1, h, w -> only images
            attention_mask = attention_mask.to(self.dtype)
            attention_mask = attention_mask.unsqueeze(1)
            # attention_mask.shape: [b 1 t h w]
            attention_mask = F.max_pool3d(
                attention_mask, 
                kernel_size=(self.config.patch_size_t, self.config.patch_size, self.config.patch_size), 
                stride=(self.config.patch_size_t, self.config.patch_size, self.config.patch_size)
                )
            attention_mask = rearrange(attention_mask, 'b 1 t h w -> (b 1) 1 (t h w)') 
            attention_mask = (1 - attention_mask.bool().to(self.dtype)) * -10000.0
            # attention_mask: [b, 1, thw]

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 3:  
            # encoder_attention_mask: [b, 1, thw]
            encoder_attention_mask = (1 - encoder_attention_mask.to(self.dtype)) * -10000.0
            key_encoder_attention_mask = (1 - key_encoder_attention_mask.to(self.dtype)) * -10000.0

        # 1. Input
        frame = ((frame - 1) // self.config.patch_size_t + 1) if frame % 2 == 1 else frame // self.config.patch_size_t  # patchfy
        height, width = hidden_states.shape[-2] // self.config.patch_size, hidden_states.shape[-1] // self.config.patch_size

        hidden_states, encoder_hidden_states, timestep, embedded_timestep, embedded_edge, weights, key_encoder_hidden_states = self._operate_on_patched_inputs(
            hidden_states, encoder_hidden_states, timestep, batch_size, frame, key_frame_edge, key_frame_idx, key_encoder_hidden_states
        )
        # embedded_edge: [b, thw, dim], weights: [b, thw, 1]
        
        weighted_embedded_edge = embedded_edge * weights
        
        # To
        # x            (t*h*w b d) or (t//sp*h*w b d)
        # cond_1       (l b d) or (l//sp b d)
        hidden_states = rearrange(hidden_states, 'b s h -> s b h', b=batch_size).contiguous()
        encoder_hidden_states = rearrange(encoder_hidden_states, 'b s h -> s b h', b=batch_size).contiguous()
        key_encoder_hidden_states = rearrange(key_encoder_hidden_states, 'b s h -> s b h', b=batch_size).contiguous()
        weighted_embedded_edge = rearrange(weighted_embedded_edge, 'b s h -> s b h', b=batch_size).contiguous()
        timestep = timestep.view(batch_size, 6, -1).transpose(0, 1).contiguous()

        sparse_mask = {}
        if npu_config is None:
            if get_sequence_parallel_state():
                head_num = self.config.num_attention_heads // nccl_info.world_size
            else:
                head_num = self.config.num_attention_heads
        else:
            head_num = None
        
        for sparse_n in [1, 4]:
            sparse_mask[sparse_n] = Attention.prepare_sparse_mask(attention_mask, encoder_attention_mask, key_encoder_attention_mask, sparse_n, head_num)
        
        # 2. Blocks
        for i, block in enumerate(self.transformer_blocks):
            if i > 1 and i < 30:
                attention_mask, encoder_attention_mask, key_encoder_attention_mask = sparse_mask[block.attn1.processor.sparse_n][block.attn1.processor.sparse_group]
            else:
                attention_mask, encoder_attention_mask, key_encoder_attention_mask = sparse_mask[1][block.attn1.processor.sparse_group]
            
            # self.training = true, self.gradient_checkpointing = true
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    frame, 
                    height, 
                    width, 
                    key_encoder_hidden_states,
                    key_encoder_attention_mask,
                    weighted_embedded_edge,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    frame=frame, 
                    height=height, 
                    width=width, 
                    key_encoder_hidden_states=key_encoder_hidden_states,
                    key_encoder_attention_mask=key_encoder_attention_mask,
                    weighted_embedded_edge=weighted_embedded_edge,
                )

        # To (b, t*h*w, h) or (b, t//sp*h*w, h)
        hidden_states = rearrange(hidden_states, 's b h -> b s h', b=batch_size).contiguous()

        # 3. Output
        output = self._get_output_for_patched_inputs(
            hidden_states=hidden_states,
            timestep=timestep,
            embedded_timestep=embedded_timestep,
            num_frames=frame, 
            height=height,
            width=width,
        )  # b c t h w

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
    
def OpenSoraTransition_2B_122(**kwargs):
   return OpenSoraTransition(
        num_layers=32, attention_head_dim=96, num_attention_heads=24, patch_size_t=1, patch_size=2,
        caption_channels=4096, cross_attention_dim=2304, activation_fn="gelu-approximate", **kwargs
    )

OpenSoraTransition_models = {
    "OpenSoraTransition-2B/122": OpenSoraTransition_2B_122,  # 2.7B
}

OpenSoraTransition_models_class = {
    "OpenSoraTransition-2B/122": OpenSoraTransition,
}
