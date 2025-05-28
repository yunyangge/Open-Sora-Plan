from selectors import EpollSelector
from typing import Any, Dict, Optional, Tuple, List

from einops import rearrange, repeat
import torch
from torch import nn
import torch.nn.functional as F
from diffusers.utils import is_torch_version
from megatron.legacy.model.rms_norm import RMSNorm
from megatron.legacy.model.layer_norm import LayerNorm
from megatron.core import mpu, tensor_parallel
from megatron.training import get_args
from megatron.training.utils import print_rank_0 as print

from mindspeed_mm.models.common import MultiModalModule
from mindspeed_mm.models.common.embeddings import PatchEmbed2D, RoPE3D, PositionGetter3D, apply_rotary_emb
from mindspeed_mm.models.common.ffn import FeedForward
from mindspeed_mm.models.common.attention import MultiHeadSparseMMAttentionSBH
from mindspeed_mm.models.common.normalize import normalize
from mindspeed_mm.models.common.communications import split_forward_gather_backward, gather_forward_split_backward

from mindspeed_mm.models.predictor.dits.modules import CombinedTimestepTextProjEmbeddings, AdaNorm, OpenSoraNormZero
from mindspeed_mm.models.predictor.dits.sparseu_mmdit import SparseUMMDiT, SparseMMDiTBlock

selective_recom = True
recom_ffn_layers = 32

def zero_initialize(module):
    for param in module.parameters():
        nn.init.zeros_(param)
    return module

class ResI2VMMDiT(SparseUMMDiT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_frame_patch_embed = nn.Sequential(
            PatchEmbed2D(
                patch_size=self.patch_size,
                in_channels=self.in_channels,
                embed_dim=self.hidden_size,
            ),
            zero_initialize(nn.Linear(self.hidden_size, self.hidden_size))
        )

    def _operate_on_patched_inputs(self, hidden_states, encoder_hidden_states, timestep, pooled_projections):
        hidden_states, start_frame = hidden_states[:, :self.in_channels], hidden_states[:, self.in_channels:]
        hidden_states = self.patch_embed(hidden_states.to(self.dtype)) + self.start_frame_patch_embed(start_frame.to(self.dtype))
        if pooled_projections.shape[1] != 1:
            raise AssertionError("Pooled projection should have shape (b, 1, 1, d)")
        pooled_projections = pooled_projections.squeeze(1)  # b 1 1 d -> b 1 d
        timesteps_emb = self.time_text_embed(timestep, pooled_projections)  # (N, D)
            
        encoder_hidden_states = self.caption_projection(encoder_hidden_states)  # b, 1, l, d
        if encoder_hidden_states.shape[1] != 1:
            raise AssertionError("Encoder hidden states should have shape (b, 1, l, d)")
        encoder_hidden_states = encoder_hidden_states.squeeze(1)

        return hidden_states, encoder_hidden_states, timesteps_emb
