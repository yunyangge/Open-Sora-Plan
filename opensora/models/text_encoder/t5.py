import torch
from torch import nn
from transformers import T5EncoderModel

try:
    import torch_npu
except:
    torch_npu = None
from opensora.utils.custom_logger import get_logger
import os
logger = get_logger(os.path.relpath(__file__))

class T5Wrapper(nn.Module):
    def __init__(self, args, **kwargs):
        super(T5Wrapper, self).__init__()
        self.model_name = args.text_encoder_name_1
        if torch_npu is not None:
            self.model_name = '/home/save_dir/pretrained/mt5-xxl'
        logger.debug(f'Loading T5 model from {self.model_name}...')
        self.text_enc = T5EncoderModel.from_pretrained(self.model_name, cache_dir=args.cache_dir, **kwargs).eval()

    def forward(self, input_ids, attention_mask):
        text_encoder_embs = self.text_enc(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']
        return text_encoder_embs.detach()

