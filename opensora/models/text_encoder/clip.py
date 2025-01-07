import torch
from torch import nn
from transformers import CLIPTextModelWithProjection

try:
    import torch_npu
except:
    torch_npu = None

class CLIPWrapper(nn.Module):
    def __init__(self, args, **kwargs):
        super(CLIPWrapper, self).__init__()
        self.model_name = args.text_encoder_name_2
        if torch_npu is not None:
            self.model_name = '/home/ma-user/work/checkpoint/pretrained/clip'
        else:
            self.model_name = '/storage/cache_dir/CLIP-ViT-bigG-14-laion2B-39B-b160k'
        print(f'Loading CLIP model from {self.model_name}...')
        self.text_enc = CLIPTextModelWithProjection.from_pretrained(self.model_name, cache_dir=args.cache_dir, **kwargs).eval()

    def forward(self, input_ids, attention_mask): 
        text_encoder_embs = self.text_enc(input_ids=input_ids, output_hidden_states=True)[0]
        return text_encoder_embs.detach()
