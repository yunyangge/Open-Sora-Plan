
from typing import List, Optional, Union
from tqdm import tqdm

import torch
import torch.nn.functional as F

try: 
    import torch_npu
    from npu_config import npu_config
except:
    torch_npu = None
    npu_config = None

from opensora.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerScheduler
from opensora.schedulers.inversion import flow_matching_inversion

class InversionEvaluationPipeline:
    def __init__(
        self,
        model,
        num_inference_steps: int,
        num_inversion_steps: Union[int, List[int]],
        sigmas: Optional[List[torch.Tensor]] = None,
        do_classifier_free_guidance: bool = True,
        guidance_scale: float = 7.0,
        loss_fn = F.mse_loss,
    ):
        self.model = model
        self.num_inference_steps = num_inference_steps

        if isinstance(num_inversion_steps, int):
            num_inversion_steps = [num_inversion_steps]

        self.num_inversion_steps = num_inversion_steps

        self.scheduler = FlowMatchEulerScheduler()

        self.sigmas = sigmas

        self.do_classifier_free_guidance = do_classifier_free_guidance
        self.guidance_scale = guidance_scale

        self.loss_fn = loss_fn

    def batch_eval(
        self,
        latents,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        pooled_projections,
        **kwargs,
    ):
        
        validation_losses = []

        for num_inv_steps in tqdm(self.num_inversion_steps, desc="Inversion Evaluation"):
            resampled_latents = flow_matching_inversion(
                model=self.model,
                scheduler=self.scheduler,
                latents=latents,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                pooled_projections=pooled_projections,
                sigmas=self.sigmas,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inversion_steps,
                num_inverse_steps=num_inv_steps,
            )
            validation_loss = self.loss_fn(resampled_latents, latents)
            validation_losses.append(validation_loss)

        return validation_losses
    
    



