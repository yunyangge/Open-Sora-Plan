import torch
import torch.nn.functional as F


from copy import deepcopy
import numpy as np

from diffusers import FlowMatchEulerDiscreteScheduler


class FlowMatchingLossComputer:
    def __init__(self, scheduler=FlowMatchEulerDiscreteScheduler(), sample_steps=50, max_training_steps=1000, loss_weighting=1.0):
        if isinstance(sample_steps, int):
            self.sample_steps = np.linspace(1, max_training_steps, sample_steps)
        elif isinstance(sample_steps, list):    
            assert max(sample_steps) <= max_training_steps and min(sample_steps) >= 1
            self.sample_steps = sample_steps
        else:
            raise ValueError(f"Invalid sample_steps type: {type(sample_steps)}")
        self.scheduler = deepcopy(scheduler)
        self.max_training_steps = max_training_steps

        if loss_weighting is not None:
            if isinstance(loss_weighting, float):
                self.weighting = torch.stack([loss_weighting] * len(self.sample_steps), dtype=torch.float32)
            elif isinstance(loss_weighting, list):
                self.weighting = torch.stack(loss_weighting, dtype=torch.float32)
            elif isinstance(loss_weighting, torch.Tensor) and loss_weighting.ndim == 1 and loss_weighting.shape[0] == len(self.sample_steps):
                self.weighting = loss_weighting
            else:
                raise ValueError(f"Invalid loss_weighting type: {type(loss_weighting)}")
        else:
            self.weighting = torch.ones(len(self.sample_steps), dtype=torch.float32)

    def _timestep_to_sigma(self, timesteps, n_dim=4, device='cuda', dtype=torch.float32):
        sigmas = self.scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.scheduler.timesteps.to(device=device)
        timesteps = timesteps.to(device=device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def _get_loss(self, model_pred, model_input, mask=None):
        loss = F.mse_loss(model_pred.float(), model_input.float(), reduction="none")
        loss = loss.reshape(loss.shape[0], -1)
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()
        return loss

    def __call__(self, model, model_input, model_kwargs):
        loss = []
        for step in self.sample_steps:
            sigma = self._timestep_to_sigma(step, n_dim=model_input.ndim, device=model_input.device, dtype=model_input.dtype)
            noisy_model_input = (1.0 - sigma) * model_input + sigma * torch.randn_like(model_input)
            model_pred = model(noisy_model_input, step, **model_kwargs)[0]
            loss.append(self._get_loss(model_pred, model_input, mask=model_kwargs.get('attention_mask', None)))
        return loss

        

class ValidationLossProfiler:
    def __init__(
        self, 
        scheduler, 
        vae, 
        text_enc_1, 
        text_enc_2, 
        model, 
        validation_dataloader, 
        validation_dataset=None,
        sample_steps=50,
        max_training_steps=1000,
        device='cuda'
    ):
        self.scheduler = scheduler
        self.vae = vae
        self.text_enc_1 = text_enc_1
        self.text_enc_2 = text_enc_2
        self.model = model

        self.validation_dataloader = validation_dataloader
        self.device = device

        self.data_num = f'{len(validation_dataset)} samples' if validation_dataset is not None else f'{len(validation_dataloader)} batches'

        self.loss_computer = FlowMatchingLossComputer(scheduler=scheduler, sample_steps=sample_steps, max_training_steps=max_training_steps)

    @torch.no_grad()
    def prepare_input(self, data_item):
        x, attn_mask, input_ids_1, cond_mask_1, input_ids_2, cond_mask_2 = data_item
        x = x.to(device=self.device, dtype=self.vae.dtype)
        attn_mask = attn_mask.to(device=self.device)
        input_ids_1 = input_ids_1.to(device=self.device)
        cond_mask_1 = cond_mask_1.to(device=self.device)
        input_ids_2 = input_ids_2.to(device=self.device) if input_ids_2 is not None else None
        cond_mask_2 = cond_mask_2.to(device=self.device) if cond_mask_2 is not None else None
        B, N, L = input_ids_1.shape  # B 1 L
        # use batch inference
        input_ids_1 = input_ids_1.reshape(-1, L)
        cond_mask_1 = cond_mask_1.reshape(-1, L)
        cond_1 = self.text_enc_1(input_ids_1, cond_mask_1)  # B L D
        cond_1 = cond_1.reshape(B, N, L, -1)
        cond_mask_1 = cond_mask_1.reshape(B, N, L)
        if self.text_enc_2 is not None:
            B_, N_, L_ = input_ids_2.shape  # B 1 L
            input_ids_2 = input_ids_2.reshape(-1, L_)
            cond_2 = self.text_enc_2(input_ids_2, cond_mask_2)  # B D
            cond_2 = cond_2.reshape(B_, 1, -1)  # B 1 D
        else:
            cond_2 = None
        x = self.vae.encode(x)  # B C T H W
        return dict(
            hidden_states=x,
            encoder_hidden_states=cond_1,
            attention_mask=attn_mask, 
            encoder_attention_mask=cond_mask_1, 
            pooled_projections=cond_2
        )
            

    def compute_total_loss(self, model, model_kwargs):
        model.eval()
        total_loss = []
        with torch.no_grad():
            for data_item in self.validation_dataloader:
                data = self.prepare_input(data_item)
                model_input = data.pop('hidden_states')
                loss = self.loss_computer(model, model_input, model_kwargs)
                total_loss.append(loss)
        total_loss = [sum(loss) for loss in zip(*total_loss)]
        item_num = len(self.validation_dataloader)
        return [loss / item_num for loss in total_loss]
    
    def __call__(self, model, model_kwargs, output_type='list'):
        total_loss = self.compute_total_loss(model, model_kwargs, output_type=output_type, output_type='list')
        loss_list = [*zip(self.loss_computer.sample_steps, total_loss)]
        loss_list = sorted(loss_list, key=lambda x: x[1])
        print("\n-----------------------------validation loss profile-----------------------------")
        print(f"Data: {self.data_num}")
        print(f"The max loss step: {loss_list[-1][0]}, loss: {loss_list[-1][1]}")
        print(f"The min loss step: {loss_list[0][0]}, loss: {loss_list[0][1]}")
        if len(self.loss_computer.sample_steps) > 5:
            print(f"Top 5 loss steps and their loss: {loss_list[-5:-1]}")
            print(f"Bottom 5 loss steps and their loss: {loss_list[0:5]}")
        print(f"Mean Total Loss:{np.mean(total_loss)}")
        print("-----------------------------------------------------------------------------------\n")
        return total_loss

if __name__ == "__main__":
