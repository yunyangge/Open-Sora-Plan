from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config

from diffusers import DDPMScheduler, EulerAncestralDiscreteScheduler

def get_opensoraplan_scheduler(scheduler, snr_shift_scale=2.0):
    print(f"Using {scheduler} scheduler")
    print(f"snr_shift_scale: {snr_shift_scale}")
    if scheduler == DDPMScheduler or scheduler == 'ddpm':
        from diffusers.schedulers.scheduling_ddpm import betas_for_alpha_bar, rescale_zero_terminal_snr
        class OpenSoraPlanScheduler(DDPMScheduler):
            @register_to_config
            def __init__(
                self,
                num_train_timesteps: int = 1000,
                beta_start: float = 0.0001,
                beta_end: float = 0.02,
                beta_schedule: str = "linear",
                trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
                variance_type: str = "fixed_small",
                clip_sample: bool = True,
                prediction_type: str = "epsilon",
                thresholding: bool = False,
                dynamic_thresholding_ratio: float = 0.995,
                clip_sample_range: float = 1.0,
                sample_max_value: float = 1.0,
                timestep_spacing: str = "leading",
                steps_offset: int = 0,
                rescale_betas_zero_snr: int = False,
            ):
                if trained_betas is not None:
                    self.betas = torch.tensor(trained_betas, dtype=torch.float32)
                elif beta_schedule == "linear":
                    self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
                elif beta_schedule == "scaled_linear":
                    # this schedule is very specific to the latent diffusion model.
                    self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
                elif beta_schedule == "squaredcos_cap_v2":
                    # Glide cosine schedule
                    self.betas = betas_for_alpha_bar(num_train_timesteps)
                elif beta_schedule == "sigmoid":
                    # GeoDiff sigmoid schedule
                    betas = torch.linspace(-6, 6, num_train_timesteps)
                    self.betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
                else:
                    raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

                # Rescale for zero SNR
                if rescale_betas_zero_snr:
                    self.betas = rescale_zero_terminal_snr(self.betas)

                self.alphas = 1.0 - self.betas
                self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

                self.alphas_cumprod = self.alphas_cumprod / (snr_shift_scale + (1 - snr_shift_scale) * self.alphas_cumprod)

                self.one = torch.tensor(1.0)

                # standard deviation of the initial noise distribution
                self.init_noise_sigma = 1.0

                # setable values
                self.custom_timesteps = False
                self.num_inference_steps = None
                self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())

                self.variance_type = variance_type

    if scheduler == EulerAncestralDiscreteScheduler or scheduler == 'EulerAncestralDiscrete':
        from diffusers.schedulers.scheduling_euler_ancestral_discrete import betas_for_alpha_bar, rescale_zero_terminal_snr
        class OpenSoraPlanScheduler(EulerAncestralDiscreteScheduler):
            @register_to_config
            def __init__(
                self,
                num_train_timesteps: int = 1000,
                beta_start: float = 0.0001,
                beta_end: float = 0.02,
                beta_schedule: str = "linear",
                trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
                prediction_type: str = "epsilon",
                timestep_spacing: str = "linspace",
                steps_offset: int = 0,
                rescale_betas_zero_snr: bool = False,
            ):
                if trained_betas is not None:
                    self.betas = torch.tensor(trained_betas, dtype=torch.float32)
                elif beta_schedule == "linear":
                    self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
                elif beta_schedule == "scaled_linear":
                    # this schedule is very specific to the latent diffusion model.
                    self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
                elif beta_schedule == "squaredcos_cap_v2":
                    # Glide cosine schedule
                    self.betas = betas_for_alpha_bar(num_train_timesteps)
                else:
                    raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

                if rescale_betas_zero_snr:
                    self.betas = rescale_zero_terminal_snr(self.betas)

                self.alphas = 1.0 - self.betas
                self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

                self.alphas_cumprod = self.alphas_cumprod / (snr_shift_scale + (1 - snr_shift_scale) * self.alphas_cumprod)

                if rescale_betas_zero_snr:
                    # Close to 0 without being 0 so first sigma is not inf
                    # FP16 smallest positive subnormal works well here
                    self.alphas_cumprod[-1] = 2**-24

                sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
                sigmas = np.concatenate([sigmas[::-1], [0.0]]).astype(np.float32)
                self.sigmas = torch.from_numpy(sigmas)

                # setable values
                self.num_inference_steps = None
                timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=float)[::-1].copy()
                self.timesteps = torch.from_numpy(timesteps)
                self.is_scale_input_called = False

                self._step_index = None
                self._begin_index = None
                self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication
    
    else:
        raise NotImplementedError(f"{scheduler} is not implemented for OpenSoraPlanScheduler")
    return OpenSoraPlanScheduler