import numpy as np

from diffusers import FlowMatchEulerDiscreteScheduler

def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)  # 0.001, 1
    schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)  # 1, 1000
    timesteps = timesteps.to(accelerator.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma

class FlowMatchingLossComputer:
    def __init__(self, training_scheduler=sample_steps=50, max_training_steps=1000):
        if isinstance(sample_steps, int):
            self.sample_steps = np.linspace(1, max_training_steps, sample_steps)
        elif isinstance(sample_steps, list):    
            self.sample_steps = sample_steps
        else:
            self.sample_steps = None
        self.max_training_steps = max_training_steps

    def _timestep_to_sigma(self, sigma):


    def __call__(self, model, model_input, model_kwargs):
        loss = 0.0
        for step in self.sample_steps:
            noisy_model_input = 
            model_output = model(model_input, step, **model_kwargs)
            loss += self.loss(model_output, model_input)

class ValidationLossProfiler:
    def __init__(self, scheduler, validation_loader, loss_computer):
        self.scheduler = scheduler
        self.validation_loader = validation_loader
        self.loss_computer = loss_computer


    