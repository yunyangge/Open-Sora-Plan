import torch
import torch.nn.functional as F
import math
from diffusers.utils.torch_utils import randn_tensor

from opensora.sample.pipeline_inpaint import OpenSoraInpaintPipeline
from einops import rearrange, repeat


class OpenSora4DPipeline(OpenSoraInpaintPipeline):

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            math.ceil((int(num_frames))), 
            math.ceil(int(height) / self.vae.vae_scale_factor[1]),
            math.ceil(int(width) / self.vae.vae_scale_factor[2]),
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        return latents

    def prepare_mask_masked_video(
        self, 
        conditional_images, 
        conditional_images_indices, 
        num_frames, 
        batch_size,
        height,
        width,
        num_images_per_prompt=1, 
        use_vae_preprocessed_mask=False, 
        do_classifier_free_guidance=True,
        device="cuda"
    ):
        
        # NOTE inpaint
        assert isinstance(conditional_images_indices, list) and len(conditional_images_indices) == len(conditional_images) and isinstance(conditional_images_indices[0], int), "conditional_images_indices should be a list of int" 
        if isinstance(conditional_images, list) and isinstance(conditional_images[0], torch.Tensor):
            if len(conditional_images[0].shape) == 3:
                conditional_images = [conditional_image.unsqueeze(1) for conditional_image in conditional_images] # C H W -> C 1 H W
            elif len(conditional_images[0].shape) == 4:
                conditional_images = [conditional_image.transpose(0, 1) for conditional_image in conditional_images] # 1 C H W -> C 1 H W
            conditional_images = torch.cat(conditional_images, dim=1).to(device=device) # C F H W
        elif isinstance(conditional_images, torch.Tensor):
            assert len(conditional_images.shape) == 4, "The shape of conditional_images should be a tensor with 4 dim"
            conditional_images = conditional_images.transpose(0, 1) # F C H W -> C F H W
            conditional_images = conditional_images.to(device=device)
        else:
            raise ValueError("conditional_images should be a list of torch.Tensor")

        input_video = torch.zeros([3, num_frames, height, width], dtype=self.vae.vae.dtype, device=device)
        input_video[:, conditional_images_indices] = conditional_images.to(input_video.dtype)

        print(f"conditional_images_indices: {conditional_images_indices}")

        input_video = input_video.unsqueeze(0).repeat(batch_size * num_images_per_prompt, 1, 1, 1, 1)
        
        # default mode
        if not use_vae_preprocessed_mask:
            B, C, T, H, W = input_video.shape
            mask = torch.ones([B, 1, T, H, W], device=device)
            mask[:, :, conditional_images_indices] = 0
            masked_video = input_video * (mask < 0.5) 

            masked_video = rearrange(masked_video, 'b c t h w -> (b t) c 1 h w')
            masked_video = self.vae.encode(masked_video).to(device)
            masked_video = rearrange(masked_video, '(b t) c 1 h w -> b c t h w', t=T, b=B)

            mask = rearrange(mask, 'b c t h w -> (b c t) 1 h w')
            latent_size = (height // self.vae.vae_scale_factor[1], width // self.vae.vae_scale_factor[2])
            mask = F.interpolate(mask, size=latent_size, mode='bilinear')
            mask = rearrange(mask, '(b c t) 1 h w -> b c t h w', t=T, b=B)
            mask = repeat(mask, 'b c t h w -> b (k c) t h w', k=self.vae.vae_scale_factor[0])

        else:
            mask = torch.ones_like(input_video, device=device)
            mask[:, :, conditional_images_indices] = 0
            masked_video = input_video * (mask < 0.5) 
            masked_video = self.vae.encode(masked_video).to(device)

            mask = self.vae.encode(mask).to(device)
        

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_video = torch.cat([masked_video] * 2) if do_classifier_free_guidance else masked_video

        return mask, masked_video
    
    def decode_latents(self, latents):
        # print(f'before vae decode', torch.max(latents).item(), torch.min(latents).item(), torch.mean(latents).item(), torch.std(latents).item())
        bs = latents.shape[0]
        latents = rearrange(latents, 'b c t h w -> (b t) c 1 h w')
        video = self.vae.decode(latents.to(self.vae.vae.dtype))
        video = rearrange(video, '(b t) 1 c h w -> b t c h w', b=bs)
        # print(f'after vae decode', torch.max(video).item(), torch.min(video).item(), torch.mean(video).item(), torch.std(video).item())
        video = ((video / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().permute(0, 1, 3, 4, 2).contiguous() # b t h w c
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        return video




