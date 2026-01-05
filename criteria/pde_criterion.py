"""
PDE/Stein Residual Criterion

This criterion evaluates images based on the PDE residual of the diffusion score.
Based on the Stein identity: E[div(f) + <f, score>] = 0 for score matching.

The statistic per image is:
    r(z_t, t) = div(f) + <f(z_t), s_theta(z_t, t)>

where:
    - f = ∇g, g(z) = ||Δz||² (Δ: 2D Laplacian per-channel)
    - f(z) = 2 Δ² z
    - div(f) = trace(Hess g) = 40 * (C*H*W) for Laplacian kernel [[0,1,0],[1,-4,1],[0,1,0]]
    - Score from SD: s_theta = -eps_pred / sigma_t
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional

from criteria.base import BaseCriterion, register_criterion


@register_criterion("pde")
class PDECriterion(BaseCriterion):
    """
    PDE/Stein residual criterion for diffusion models.
    
    Lower residual values indicate better alignment with the learned distribution.
    """
    
    def __init__(self, config, model_manager):
        super().__init__(config, model_manager)
        
        # Laplacian convolution kernel (depthwise, per-channel)
        self._lap_kernel = torch.tensor(
            [[0.0, 1.0, 0.0],
             [1.0, -4.0, 1.0],
             [0.0, 1.0, 0.0]], 
            device=self.device, 
            dtype=torch.float32
        ).view(1, 1, 3, 3)
    
    def laplacian(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Laplacian operator per channel.
        
        Args:
            x: Tensor of shape (B, C, H, W)
            
        Returns:
            Laplacian of x with same shape
        """
        B, C, H, W = x.shape
        kernel = self._lap_kernel.repeat(C, 1, 1, 1)  # (C, 1, 3, 3) depthwise
        return F.conv2d(x, kernel, padding=1, groups=C)
    
    def evaluate_batch(
        self,
        images: torch.Tensor,
        prompts: List[str],
        images_raw: Optional[List[torch.Tensor]] = None
    ) -> List[Dict]:
        """
        Evaluate images using PDE residual criterion.
        
        The algorithm:
        1. Encode images to latent space
        2. Add noise at timestep t
        3. Predict noise with UNet
        4. Compute score from noise prediction
        5. Compute f = 2Δ²z and div(f)
        6. Compute residual r = div(f) + <f, score>
        
        Args:
            images: Preprocessed images (B, C, H, W) in [-1, 1]
            prompts: Text prompts for each image
            images_raw: Not used in this criterion
            
        Returns:
            List of result dicts with 'criterion' key
        """
        num_images = images.shape[0]
        num_noise = self.config.num_noise
        
        # Get model components
        unet = self.model_manager.unet
        scheduler = self.model_manager.scheduler
        
        # Compute text embeddings (repeated for noise samples)
        text_emb = self.compute_text_embeddings(prompts, num_noise)
        
        # Encode images to latent space
        z0 = self.encode_images_to_latents(images)
        
        # Repeat latents for multiple noise samples
        z0 = z0.repeat_interleave(num_noise, dim=0)
        if z0.dtype != torch.float16:
            z0 = z0.half()
        
        # Sample spherical noise
        eps = torch.randn_like(z0, device=self.device, dtype=torch.float16)
        eps = self.normalize_batch(eps, self.config.epsilon_reg)
        
        # Scale noise by sqrt(dimension)
        sqrt_d = torch.prod(torch.tensor(z0.shape[1:], device=self.device)).float().sqrt()
        eps = eps * sqrt_d
        
        # Get timestep
        t = int(self.config.time_frac * scheduler.config.num_train_timesteps)
        t_tensor = torch.full((z0.shape[0],), t, device=self.device, dtype=torch.long)
        
        # Add noise to get z_t
        zt = scheduler.add_noise(z0, eps, t_tensor)
        if zt.dtype != torch.float16:
            zt = zt.half()
        
        # Predict noise with UNet
        with torch.no_grad():
            eps_pred = unet(zt, t_tensor, encoder_hidden_states=text_emb)[0]
        
        if eps_pred.dtype != torch.float16:
            eps_pred = eps_pred.half()
        
        # Compute score: s = -eps_pred / sigma_t
        alpha_bar = scheduler.alphas_cumprod[t].to(self.device).float()
        sigma_t = torch.sqrt(1.0 - alpha_bar).clamp_min(1e-8)
        score = -(eps_pred.float() / sigma_t).half()
        
        # Compute f = 2 * Δ² z (in float32 for numerical stability)
        zt_f = zt.float()
        lap1 = self.laplacian(zt_f)
        lap2 = self.laplacian(lap1)
        f_field = (2.0 * lap2).half()
        
        # Compute divergence (constant for this f)
        _, C, H, W = zt.shape
        div_f = 40.0 * float(C * H * W)
        
        # Compute residual: r = div(f) + <f, score>
        dot = (f_field * score).float().sum(dim=(1, 2, 3))
        r = dot + div_f
        
        # Aggregate per image (mean over noise samples)
        r = r.view(num_images, num_noise)
        stat = r.abs().mean(dim=1)
        
        # Build results
        results = []
        for i in range(num_images):
            result = {"criterion": float(stat[i].item())}
            
            if self.config.return_terms:
                result.update({
                    "residual_abs_mean": float(stat[i].item()),
                    "div_f_const": float(div_f),
                    "dot_mean": float(r[i].mean().item()),
                    "residual_std": float(r[i].std().item()),
                })
            
            results.append(result)
        
        return results
