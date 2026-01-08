"""
Latent Space Criterion (方案C)

核心改动：把原文公式从 CLIP 空间搬到 latent 空间

原文：在 CLIP 空间计算 criterion
改进：在 latent 空间直接计算，不经过 CLIP

理论依据 (PDE 视角)：
- 噪声预测 h ≈ score function = ∇log p
- Score 是 Fokker-Planck 方程的解的梯度场
- 在 latent 空间直接操作，保留 PDE 解的完整信息
- CLIP 映射会损失高频/高阶信息
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional

from criteria.base import BaseCriterion, register_criterion


@register_criterion("latent")
class LatentSpaceCriterion(BaseCriterion):
    """
    Latent Space Criterion
    
    在 latent 空间直接计算 manifold criterion，不经过 CLIP。
    
    公式 (与原文相同，只是空间不同)：
        C(x₀) = ⟨-h/||h||, a·u_d - b·h + c·√d·z₀⟩
    
    其中所有量都在 latent 空间：
        - z₀: 原图的 latent
        - h: 噪声预测 (latent 空间)
        - u_d: 球面扰动 (latent 空间)
    """
    
    def __init__(self, config, model_manager):
        super().__init__(config, model_manager)
        
        # Criterion coefficients (与原文一致)
        self.a = 1.0
        self.b = 1.0
        self.c = 1.0
    
    def evaluate_batch(
        self,
        images: torch.Tensor,
        prompts: List[str],
        images_raw: Optional[List[torch.Tensor]] = None
    ) -> List[Dict]:
        """在 latent 空间计算 criterion"""
        
        num_images = images.shape[0]
        num_noise = self.config.num_noise
        
        # Get models
        unet = self.model_manager.unet
        vae = self.model_manager.vae
        scheduler = self.model_manager.scheduler
        tokenizer = self.model_manager.tokenizer
        text_encoder = self.model_manager.text_encoder
        
        # --------- Text embeddings ---------
        expanded_prompts = []
        for prompt in prompts:
            expanded_prompts.extend([prompt] * num_noise)
        
        text_tokens = tokenizer(
            expanded_prompts, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt"
        )
        text_emb = text_encoder(text_tokens.input_ids.to(self.device)).last_hidden_state
        
        # --------- Encode images to latent ---------
        with torch.no_grad():
            z0 = vae.encode(images).latent_dist.sample()
            z0 = z0 * vae.config.scaling_factor
        
        # z0: (B, 4, H/8, W/8)
        latent_dim = z0.shape[1] * z0.shape[2] * z0.shape[3]
        sqrt_d = np.sqrt(latent_dim)
        
        # Repeat for multiple noise samples
        z0_repeated = z0.repeat_interleave(num_noise, dim=0).half()
        
        # --------- Spherical noise ---------
        gauss_noise = torch.randn_like(z0_repeated, device=self.device).half()
        u_d = self.normalize_batch(gauss_noise, self.config.epsilon_reg).half()
        u_d = u_d * sqrt_d  # Scale to sphere of radius √d
        
        # --------- Add noise (spherical perturbation) ---------
        # x̃ = √(1-α)·z₀ + √α·u_d
        timestep = self.config.time_frac * scheduler.config.num_train_timesteps
        timestep_tensor = torch.full((z0_repeated.shape[0],), timestep, 
                                     device=self.device, dtype=torch.long)
        
        noisy_latents = scheduler.add_noise(z0_repeated, u_d, timestep_tensor).half()
        
        # --------- Predict noise (score) ---------
        with torch.no_grad():
            h = unet(noisy_latents, timestep_tensor, encoder_hidden_states=text_emb)[0]
        
        del noisy_latents, text_emb
        
        # --------- Compute criterion in latent space ---------
        results = []
        
        for i in range(num_images):
            # Get slices for this image
            start_idx = i * num_noise
            end_idx = (i + 1) * num_noise
            
            z0_i = z0[i:i+1]  # (1, 4, H, W)
            h_i = h[start_idx:end_idx]  # (num_noise, 4, H, W)
            u_d_i = u_d[start_idx:end_idx]  # (num_noise, 4, H, W)
            
            # Flatten to vectors
            z0_flat = z0_i.flatten(1)  # (1, d)
            h_flat = h_i.flatten(1)  # (num_noise, d)
            u_d_flat = u_d_i.flatten(1)  # (num_noise, d)
            
            # Normalize h: -h / ||h||
            h_norms = torch.norm(h_flat, p=2, dim=1, keepdim=True) + 1e-8
            h_normalized = -h_flat / h_norms
            
            # Expand z0
            z0_expanded = z0_flat.expand(num_noise, -1)  # (num_noise, d)
            
            # Combined term: a·u_d - b·h + c·√d·z₀
            combined = (self.a * u_d_flat - 
                       self.b * h_flat + 
                       self.c * sqrt_d * z0_expanded)
            
            # Inner product (cosine similarity)
            cos_sim = F.cosine_similarity(h_normalized, combined, dim=1)
            
            # Average over noise samples
            C = cos_sim.mean().item()
            
            # Normalize to [0, 1]
            C_normalized = (C + 1) / (self.a + self.b + self.c + 1)
            
            result = {"criterion": float(C_normalized)}
            
            if self.config.return_terms:
                # Additional diagnostics
                h_norm_mean = h_norms.mean().item()
                u_d_norm_mean = torch.norm(u_d_flat, p=2, dim=1).mean().item()
                
                result.update({
                    "h_norm": float(h_norm_mean),
                    "u_d_norm": float(u_d_norm_mean),
                    "cos_sim_mean": float(C),
                    "cos_sim_std": float(cos_sim.std().item()),
                })
            
            results.append(result)
        
        return results