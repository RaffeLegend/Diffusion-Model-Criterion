"""
PDE/Stein Residual Criterion - 多种 Scheme 实现

包含以下方法:
1. RMSE: 简单的噪声预测误差 (AUC ~73%)
2. Stein-A: 原始 Stein 残差 with Laplacian test function
3. Stein-B: 简化的 Stein with identity test function  
4. Combined: RMSE + Stein 组合
5. Score-Norm: Score 范数统计
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional

from criteria.base import BaseCriterion, register_criterion


@register_criterion("pde")
class PDECriterion(BaseCriterion):
    """
    PDE/Stein residual criterion with multiple schemes.
    
    Config options:
        - scheme: "rmse", "stein_lap", "stein_simple", "combined", "score_norm"
        - combine_alpha: weight for RMSE in combined scheme (default 0.5)
    """
    
    def __init__(self, config, model_manager):
        super().__init__(config, model_manager)
        
        # Laplacian kernel
        self._lap_kernel = torch.tensor(
            [[0.0, 1.0, 0.0],
             [1.0, -4.0, 1.0],
             [0.0, 1.0, 0.0]], 
            device=self.device, 
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        
        # 默认 scheme
        self.scheme = getattr(config, 'scheme', 'combined')
        self.combine_alpha = getattr(config, 'combine_alpha', 0.5)
    
    def laplacian(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Laplacian operator per channel."""
        C = x.shape[1]
        kernel = self._lap_kernel.repeat(C, 1, 1, 1)
        return F.conv2d(x, kernel, padding=1, groups=C)
    
    def compute_rmse(self, eps_pred: torch.Tensor, eps: torch.Tensor, 
                     num_images: int, num_noise: int) -> torch.Tensor:
        """
        Scheme 1: 简单的噪声预测 RMSE
        
        理论: eps_pred 应该准确预测注入的噪声 eps
        对于 OOD 图片，预测误差更大
        """
        err = (eps_pred.float() - eps.float())
        per_sample = err.pow(2).mean(dim=(1, 2, 3)).sqrt()  # RMSE
        per_sample = per_sample.view(num_images, num_noise)
        return per_sample.mean(dim=1)
    
    def compute_stein_laplacian(self, zt: torch.Tensor, score: torch.Tensor,
                                 num_images: int, num_noise: int) -> torch.Tensor:
        """
        Scheme 2: 原始 Stein 残差 with Laplacian test function
        
        r = div(f) + <f, score>
        where f = 2Δ²z, div(f) = 40 * C * H * W
        """
        zt_f = zt.float()
        lap1 = self.laplacian(zt_f)
        lap2 = self.laplacian(lap1)
        f_field = (2.0 * lap2).half()
        
        _, C, H, W = zt.shape
        div_f = 40.0 * float(C * H * W)
        
        dot = (f_field.float() * score.float()).sum(dim=(1, 2, 3))
        r = dot + div_f
        
        r = r.view(num_images, num_noise)
        return r.abs().mean(dim=1)
    
    def compute_stein_simple(self, zt: torch.Tensor, score: torch.Tensor,
                              num_images: int, num_noise: int) -> torch.Tensor:
        """
        Scheme 3: 简化的 Stein with identity-like test function
        
        使用 f = z (或 f = 2z)
        div(f) = d (维度)
        r = d + <z, score>
        
        理论上，对于真实分布 p, E[<z, score>] = -d (by Stein identity with f=z)
        所以 E[r] = 0
        """
        f_field = zt.float()  # f = z
        
        _, C, H, W = zt.shape
        div_f = float(C * H * W)  # trace(I) = d
        
        dot = (f_field * score.float()).sum(dim=(1, 2, 3))
        r = dot + div_f
        
        r = r.view(num_images, num_noise)
        return r.abs().mean(dim=1)
    
    def compute_score_norm(self, score: torch.Tensor,
                           num_images: int, num_noise: int) -> torch.Tensor:
        """
        Scheme 4: Score 范数
        
        对于 OOD 图片，score 的范数可能异常（过大或过小）
        """
        score_norm = score.float().pow(2).mean(dim=(1, 2, 3)).sqrt()
        score_norm = score_norm.view(num_images, num_noise)
        return score_norm.mean(dim=1)
    
    def compute_prediction_error_weighted(self, eps_pred: torch.Tensor, eps: torch.Tensor,
                                          zt: torch.Tensor, sigma_t: float,
                                          num_images: int, num_noise: int) -> torch.Tensor:
        """
        Scheme 5: 加权预测误差
        
        使用 Laplacian 加权的预测误差，强调高频区域
        """
        err = (eps_pred.float() - eps.float())
        
        # Laplacian 加权
        zt_f = zt.float()
        lap = self.laplacian(zt_f)
        weight = lap.abs() + 1.0  # 避免零权重
        
        weighted_err = (err.pow(2) * weight).mean(dim=(1, 2, 3)).sqrt()
        weighted_err = weighted_err.view(num_images, num_noise)
        
        return weighted_err.mean(dim=1)
    
    def compute_stein_noise_domain(self, eps_pred: torch.Tensor, eps: torch.Tensor,
                                    zt: torch.Tensor, sigma_t: float,
                                    num_images: int, num_noise: int,
                                    K: int = 4) -> torch.Tensor:
        """
        Scheme 6: Noise domain Stein residual (修复版)
        
        在噪声域计算 Stein 残差
        """
        # f = 2 * Δ² z_t
        zt_f32 = zt.detach().float().requires_grad_(True)
        lap1 = self.laplacian(zt_f32)
        lap2 = self.laplacian(lap1)
        f = 2.0 * lap2
        
        # Hutchinson trace estimator for div(f)
        div_est = torch.zeros(zt.shape[0], device=self.device)
        for _ in range(K):
            v = torch.randint(0, 2, zt_f32.shape, device=self.device, dtype=torch.float32) * 2 - 1
            inner = (f * v).sum()
            grad = torch.autograd.grad(inner, zt_f32, create_graph=False, retain_graph=True)[0]
            div_est += (grad * v).sum(dim=(1, 2, 3))
        div_est = div_est / K
        
        # 残差: sigma_t * div(f) + <f, eps_pred - eps>
        err = eps_pred.float() - eps.float()
        dot = (f.detach() * err).sum(dim=(1, 2, 3))
        
        r = sigma_t * div_est.detach() + dot
        r = r.view(num_images, num_noise)
        
        return r.abs().mean(dim=1)
    
    def evaluate_batch(
        self,
        images: torch.Tensor,
        prompts: List[str],
        images_raw: Optional[List[torch.Tensor]] = None
    ) -> List[Dict]:
        """Evaluate images using selected scheme."""
        
        num_images = images.shape[0]
        num_noise = self.config.num_noise
        
        # Get model components
        unet = self.model_manager.unet
        scheduler = self.model_manager.scheduler
        tokenizer = self.model_manager.tokenizer
        text_encoder = self.model_manager.text_encoder
        vae = self.model_manager.vae
        
        # Text embeddings
        expanded_prompts = []
        for p in prompts:
            expanded_prompts.extend([p] * num_noise)
        
        text_tokens = tokenizer(
            expanded_prompts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_tokens.input_ids.to(self.device)
        with torch.no_grad():
            text_emb = text_encoder(input_ids).last_hidden_state
        
        # Encode images to latents
        with torch.no_grad():
            z0 = vae.encode(images).latent_dist.sample()
            z0 = z0 * vae.config.scaling_factor
        
        z0 = z0.repeat_interleave(num_noise, dim=0).half()
        
        # Sample spherical noise
        eps = torch.randn_like(z0, device=self.device).half()
        eps = self.normalize_batch(eps, self.config.epsilon_reg).half()
        sqrt_d = torch.prod(torch.tensor(z0.shape[1:], device=self.device)).float().sqrt()
        eps = eps * sqrt_d
        
        # Timestep
        t = int(self.config.time_frac * scheduler.config.num_train_timesteps)
        t_tensor = torch.full((z0.shape[0],), t, device=self.device, dtype=torch.long)
        
        # z_t
        zt = scheduler.add_noise(original_samples=z0, noise=eps, timesteps=t_tensor).half()
        
        # UNet prediction
        with torch.no_grad():
            eps_pred = unet(zt, t_tensor, encoder_hidden_states=text_emb)[0].half()
        
        # Score
        alpha_bar = scheduler.alphas_cumprod[t].to(self.device).float()
        sigma_t = torch.sqrt(1.0 - alpha_bar).clamp_min(1e-8)
        score = -(eps_pred.float() / sigma_t).half()
        
        # 根据 scheme 计算 criterion
        scheme = self.scheme
        
        if scheme == "rmse":
            stat = self.compute_rmse(eps_pred, eps, num_images, num_noise)
            
        elif scheme == "stein_lap":
            stat = self.compute_stein_laplacian(zt, score, num_images, num_noise)
            
        elif scheme == "stein_simple":
            stat = self.compute_stein_simple(zt, score, num_images, num_noise)
            
        elif scheme == "score_norm":
            stat = self.compute_score_norm(score, num_images, num_noise)
            
        elif scheme == "weighted_err":
            stat = self.compute_prediction_error_weighted(
                eps_pred, eps, zt, sigma_t.item(), num_images, num_noise)
            
        elif scheme == "stein_noise":
            stat = self.compute_stein_noise_domain(
                eps_pred, eps, zt, sigma_t.item(), num_images, num_noise)
            
        elif scheme == "combined":
            # 组合 RMSE 和 Stein
            rmse = self.compute_rmse(eps_pred, eps, num_images, num_noise)
            stein = self.compute_stein_laplacian(zt, score, num_images, num_noise)
            
            # 归一化后组合
            rmse_norm = (rmse - rmse.mean()) / (rmse.std() + 1e-8)
            stein_norm = (stein - stein.mean()) / (stein.std() + 1e-8)
            
            alpha = self.combine_alpha
            stat = alpha * rmse_norm + (1 - alpha) * stein_norm
            
        elif scheme == "combined_v2":
            # 组合 RMSE 和 Score Norm
            rmse = self.compute_rmse(eps_pred, eps, num_images, num_noise)
            score_n = self.compute_score_norm(score, num_images, num_noise)
            
            # 归一化
            rmse_norm = (rmse - rmse.mean()) / (rmse.std() + 1e-8)
            score_norm = (score_n - score_n.mean()) / (score_n.std() + 1e-8)
            
            stat = rmse_norm - 0.3 * score_norm  # 负相关可能有帮助
            
        else:
            raise ValueError(f"Unknown scheme: {scheme}")
        
        # Build results
        results = []
        for i in range(num_images):
            result = {"criterion": float(stat[i].item())}
            
            if self.config.return_terms:
                # 额外计算各项指标用于分析
                rmse_val = self.compute_rmse(eps_pred, eps, num_images, num_noise)[i].item()
                result.update({
                    "scheme": scheme,
                    "rmse": rmse_val,
                    "sigma_t": sigma_t.item(),
                })
            
            results.append(result)
        
        return results