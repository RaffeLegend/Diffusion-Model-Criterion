"""
PDE-Grounded Diffusion Criterion

从 PDE/Fokker-Planck 角度推导的检测方法。

核心理论：
---------
扩散过程满足 Fokker-Planck 方程:
    ∂_t p_t(x) = β_t Δ p_t(x)

Score function s(x,t) = ∇ log p_t(x) 满足:
    对于 x ~ p_t: E[Δg - ∇g·s] = 0  (Stein identity)

我们提出多种 PDE 一致性检验:

1. Score Estimation Residual (SER):
   R_SER = ||ε_θ - ε|| = σ_t ||s_θ - s_true||
   
2. Normalized Stein Residual (NSR):
   R_NSR = |<f, s_θ>| / ||f||  (去掉常数 div(f))

3. Local Curvature Consistency (LCC):
   检验 score 在局部扰动下的一致性

4. Multi-scale PDE Residual:
   在多个时间尺度 t 上聚合残差
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

from criteria.base import BaseCriterion, register_criterion


@register_criterion("pde")
@register_criterion("pde_v3")
class PDEGroundedCriterion(BaseCriterion):
    """
    PDE-grounded criterion for detecting off-manifold images.
    
    Config options:
        - method: "ser", "nsr", "lcc", "multiscale", "combined"
        - time_fracs: list of time fractions for multiscale (default: [0.01])
        - normalize_stein: whether to normalize Stein residual (default: True)
    """
    
    def __init__(self, config, model_manager):
        super().__init__(config, model_manager)
        
        self.method = getattr(config, 'method', 'ser')
        self.time_fracs = getattr(config, 'time_fracs', [0.01])
        self.normalize_stein = getattr(config, 'normalize_stein', True)
        
        # Laplacian kernel
        self._lap_kernel = torch.tensor(
            [[0.0, 1.0, 0.0],
             [1.0, -4.0, 1.0],
             [0.0, 1.0, 0.0]], 
            device=self.device, 
            dtype=torch.float32
        ).view(1, 1, 3, 3)
    
    def laplacian(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Laplacian operator per channel."""
        C = x.shape[1]
        kernel = self._lap_kernel.repeat(C, 1, 1, 1)
        return F.conv2d(x, kernel, padding=1, groups=C)
    
    def compute_ser(
        self, 
        eps_pred: torch.Tensor, 
        eps: torch.Tensor,
        num_images: int,
        num_noise: int
    ) -> torch.Tensor:
        """
        Score Estimation Residual (SER)
        
        理论: R_SER = ||ε_θ - ε|| ∝ ||s_θ - s_true||
        
        对于 on-manifold 样本, score 估计准确, R_SER 小
        对于 off-manifold 样本, score 估计偏差大, R_SER 大
        """
        err = (eps_pred.float() - eps.float())
        per_sample = err.pow(2).mean(dim=(1, 2, 3)).sqrt()
        per_sample = per_sample.view(num_images, num_noise)
        return per_sample.mean(dim=1)
    
    def compute_nsr(
        self,
        zt: torch.Tensor,
        score: torch.Tensor,
        num_images: int,
        num_noise: int
    ) -> torch.Tensor:
        """
        Normalized Stein Residual (NSR)
        
        理论: R = <f, s> + div(f), 其中 f = 2Δ²z
        
        问题: div(f) = 40*C*H*W 是巨大常数
        解决: 归一化为 R_NSR = |<f, s>| / ||f||
        
        这度量 score 与 test function 的对齐程度
        """
        zt_f = zt.float()
        lap1 = self.laplacian(zt_f)
        lap2 = self.laplacian(lap1)
        f = 2.0 * lap2
        
        # <f, score>
        dot = (f * score.float()).sum(dim=(1, 2, 3))
        
        # ||f||
        f_norm = f.pow(2).sum(dim=(1, 2, 3)).sqrt().clamp_min(1e-8)
        
        # Normalized: |<f, s>| / ||f||
        nsr = dot.abs() / f_norm
        
        nsr = nsr.view(num_images, num_noise)
        return nsr.mean(dim=1)
    
    def compute_score_divergence(
        self,
        zt: torch.Tensor,
        score: torch.Tensor,
        num_images: int,
        num_noise: int,
        K: int = 4
    ) -> torch.Tensor:
        """
        Score Divergence Estimator
        
        理论: div(s) = Δ log p_t 应该满足某些性质
        
        使用 Hutchinson estimator: div(s) ≈ E_v[v^T ∇(s·v)]
        """
        zt_req = zt.detach().float().requires_grad_(True)
        
        # 这里我们估计 score 的 divergence 的某种统计量
        # 但这计算量大且不稳定，暂时跳过
        
        # 简化版: 直接用 score 的范数
        score_norm = score.float().pow(2).mean(dim=(1, 2, 3)).sqrt()
        score_norm = score_norm.view(num_images, num_noise)
        return score_norm.mean(dim=1)
    
    def compute_projection_residual(
        self,
        eps_pred: torch.Tensor,
        eps: torch.Tensor,
        zt: torch.Tensor,
        num_images: int,
        num_noise: int
    ) -> torch.Tensor:
        """
        Projection Residual
        
        理论: 将预测误差投影到高频子空间
        
        生成图像通常在高频部分有更大误差
        """
        err = eps_pred.float() - eps.float()
        
        # 高频部分 (Laplacian)
        err_hf = self.laplacian(err)
        
        # 高频误差的范数
        hf_err = err_hf.pow(2).mean(dim=(1, 2, 3)).sqrt()
        
        # 总误差
        total_err = err.pow(2).mean(dim=(1, 2, 3)).sqrt()
        
        # 高频比例
        hf_ratio = hf_err / (total_err + 1e-8)
        
        hf_ratio = hf_ratio.view(num_images, num_noise)
        return hf_ratio.mean(dim=1)
    
    def compute_local_consistency(
        self,
        z0: torch.Tensor,
        vae,
        unet,
        scheduler,
        text_emb: torch.Tensor,
        num_images: int,
        num_noise: int
    ) -> torch.Tensor:
        """
        Local Consistency Check
        
        理论: 对同一图像的多个噪声扰动，score 响应应该一致
        
        计算不同噪声样本的 eps_pred 的方差
        """
        # 这个在 evaluate_batch 中实现会更清晰
        pass
    
    def forward_diffusion_step(
        self,
        z0: torch.Tensor,
        scheduler,
        time_frac: float,
        num_noise: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """执行前向扩散一步"""
        z0_rep = z0.repeat_interleave(num_noise, dim=0).half()
        
        eps = torch.randn_like(z0_rep, device=self.device).half()
        eps = self.normalize_batch(eps, self.config.epsilon_reg).half()
        sqrt_d = torch.prod(torch.tensor(z0_rep.shape[1:], device=self.device)).float().sqrt()
        eps = eps * sqrt_d
        
        t = int(time_frac * scheduler.config.num_train_timesteps)
        t_tensor = torch.full((z0_rep.shape[0],), t, device=self.device, dtype=torch.long)
        
        zt = scheduler.add_noise(original_samples=z0_rep, noise=eps, timesteps=t_tensor).half()
        
        return zt, eps, t_tensor
    
    def evaluate_batch(
        self,
        images: torch.Tensor,
        prompts: List[str],
        images_raw: Optional[List[torch.Tensor]] = None
    ) -> List[Dict]:
        """Evaluate images using PDE-grounded criterion."""
        
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
        
        # Encode to latent
        with torch.no_grad():
            z0 = vae.encode(images).latent_dist.sample()
            z0 = z0 * vae.config.scaling_factor
        
        # 根据方法选择
        method = self.method
        
        if method == "multiscale":
            # 多尺度聚合
            all_stats = []
            for tf in self.time_fracs:
                zt, eps, t_tensor = self.forward_diffusion_step(z0, scheduler, tf, num_noise)
                
                with torch.no_grad():
                    eps_pred = unet(zt, t_tensor, encoder_hidden_states=text_emb)[0].half()
                
                stat = self.compute_ser(eps_pred, eps, num_images, num_noise)
                all_stats.append(stat)
            
            # 平均
            stat = torch.stack(all_stats).mean(dim=0)
        
        else:
            # 单尺度
            zt, eps, t_tensor = self.forward_diffusion_step(z0, scheduler, self.config.time_frac, num_noise)
            
            with torch.no_grad():
                eps_pred = unet(zt, t_tensor, encoder_hidden_states=text_emb)[0].half()
            
            # Score (for methods that need it)
            t = int(self.config.time_frac * scheduler.config.num_train_timesteps)
            alpha_bar = scheduler.alphas_cumprod[t].to(self.device).float()
            sigma_t = torch.sqrt(1.0 - alpha_bar).clamp_min(1e-8)
            score = -(eps_pred.float() / sigma_t).half()
            
            if method == "ser":
                stat = self.compute_ser(eps_pred, eps, num_images, num_noise)
            
            elif method == "nsr":
                stat = self.compute_nsr(zt, score, num_images, num_noise)
            
            elif method == "score_norm":
                stat = self.compute_score_divergence(zt, score, num_images, num_noise)
            
            elif method == "projection":
                stat = self.compute_projection_residual(eps_pred, eps, zt, num_images, num_noise)
            
            elif method == "combined":
                # 组合 SER 和 NSR
                ser = self.compute_ser(eps_pred, eps, num_images, num_noise)
                nsr = self.compute_nsr(zt, score, num_images, num_noise)
                
                # SER 归一化
                ser_norm = (ser - ser.mean()) / (ser.std() + 1e-8)
                # NSR 是负相关的（越小越 fake），取反
                nsr_norm = (nsr - nsr.mean()) / (nsr.std() + 1e-8)
                
                stat = ser_norm - 0.3 * nsr_norm
            
            elif method == "ser_hf":
                # SER + 高频加权
                err = eps_pred.float() - eps.float()
                
                # Laplacian 加权
                lap_zt = self.laplacian(zt.float())
                weight = (lap_zt.abs() + 0.1)
                weight = weight / weight.mean()
                
                weighted_err = (err.pow(2) * weight).mean(dim=(1, 2, 3)).sqrt()
                weighted_err = weighted_err.view(num_images, num_noise)
                stat = weighted_err.mean(dim=1)
            
            else:
                raise ValueError(f"Unknown method: {method}")
        
        # Build results
        results = []
        for i in range(num_images):
            result = {"criterion": float(stat[i].item())}
            
            if self.config.return_terms:
                result.update({
                    "method": method,
                })
            
            results.append(result)
        
        return results