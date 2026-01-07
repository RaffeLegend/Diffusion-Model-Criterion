#!/usr/bin/env python
"""
改进版 High-Order Structural Criterion

针对跨生成器泛化问题的改进：
1. 多时间尺度聚合
2. 自适应方向检测
3. 多阶融合
4. 相对误差（而非绝对误差）
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

from criteria.base import BaseCriterion, register_criterion


@register_criterion("hosc_v2")
@register_criterion("pde")
class ImprovedHOSCCriterion(BaseCriterion):
    """
    改进版 HOSC，增强跨生成器泛化能力
    
    Methods:
        - "multiscale": 多时间尺度聚合
        - "relative": 相对误差 ||e|| / ||eps||
        - "gradient_align": 梯度对齐度
        - "curvature": 局部曲率估计
        - "ensemble": 多方法融合
    """
    
    def __init__(self, config, model_manager):
        super().__init__(config, model_manager)
        
        self.method = getattr(config, 'method', 'multiscale')
        self.time_fracs = getattr(config, 'time_fracs', [0.005, 0.01, 0.02, 0.05])
        
        # Laplacian kernel
        self._lap_kernel = torch.tensor(
            [[0.0, 1.0, 0.0],
             [1.0, -4.0, 1.0],
             [0.0, 1.0, 0.0]], 
            device=self.device, 
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        
        # Gradient kernels (Sobel)
        self._sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]], 
            device=self.device, 
            dtype=torch.float32
        ).view(1, 1, 3, 3) / 4.0
        
        self._sobel_y = torch.tensor(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]], 
            device=self.device, 
            dtype=torch.float32
        ).view(1, 1, 3, 3) / 4.0
    
    def laplacian(self, x: torch.Tensor) -> torch.Tensor:
        C = x.shape[1]
        kernel = self._lap_kernel.repeat(C, 1, 1, 1)
        return F.conv2d(x, kernel, padding=1, groups=C)
    
    def gradient(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算 x 和 y 方向梯度"""
        C = x.shape[1]
        kx = self._sobel_x.repeat(C, 1, 1, 1)
        ky = self._sobel_y.repeat(C, 1, 1, 1)
        gx = F.conv2d(x, kx, padding=1, groups=C)
        gy = F.conv2d(x, ky, padding=1, groups=C)
        return gx, gy
    
    def forward_diffusion_step(
        self,
        z0: torch.Tensor,
        scheduler,
        time_frac: float,
        num_noise: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """执行前向扩散"""
        z0_rep = z0.repeat_interleave(num_noise, dim=0).half()
        
        eps = torch.randn_like(z0_rep, device=self.device).half()
        eps = self.normalize_batch(eps, self.config.epsilon_reg).half()
        sqrt_d = torch.prod(torch.tensor(z0_rep.shape[1:], device=self.device)).float().sqrt()
        eps = eps * sqrt_d
        
        t = int(time_frac * scheduler.config.num_train_timesteps)
        t = max(1, min(t, scheduler.config.num_train_timesteps - 1))
        t_tensor = torch.full((z0_rep.shape[0],), t, device=self.device, dtype=torch.long)
        
        zt = scheduler.add_noise(original_samples=z0_rep, noise=eps, timesteps=t_tensor).half()
        
        # sigma_t
        alpha_bar = scheduler.alphas_cumprod[t].to(self.device).float()
        sigma_t = torch.sqrt(1.0 - alpha_bar).item()
        
        return zt, eps, t_tensor, sigma_t
    
    def compute_projection(self, err: torch.Tensor) -> torch.Tensor:
        """高频投影比例"""
        err_hf = self.laplacian(err.float())
        hf_norm = err_hf.pow(2).sum(dim=(1, 2, 3)).sqrt()
        total_norm = err.float().pow(2).sum(dim=(1, 2, 3)).sqrt().clamp_min(1e-8)
        return hf_norm / total_norm
    
    def compute_relative_error(
        self, 
        eps_pred: torch.Tensor, 
        eps: torch.Tensor
    ) -> torch.Tensor:
        """相对误差: ||eps_pred - eps|| / ||eps||"""
        err = eps_pred.float() - eps.float()
        err_norm = err.pow(2).sum(dim=(1, 2, 3)).sqrt()
        eps_norm = eps.float().pow(2).sum(dim=(1, 2, 3)).sqrt().clamp_min(1e-8)
        return err_norm / eps_norm
    
    def compute_gradient_alignment(
        self,
        eps_pred: torch.Tensor,
        eps: torch.Tensor
    ) -> torch.Tensor:
        """
        梯度对齐度: 预测噪声和真实噪声的梯度方向一致性
        
        理论: 对于 in-distribution 样本，梯度方向应该高度一致
        """
        # 计算梯度
        gx_pred, gy_pred = self.gradient(eps_pred.float())
        gx_true, gy_true = self.gradient(eps.float())
        
        # 梯度向量
        grad_pred = torch.stack([gx_pred, gy_pred], dim=-1)  # (B, C, H, W, 2)
        grad_true = torch.stack([gx_true, gy_true], dim=-1)
        
        # 逐点 cosine similarity
        dot = (grad_pred * grad_true).sum(dim=-1)
        norm_pred = grad_pred.pow(2).sum(dim=-1).sqrt().clamp_min(1e-8)
        norm_true = grad_true.pow(2).sum(dim=-1).sqrt().clamp_min(1e-8)
        
        cos_sim = dot / (norm_pred * norm_true)
        
        # 平均 cosine similarity
        return cos_sim.mean(dim=(1, 2, 3))
    
    def compute_error_smoothness(
        self,
        err: torch.Tensor
    ) -> torch.Tensor:
        """
        误差平滑度: Laplacian 能量 / 总能量
        
        理论: 生成图像的误差可能更平滑（低频主导）或更不平滑
        """
        err_lap = self.laplacian(err.float())
        lap_energy = err_lap.pow(2).mean(dim=(1, 2, 3))
        total_energy = err.float().pow(2).mean(dim=(1, 2, 3)).clamp_min(1e-8)
        return lap_energy / total_energy
    
    def compute_cosine_similarity(
        self,
        eps_pred: torch.Tensor,
        eps: torch.Tensor
    ) -> torch.Tensor:
        """预测噪声和真实噪声的整体 cosine similarity"""
        pred_flat = eps_pred.float().view(eps_pred.shape[0], -1)
        true_flat = eps.float().view(eps.shape[0], -1)
        
        cos_sim = F.cosine_similarity(pred_flat, true_flat, dim=1)
        return cos_sim
    
    def evaluate_single_timestep(
        self,
        z0: torch.Tensor,
        text_emb: torch.Tensor,
        scheduler,
        unet,
        time_frac: float,
        num_images: int,
        num_noise: int
    ) -> Dict[str, torch.Tensor]:
        """在单个时间步评估"""
        zt, eps, t_tensor, sigma_t = self.forward_diffusion_step(
            z0, scheduler, time_frac, num_noise
        )
        
        with torch.no_grad():
            eps_pred = unet(zt, t_tensor, encoder_hidden_states=text_emb)[0].half()
        
        err = eps_pred.float() - eps.float()
        
        # 计算各种统计量
        results = {
            'projection': self.compute_projection(err),
            'relative_err': self.compute_relative_error(eps_pred, eps),
            'grad_align': self.compute_gradient_alignment(eps_pred, eps),
            'smoothness': self.compute_error_smoothness(err),
            'cosine': self.compute_cosine_similarity(eps_pred, eps),
            'rmse': err.pow(2).mean(dim=(1, 2, 3)).sqrt(),
            'sigma_t': sigma_t,
        }
        
        return results
    
    def evaluate_batch(
        self,
        images: torch.Tensor,
        prompts: List[str],
        images_raw: Optional[List[torch.Tensor]] = None
    ) -> List[Dict]:
        """评估"""
        
        num_images = images.shape[0]
        num_noise = self.config.num_noise
        method = self.method
        
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
        
        if method == "multiscale":
            # 多时间尺度聚合
            all_projections = []
            all_rmse = []
            
            for tf in self.time_fracs:
                results = self.evaluate_single_timestep(
                    z0, text_emb, scheduler, unet, tf, num_images, num_noise
                )
                proj = results['projection'].view(num_images, num_noise).mean(dim=1)
                rmse = results['rmse'].view(num_images, num_noise).mean(dim=1)
                all_projections.append(proj)
                all_rmse.append(rmse)
            
            # 聚合: 取平均或取 std
            proj_stack = torch.stack(all_projections, dim=0)  # (T, B)
            rmse_stack = torch.stack(all_rmse, dim=0)
            
            # 方案1: 平均
            stat = proj_stack.mean(dim=0)
            
            # # 方案2: 跨时间的变化 (std)
            # stat = proj_stack.std(dim=0)
            
        elif method == "relative":
            results = self.evaluate_single_timestep(
                z0, text_emb, scheduler, unet, self.config.time_frac, num_images, num_noise
            )
            stat = results['relative_err'].view(num_images, num_noise).mean(dim=1)
            
        elif method == "grad_align":
            results = self.evaluate_single_timestep(
                z0, text_emb, scheduler, unet, self.config.time_frac, num_images, num_noise
            )
            stat = results['grad_align'].view(num_images, num_noise).mean(dim=1)
            
        elif method == "cosine":
            results = self.evaluate_single_timestep(
                z0, text_emb, scheduler, unet, self.config.time_frac, num_images, num_noise
            )
            stat = results['cosine'].view(num_images, num_noise).mean(dim=1)
            
        elif method == "smoothness":
            results = self.evaluate_single_timestep(
                z0, text_emb, scheduler, unet, self.config.time_frac, num_images, num_noise
            )
            stat = results['smoothness'].view(num_images, num_noise).mean(dim=1)
            
        elif method == "ensemble":
            # 多方法融合
            results = self.evaluate_single_timestep(
                z0, text_emb, scheduler, unet, self.config.time_frac, num_images, num_noise
            )
            
            proj = results['projection'].view(num_images, num_noise).mean(dim=1)
            rmse = results['rmse'].view(num_images, num_noise).mean(dim=1)
            cosine = results['cosine'].view(num_images, num_noise).mean(dim=1)
            
            # 归一化
            proj_n = (proj - proj.mean()) / (proj.std() + 1e-8)
            rmse_n = (rmse - rmse.mean()) / (rmse.std() + 1e-8)
            cosine_n = (cosine - cosine.mean()) / (cosine.std() + 1e-8)
            
            # 组合: rmse (higher=fake for SD) + cosine (lower=fake)
            stat = rmse_n - cosine_n
            
        else:
            # 默认: projection
            results = self.evaluate_single_timestep(
                z0, text_emb, scheduler, unet, self.config.time_frac, num_images, num_noise
            )
            stat = results['projection'].view(num_images, num_noise).mean(dim=1)
        
        # Build results
        output = []
        for i in range(num_images):
            result = {"criterion": float(stat[i].item())}
            output.append(result)
        
        return output