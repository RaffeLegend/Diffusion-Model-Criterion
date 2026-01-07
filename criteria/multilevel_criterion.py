"""
Multi-Level PDE Consistency Criterion

多层级 PDE 一致性检测：
1. Latent Level: 高阶微分残差 (HOSC)
2. Image Level: 重建一致性
3. Semantic Level: CLIP 语义一致性

理论框架：
=========
扩散模型的 score function s_θ(x,t) 应该满足 Fokker-Planck 方程。
对于 on-manifold 样本，PDE 残差在各层级都应该一致。

我们检测三个层级的一致性：
- Latent: ||Δ(ε_θ - ε)|| / ||ε_θ - ε||  (高频结构)
- Image: decode(ε_θ) vs decode(ε)       (重建)
- Semantic: CLIP(decode(ε_θ), decode(ε)) (语义)

这些都是 Stein residual 的不同 test function 下的表现。
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

from criteria.base import BaseCriterion, register_criterion


@register_criterion("multilevel")
@register_criterion("pde")
class MultiLevelCriterion(BaseCriterion):
    """
    多层级 PDE 一致性检测
    
    Methods:
        - "latent": Latent 空间高阶残差
        - "clip": CLIP 语义一致性 (类似原文)
        - "hybrid": Latent + CLIP 融合
        - "image": Image 空间重建误差
    """
    
    def __init__(self, config, model_manager):
        super().__init__(config, model_manager)
        
        self.method = getattr(config, 'method', 'hybrid')
        self.clip_weight = getattr(config, 'clip_weight', 0.5)
        
        # Laplacian kernel
        self._lap_kernel = torch.tensor(
            [[0.0, 1.0, 0.0],
             [1.0, -4.0, 1.0],
             [0.0, 1.0, 0.0]], 
            device=self.device, 
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        
        # CLIP 相关
        self._clip = None
        self._clip_processor = None
    
    def laplacian(self, x: torch.Tensor) -> torch.Tensor:
        C = x.shape[1]
        kernel = self._lap_kernel.repeat(C, 1, 1, 1)
        return F.conv2d(x, kernel, padding=1, groups=C)
    
    def get_clip(self):
        """懒加载 CLIP"""
        if self._clip is None:
            clip_model, processor = self.model_manager.load_clip()
            self._clip = clip_model
            self._clip_processor = processor
        return self._clip, self._clip_processor
    
    def decode_latent(self, z: torch.Tensor, vae) -> torch.Tensor:
        """Decode latent to image space"""
        # 注意 scaling factor
        z_scaled = z / vae.config.scaling_factor
        with torch.no_grad():
            decoded = vae.decode(z_scaled, return_dict=False)[0]
        return decoded
    
    def compute_clip_features(self, images: torch.Tensor) -> torch.Tensor:
        """计算 CLIP 特征"""
        clip, processor = self.get_clip()
        
        # 归一化到 [0, 1]
        images_norm = (images.clamp(-1, 1) + 1) / 2
        
        # 转换为 CLIP 输入格式
        # CLIP 期望 (B, 3, 224, 224)
        images_resized = F.interpolate(images_norm, size=(224, 224), mode='bilinear', align_corners=False)
        
        # 标准化 (CLIP 的均值和标准差)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)
        images_normalized = (images_resized - mean) / std
        
        with torch.no_grad():
            features = clip.get_image_features(pixel_values=images_normalized)
            features = features / features.norm(dim=-1, keepdim=True)
        
        return features
    
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
        
        alpha_bar = scheduler.alphas_cumprod[t].to(self.device).float()
        sigma_t = torch.sqrt(1.0 - alpha_bar).item()
        
        return zt, eps, t_tensor, sigma_t
    
    def compute_latent_criterion(
        self,
        eps_pred: torch.Tensor,
        eps: torch.Tensor
    ) -> torch.Tensor:
        """Latent 层级: 高阶投影比例"""
        err = eps_pred.float() - eps.float()
        err_hf = self.laplacian(err)
        hf_norm = err_hf.pow(2).sum(dim=(1, 2, 3)).sqrt()
        total_norm = err.pow(2).sum(dim=(1, 2, 3)).sqrt().clamp_min(1e-8)
        return hf_norm / total_norm
    
    def compute_clip_criterion(
        self,
        eps_pred: torch.Tensor,
        eps: torch.Tensor,
        vae,
        original_images: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        CLIP 层级: 语义一致性
        
        计算:
        - B: cos(original, decoded_pred) - 原图与预测响应的对齐
        - κ: cos(decoded_pred, decoded_ref) - 预测与参考的对齐
        - D: ||features|| - 特征能量
        """
        # Decode to image space
        # 注意: 这里我们 decode 的是 noise (不是完整图像)
        # 需要先缩放
        scaling = vae.config.scaling_factor
        
        # 方案1: decode 噪声差异（误差）
        err = eps_pred - eps
        err_scaled = err / scaling
        
        # 分批 decode 避免 OOM
        batch_size = eps_pred.shape[0]
        max_batch = 8
        
        decoded_pred_list = []
        decoded_ref_list = []
        
        for i in range(0, batch_size, max_batch):
            end = min(i + max_batch, batch_size)
            
            # Decode predicted noise
            pred_batch = eps_pred[i:end] / scaling
            with torch.no_grad():
                dec_pred = vae.decode(pred_batch.float(), return_dict=False)[0]
            decoded_pred_list.append(dec_pred)
            
            # Decode reference noise
            ref_batch = eps[i:end] / scaling
            with torch.no_grad():
                dec_ref = vae.decode(ref_batch.float(), return_dict=False)[0]
            decoded_ref_list.append(dec_ref)
        
        decoded_pred = torch.cat(decoded_pred_list, dim=0)
        decoded_ref = torch.cat(decoded_ref_list, dim=0)
        
        # 计算 CLIP 特征
        feat_pred = self.compute_clip_features(decoded_pred)
        feat_ref = self.compute_clip_features(decoded_ref)
        
        # κ: 预测与参考的 cosine similarity
        kappa = F.cosine_similarity(feat_pred, feat_ref, dim=-1)
        
        # D: 预测特征的能量 (norm)
        D = feat_pred.norm(dim=-1)
        
        # B: 如果有原图，计算与原图的对齐
        if original_images is not None:
            feat_orig = self.compute_clip_features(original_images.repeat_interleave(
                eps_pred.shape[0] // original_images.shape[0], dim=0
            ))
            B = F.cosine_similarity(feat_orig, feat_pred, dim=-1)
        else:
            B = torch.zeros_like(kappa)
        
        return {
            'B': B,      # 原图对齐
            'kappa': kappa,  # 预测-参考对齐
            'D': D,      # 能量
        }
    
    def compute_original_clip_criterion(
        self,
        eps_pred: torch.Tensor,
        eps: torch.Tensor,
        vae,
        z0: torch.Tensor,
        num_images: int,
        num_noise: int
    ) -> torch.Tensor:
        """
        原文的 CLIP criterion (简化版)
        
        R(x) = 1 + (√d·B - D + κ) / (√d + 2)
        """
        d_clip = 512  # CLIP 维度
        sqrt_d = np.sqrt(d_clip)
        
        clip_results = self.compute_clip_criterion(eps_pred, eps, vae)
        
        B = clip_results['B']
        kappa = clip_results['kappa']
        D = clip_results['D']
        
        # 聚合每张图片
        B = B.view(num_images, num_noise).mean(dim=1)
        kappa = kappa.view(num_images, num_noise).mean(dim=1)
        D = D.view(num_images, num_noise).mean(dim=1)
        
        # 原文公式
        criterion = 1 + (sqrt_d * B - D + kappa) / (sqrt_d + 2)
        
        return criterion
    
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
        
        # Forward diffusion
        zt, eps, t_tensor, sigma_t = self.forward_diffusion_step(
            z0, scheduler, self.config.time_frac, num_noise
        )
        
        # UNet prediction
        with torch.no_grad():
            eps_pred = unet(zt, t_tensor, encoder_hidden_states=text_emb)[0].half()
        
        if method == "latent":
            # 纯 Latent 层级
            stat = self.compute_latent_criterion(eps_pred, eps)
            stat = stat.view(num_images, num_noise).mean(dim=1)
            
        elif method == "clip":
            # 纯 CLIP 层级 (类似原文)
            stat = self.compute_original_clip_criterion(
                eps_pred, eps, vae, z0, num_images, num_noise
            )
            
        elif method == "hybrid":
            # Latent + CLIP 融合
            latent_stat = self.compute_latent_criterion(eps_pred, eps)
            latent_stat = latent_stat.view(num_images, num_noise).mean(dim=1)
            
            clip_stat = self.compute_original_clip_criterion(
                eps_pred, eps, vae, z0, num_images, num_noise
            )
            
            # 归一化后融合
            latent_norm = (latent_stat - latent_stat.mean()) / (latent_stat.std() + 1e-8)
            clip_norm = (clip_stat - clip_stat.mean()) / (clip_stat.std() + 1e-8)
            
            # latent: lower=fake, clip: lower=fake (通常)
            # 所以直接加
            alpha = self.clip_weight
            stat = (1 - alpha) * latent_norm + alpha * clip_norm
            
        elif method == "image":
            # Image 层级: decode 后直接比较
            err = eps_pred.float() - eps.float()
            
            # Decode 误差
            err_scaled = err / vae.config.scaling_factor
            decoded_err_list = []
            batch_size = err.shape[0]
            max_batch = 8
            
            for i in range(0, batch_size, max_batch):
                end = min(i + max_batch, batch_size)
                with torch.no_grad():
                    dec = vae.decode(err_scaled[i:end].float(), return_dict=False)[0]
                decoded_err_list.append(dec)
            
            decoded_err = torch.cat(decoded_err_list, dim=0)
            
            # Image 空间的 RMSE
            stat = decoded_err.pow(2).mean(dim=(1, 2, 3)).sqrt()
            stat = stat.view(num_images, num_noise).mean(dim=1)
            
        elif method == "kappa_only":
            # 只用 kappa (预测-参考对齐)
            clip_results = self.compute_clip_criterion(eps_pred, eps, vae)
            stat = clip_results['kappa'].view(num_images, num_noise).mean(dim=1)
            
        elif method == "cosine_latent":
            # Latent 空间的 cosine similarity
            pred_flat = eps_pred.float().view(eps_pred.shape[0], -1)
            ref_flat = eps.float().view(eps.shape[0], -1)
            cos_sim = F.cosine_similarity(pred_flat, ref_flat, dim=1)
            stat = cos_sim.view(num_images, num_noise).mean(dim=1)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Build results
        results = []
        for i in range(num_images):
            result = {"criterion": float(stat[i].item())}
            results.append(result)
        
        return results