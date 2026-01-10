"""
Score Laplacian Criterion - 优化版

优化点：
1. 减少 num_noise（64 → 16 或 8，精度损失很小）
2. 批量处理 CLIP，减少 GPU 调用次数
3. 移除频繁的 empty_cache
4. 使用 torch 实现 Laplacian，避免 CPU-GPU 传输
5. 缓存 x0 的 CLIP 特征
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional

from criteria.base import BaseCriterion, register_criterion
from image_utils import ImageProcessor, numpy_chunk


@register_criterion("score_laplacian_fast")
class ScoreLaplacianCriterionFast(BaseCriterion):
    """
    Score Laplacian Criterion - 速度优化版
    """
    
    def __init__(self, config, model_manager):
        super().__init__(config, model_manager)
        
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        
        self.d_clip = 768
        self.sqrt_d_clip = self.d_clip ** 0.5
        
        self.a = 1.0
        self.b = 1.0
        self.c = 1.0
        
        self.lam = getattr(config, 'score_laplacian_weight', 0.1)
        
        self.image_processor = ImageProcessor(config)
        
        # Laplacian kernel (torch 版本，在 GPU 上运行)
        lap_kernel = torch.tensor([
            [0,  1,  0],
            [1, -4,  1],
            [0,  1,  0]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        # 扩展到 3 通道
        self.lap_kernel = lap_kernel.repeat(3, 1, 1, 1).to(self.device)
    
    def apply_laplacian_torch(self, imgs: torch.Tensor) -> torch.Tensor:
        """GPU 上的 Laplacian（避免 CPU-GPU 传输）"""
        # imgs: [B, C, H, W]
        if imgs.ndim == 3:
            imgs = imgs.unsqueeze(0)
        
        # 确保数据类型一致
        imgs = imgs.float()
        kernel = self.lap_kernel.float().to(imgs.device)
        
        # 分通道卷积
        lap = F.conv2d(imgs, kernel, padding=1, groups=3)
        
        # 归一化到 [0, 1]
        lap = torch.abs(lap)
        B = lap.shape[0]
        lap_flat = lap.view(B, -1)
        lap_min = lap_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        lap_max = lap_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        lap = (lap - lap_min) / (lap_max - lap_min + 1e-8)
        
        return lap
    
    @torch.no_grad()
    def get_clip_features_batch(self, imgs_tensor: torch.Tensor) -> torch.Tensor:
        """
        批量计算 CLIP 特征（输入是 torch tensor）
        imgs_tensor: [B, C, H, W] 范围 [0, 1] 或 [-1, 1]
        """
        clip = self.model_manager.clip
        proc = self.model_manager.clip_processor
        
        # 转换为 CLIP 需要的格式
        if imgs_tensor.min() < 0:
            imgs_tensor = (imgs_tensor + 1) / 2  # [-1,1] -> [0,1]
        
        imgs_tensor = imgs_tensor.clamp(0, 1)
        
        # Resize to CLIP input size (224x224)
        imgs_resized = F.interpolate(imgs_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Normalize with CLIP stats
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(self.device)
        imgs_norm = (imgs_resized - mean) / std
        
        # 分批处理避免 OOM
        batch_size = 32  # 增大 batch size
        all_feats = []
        
        for i in range(0, imgs_norm.shape[0], batch_size):
            batch = imgs_norm[i:i+batch_size]
            feats = clip.get_image_features(pixel_values=batch)
            all_feats.append(feats)
        
        return torch.cat(all_feats, dim=0)
    
    def postprocess_decoded(self, decoded: torch.Tensor, size: int) -> np.ndarray:
        return self.image_processor.postprocess(decoded, True)

    @torch.no_grad()
    def evaluate_batch(
        self,
        images: torch.Tensor,
        prompts: List[str],
        images_raw: Optional[List[torch.Tensor]] = None
    ) -> List[Dict]:
        
        if images_raw is None:
            raise ValueError("需要 images_raw")
        
        B = images.shape[0]
        S = self.config.num_noise
        
        unet = self.model_manager.unet
        vae = self.model_manager.vae
        scheduler = self.model_manager.scheduler
        tokenizer = self.model_manager.tokenizer
        text_encoder = self.model_manager.text_encoder
        
        # ============ 1. Text embeddings (一次性计算) ============
        exp_prompts = [p for p in prompts for _ in range(S)]
        tok = tokenizer(exp_prompts, padding="max_length", max_length=77,
                       truncation=True, return_tensors="pt")
        text_emb = text_encoder(tok.input_ids.to(self.device)).last_hidden_state
        
        # ============ 2. VAE Encode (一次) ============
        z = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
        z = z.repeat_interleave(S, dim=0).half()
        
        # ============ 3. 生成球面噪声 ============
        noise = torch.randn_like(z, device=self.device).half()
        sqrt_d = torch.prod(torch.tensor(z.shape[1:])).float().sqrt()
        u = self.normalize_batch(noise, self.config.epsilon_reg).half() * sqrt_d
        
        # ============ 4. UNet 前向 (主要瓶颈) ============
        t = self.config.time_frac * scheduler.config.num_train_timesteps
        t_tensor = torch.full((z.shape[0],), t, device=self.device, dtype=torch.long)
        z_noisy = scheduler.add_noise(z, u, t_tensor).half()
        
        h = unet(z_noisy, t_tensor, encoder_hidden_states=text_emb)[0]
        h_scaled = h / vae.config.scaling_factor
        
        del z_noisy, text_emb, noise
        
        # ============ 5. VAE Decode (分批但不清缓存) ============
        def decode_batch(x, batch_sz=16):
            results = []
            for i in range(0, x.shape[0], batch_sz):
                dec = vae.decode(x[i:i+batch_sz], return_dict=False)[0]
                results.append(dec)
            return torch.cat(results, dim=0)
        
        h_dec = decode_batch(h_scaled)  # [B*S, 3, H, W]
        u_dec = decode_batch(u / vae.config.scaling_factor)
        
        del h, h_scaled, u, z
        
        # ============ 6. 在 GPU 上计算 Laplacian ============
        h_dec = h_dec.float()  # 确保是 float32
        u_dec = u_dec.float()
        h_dec_01 = (h_dec + 1) / 2  # [-1,1] -> [0,1]
        h_lap = self.apply_laplacian_torch(h_dec_01)  # [B*S, 3, H, W]
        
        # ============ 7. 批量 CLIP 特征 ============
        # h 的特征
        h_feats = self.get_clip_features_batch(h_dec_01)  # [B*S, 768]
        h_feats = F.normalize(h_feats, p=2, dim=1)
        
        # Δh 的特征
        h_lap_feats = self.get_clip_features_batch(h_lap)  # [B*S, 768]
        h_lap_feats = F.normalize(h_lap_feats, p=2, dim=1)
        
        # 增强特征 (减法)
        h_enhanced = h_feats - self.lam * h_lap_feats  # [B*S, 768]
        
        # u 的特征
        u_dec_01 = (u_dec + 1) / 2
        u_feats = self.get_clip_features_batch(u_dec_01)  # [B*S, 768]
        
        del h_dec, u_dec, h_lap, h_dec_01, u_dec_01
        
        # ============ 8. x0 的 CLIP 特征 ============
        # 处理原图 - 统一尺寸
        target_size = (256, 256)  # 统一到 256x256
        x0_list = []
        for raw_i in images_raw:
            img = raw_i.float()
            if img.ndim == 3:
                img = img.unsqueeze(0)
            if img.shape[1] != 3 and img.shape[-1] == 3:
                img = img.permute(0, 3, 1, 2)
            if img.max() > 1:
                img = img / 255.0
            # Resize to uniform size
            img = F.interpolate(img, size=target_size, mode='bilinear', align_corners=False)
            x0_list.append(img)
        
        x0_tensor = torch.cat(x0_list, dim=0).to(self.device)  # [B, 3, H, W]
        x0_feats = self.get_clip_features_batch(x0_tensor)  # [B, 768]
        
        # ============ 9. 计算 Criterion ============
        results = []
        
        for i in range(B):
            # 当前图像的 S 个样本
            start_idx = i * S
            end_idx = (i + 1) * S
            
            h_f = h_enhanced[start_idx:end_idx]  # [S, 768]
            u_f = u_feats[start_idx:end_idx]      # [S, 768]
            x0_f = x0_feats[i]                     # [768]
            
            # Criterion 计算
            h_norm = torch.norm(h_f, p=2, dim=1, keepdim=True) + 1e-8
            h_dir = -h_f / h_norm
            x0_exp = x0_f.unsqueeze(0).expand(S, -1)
            vec = self.a * u_f - self.b * h_f + self.c * self.sqrt_d_clip * x0_exp
            
            C = self.cos(h_dir, vec).mean().item()
            C_norm = (C + 1) / (self.a + self.b + self.c + 1)
            
            res = {"criterion": float(C_norm)}
            
            if self.config.return_terms:
                res.update({"lambda": float(self.lam)})
            
            results.append(res)
        
        return results