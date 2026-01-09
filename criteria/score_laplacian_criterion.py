"""
Score Laplacian Criterion

核心思路：对 score (h) 做 Laplacian，而不是对原图 (x)

理论一致性：
- h ≈ ∇log p (score function，一阶微分)
- Δh ≈ Δ∇log p = ∇Δlog p (score 的 Laplacian，涉及三阶微分)
- 或者看作 ∇·∇h = ∇²h，即 score 的 Hessian 的迹

方法：
- 原文：CLIP(h)
- 改进：CLIP(h) + λ·CLIP(Δh)

这样理论（高阶微分）和方法（Laplacian 作用在 h 上）完全一致。
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional
from scipy import ndimage

from criteria.base import BaseCriterion, register_criterion
from image_utils import ImageProcessor, numpy_chunk


@register_criterion("score_laplacian")
class ScoreLaplacianCriterion(BaseCriterion):
    """
    Score Laplacian Criterion
    
    对 h（score/噪声预测）做 Laplacian，再过 CLIP。
    
    公式：
        h_enhanced = CLIP(h) + λ · CLIP(Δh)
        C = ⟨-h_enhanced/||h_enhanced||, a·u - b·h_enhanced + c·√d·x₀⟩
    
    理论依据：
        h ≈ ∇log p（一阶）
        Δh ≈ 高阶微分结构
        生成图像的 Δh 能量更大（非稳态区域）
    """
    
    def __init__(self, config, model_manager):
        super().__init__(config, model_manager)
        
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        
        self.d_clip = 768
        self.sqrt_d_clip = self.d_clip ** 0.5
        
        self.a = 1.0
        self.b = 1.0
        self.c = 1.0
        
        # Laplacian 权重
        self.lam = getattr(config, 'score_laplacian_weight', 0.1)
        
        self.image_processor = ImageProcessor(config)
        
        # Laplacian kernel
        self.lap_kernel = np.array([
            [0,  1,  0],
            [1, -4,  1],
            [0,  1,  0]
        ], dtype=np.float32)
    
    def apply_laplacian(self, imgs: np.ndarray) -> np.ndarray:
        """对图像应用 Laplacian 算子"""
        if imgs.ndim == 3:
            imgs = imgs[np.newaxis, ...]
        
        result = []
        for img in imgs:
            lap = np.zeros_like(img, dtype=np.float32)
            for c in range(img.shape[2]):
                lap[:,:,c] = ndimage.convolve(
                    img[:,:,c].astype(np.float32),
                    self.lap_kernel, mode='reflect'
                )
            # 归一化到 [0, 255]
            lap = np.abs(lap)
            if lap.max() > lap.min():
                lap = (lap - lap.min()) / (lap.max() - lap.min()) * 255
            result.append(lap.astype(np.uint8))
        
        return np.stack(result, axis=0)
    
    def get_clip_features(self, imgs: np.ndarray) -> torch.Tensor:
        """计算 CLIP 特征（分批处理避免 OOM）"""
        clip = self.model_manager.clip
        proc = self.model_manager.clip_processor
        
        if imgs.ndim == 3:
            imgs = imgs[np.newaxis, ...]
        
        # 分批处理
        batch_size = 8
        all_feats = []
        
        for i in range(0, len(imgs), batch_size):
            batch = imgs[i:i+batch_size]
            inputs = proc(images=list(batch), return_tensors="pt").to(self.device)
            with torch.no_grad():
                feats = clip.get_image_features(**inputs)
            all_feats.append(feats.detach().cpu())
            del inputs, feats
            torch.cuda.empty_cache()
        
        return torch.cat(all_feats, dim=0)
    
    def get_enhanced_features(self, imgs: np.ndarray) -> torch.Tensor:
        """计算增强特征: CLIP(h) - λ·CLIP(Δh)  (减法！)"""
        # 原始特征
        feats_orig = self.get_clip_features(imgs)
        feats_orig = F.normalize(feats_orig, p=2, dim=1)
        
        # Laplacian 特征
        imgs_lap = self.apply_laplacian(imgs)
        feats_lap = self.get_clip_features(imgs_lap)
        feats_lap = F.normalize(feats_lap, p=2, dim=1)
        
        # 融合：减法
        enhanced = feats_orig - self.lam * feats_lap
        
        return enhanced
    
    def postprocess_decoded(self, decoded: torch.Tensor, size: int) -> np.ndarray:
        return self.image_processor.postprocess(decoded, True)
    
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
        
        # Text embeddings
        exp_prompts = [p for p in prompts for _ in range(S)]
        tok = tokenizer(exp_prompts, padding="max_length", max_length=77,
                       truncation=True, return_tensors="pt")
        text_emb = text_encoder(tok.input_ids.to(self.device)).last_hidden_state
        
        # Encode
        with torch.no_grad():
            z = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
        z = z.repeat_interleave(S, dim=0).half()
        
        # Spherical noise
        noise = torch.randn_like(z, device=self.device).half()
        sqrt_d = torch.prod(torch.tensor(z.shape[1:])).float().sqrt()
        u = self.normalize_batch(noise, self.config.epsilon_reg).half() * sqrt_d
        
        # Forward
        t = self.config.time_frac * scheduler.config.num_train_timesteps
        t_tensor = torch.full((z.shape[0],), t, device=self.device, dtype=torch.long)
        z_noisy = scheduler.add_noise(z, u, t_tensor).half()
        
        with torch.no_grad():
            h = unet(z_noisy, t_tensor, encoder_hidden_states=text_emb)[0]
        h_scaled = h / vae.config.scaling_factor
        
        del z_noisy, text_emb, noise
        
        # Decode (分批)
        def decode(x):
            results = []
            batch_sz = 8
            for i in range(0, x.shape[0], batch_sz):
                with torch.no_grad():
                    dec = vae.decode(x[i:i+batch_sz], return_dict=False)[0]
                results.append(dec.cpu())
                del dec
                torch.cuda.empty_cache()
            return torch.cat(results, dim=0).to(self.device)
        
        h_dec = decode(h_scaled)
        u_dec = decode(u / vae.config.scaling_factor)
        
        del h, h_scaled, u, z
        
        sz = self.config.image_size
        h_np = self.postprocess_decoded(h_dec, sz)
        u_np = self.postprocess_decoded(u_dec, sz)
        
        del h_dec, u_dec
        
        h_chunks = numpy_chunk(h_np, B)
        u_chunks = numpy_chunk(u_np, B)
        
        results = []
        for i, (h_i, u_i, raw_i) in enumerate(zip(h_chunks, u_chunks, images_raw)):
            
            # 原图
            img = raw_i.float().cpu().numpy()
            if img.ndim == 4: img = img[0]
            if img.shape[0] == 3: img = np.transpose(img, (1,2,0))
            img = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
            
            # 特征：x₀ 用原始 CLIP，h 用增强特征
            x0_f = self.get_clip_features(img).squeeze(0)
            h_f = self.get_enhanced_features(h_i)  # 关键：对 h 做 Laplacian 增强
            u_f = self.get_clip_features(u_i)
            
            n = h_f.shape[0]
            
            # Criterion
            h_norm = torch.norm(h_f, p=2, dim=1, keepdim=True) + 1e-8
            h_dir = -h_f / h_norm
            x0_exp = x0_f.unsqueeze(0).expand(n, -1)
            vec = self.a * u_f - self.b * h_f + self.c * self.sqrt_d_clip * x0_exp
            
            C = self.cos(h_dir, vec).mean().item()
            C_norm = (C + 1) / (self.a + self.b + self.c + 1)
            
            res = {"criterion": float(C_norm)}
            
            if self.config.return_terms:
                res.update({
                    "lambda": float(self.lam),
                })
            
            results.append(res)
        
        return results