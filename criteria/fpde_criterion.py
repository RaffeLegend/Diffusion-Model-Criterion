"""
PDE-Based Manifold Criterion

从 Fokker-Planck 方程出发，推导完整的判别准则：
    C(x₀) = C₀ + ε · C₁

其中：
    C₀: 主项，score 场的方向性（与原文等价）
    C₁: 高阶修正项，score 场的空间变化率
    ε: 小参数

当 ε → 0 时退化为原文方法。
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional
from scipy import ndimage

from criteria.base import BaseCriterion, register_criterion
from image_utils import ImageProcessor, numpy_chunk


@register_criterion("fpde")
class FPDECriterion(BaseCriterion):
    """
    Fokker-Planck PDE Criterion
    
    理论框架：
        扩散过程满足 Fokker-Planck 方程：
        ∂p/∂t = ∇·(p∇U) + Δp
        
        Score function s = ∇log p，噪声预测 h ≈ s
        
        完整判别准则：
        C = C₀ + ε·C₁
        
        C₀ = ⟨-h/||h||, a·u - b·h + c·√d·x₀⟩  (主项)
        C₁ = ||Δh|| / ||h||                    (高阶修正)
    """
    
    def __init__(self, config, model_manager):
        super().__init__(config, model_manager)
        
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        
        self.d_clip = 768
        self.sqrt_d_clip = self.d_clip ** 0.5
        
        self.a = 1.0
        self.b = 1.0  
        self.c = 1.0
        
        # 高阶修正系数 ε
        self.eps = getattr(config, 'pde_epsilon', 0.001)
        
        self.image_processor = ImageProcessor(config)
        
        self.lap_kernel = np.array([
            [0,  1,  0],
            [1, -4,  1],
            [0,  1,  0]
        ], dtype=np.float32)
    
    def get_clip_features(self, imgs: np.ndarray) -> torch.Tensor:
        clip = self.model_manager.clip
        proc = self.model_manager.clip_processor
        
        if imgs.ndim == 3:
            imgs = imgs[np.newaxis, ...]
        
        inputs = proc(images=list(imgs), return_tensors="pt").to(self.device)
        with torch.no_grad():
            feats = clip.get_image_features(**inputs)
        return feats.detach().cpu()
    
    def laplacian_energy(self, imgs: np.ndarray) -> float:
        """||Δh||₂"""
        if imgs.ndim == 3:
            imgs = imgs[np.newaxis, ...]
        
        total = 0.0
        for img in imgs:
            for c in range(img.shape[2]):
                lap = ndimage.convolve(img[:,:,c].astype(np.float32), 
                                      self.lap_kernel, mode='reflect')
                total += np.sqrt(np.mean(lap**2))
        return total / (len(imgs) * imgs[0].shape[2])
    
    def image_energy(self, imgs: np.ndarray) -> float:
        """||h||₂"""
        if imgs.ndim == 3:
            imgs = imgs[np.newaxis, ...]
        return np.mean([np.sqrt(np.mean(img**2)) for img in imgs])
    
    def postprocess_decoded(self, decoded: torch.Tensor, size: int, do_resize: bool = True) -> np.ndarray:
        return self.image_processor.postprocess(decoded, do_resize)

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
        
        # Text
        exp_prompts = [p for p in prompts for _ in range(S)]
        tok = tokenizer(exp_prompts, padding="max_length", max_length=77,
                       truncation=True, return_tensors="pt")
        text_emb = text_encoder(tok.input_ids.to(self.device)).last_hidden_state
        
        # Encode
        with torch.no_grad():
            z = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
        z = z.repeat_interleave(S, dim=0).half()
        
        # Noise
        noise = torch.randn_like(z, device=self.device).half()
        sqrt_d = torch.prod(torch.tensor(z.shape[1:])).float().sqrt()
        u = self.normalize_batch(noise, self.config.epsilon_reg).half() * sqrt_d
        
        # Timestep & forward
        t = self.config.time_frac * scheduler.config.num_train_timesteps
        t_tensor = torch.full((z.shape[0],), t, device=self.device, dtype=torch.long)
        z_noisy = scheduler.add_noise(z, u, t_tensor).half()
        
        with torch.no_grad():
            h = unet(z_noisy, t_tensor, encoder_hidden_states=text_emb)[0]
        h_scaled = h / vae.config.scaling_factor
        
        del z_noisy, text_emb, noise
        
        # Decode
        def decode(x):
            with torch.no_grad():
                return vae.decode(x, return_dict=False)[0]
        
        h_dec = decode(h_scaled)
        u_dec = decode(u / vae.config.scaling_factor)
        
        del h, h_scaled, u, z
        
        sz = self.config.image_size
        h_np = self.postprocess_decoded(h_dec, sz)
        u_np = self.postprocess_decoded(u_dec, sz)
        
        del h_dec, u_dec
        
        h_chunks = numpy_chunk(h_np, B)
        u_chunks = numpy_chunk(u_np, B)
        
        # Compute
        results = []
        for i, (h_i, u_i, raw_i) in enumerate(zip(h_chunks, u_chunks, images_raw)):
            
            img = raw_i.float().cpu().numpy()
            if img.ndim == 4: img = img[0]
            if img.shape[0] == 3: img = np.transpose(img, (1,2,0))
            img = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
            
            # CLIP features
            x0_f = self.get_clip_features(img).squeeze(0)
            h_f = self.get_clip_features(h_i)
            u_f = self.get_clip_features(u_i)
            
            n = h_f.shape[0]
            
            # C₀: 主项
            h_norm = torch.norm(h_f, p=2, dim=1, keepdim=True) + 1e-8
            h_dir = -h_f / h_norm
            x0_exp = x0_f.unsqueeze(0).expand(n, -1)
            vec = self.a * u_f - self.b * h_f + self.c * self.sqrt_d_clip * x0_exp
            
            C0 = self.cos(h_dir, vec).mean().item()
            C0_norm = (C0 + 1) / (self.a + self.b + self.c + 1)
            
            # C₁: 高阶修正
            lap_e = self.laplacian_energy(h_i)
            img_e = self.image_energy(h_i) + 1e-8
            C1 = lap_e / img_e
            C1_norm = np.clip(C1 / 10.0, 0, 1)
            
            # 总 criterion
            C = C0_norm + self.eps * C1_norm
            
            res = {"criterion": float(C)}
            
            if self.config.return_terms:
                res.update({
                    "C0": float(C0_norm),
                    "C1": float(C1_norm),
                    "eps": float(self.eps),
                })
            
            results.append(res)
        
        return results