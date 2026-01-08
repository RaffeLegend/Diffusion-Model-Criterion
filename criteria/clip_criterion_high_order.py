"""
High-Order Enhanced CLIP Criterion

核心改动：在 CLIP 空间显式引入高阶微分信息

原文: x0_clip = CLIP(x0)
改进: x0_clip_enhanced = CLIP(x0) + λ · CLIP(Δx0)

理论依据：
- CLIP 等低阶特征提取器对高阶微分结构不敏感
- 显式引入 Laplacian 响应补充高阶信息
- 在统一的 CLIP 空间中融合，保持原文框架
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import ndimage

from criteria.base import BaseCriterion, register_criterion
from image_utils import ImageProcessor, numpy_chunk


class DifferentialOperators:
    """微分算子"""
    
    # Laplacian kernel (5x5)
    LAPLACIAN = np.array([
        [0,  0, -1,  0,  0],
        [0, -1, -2, -1,  0],
        [-1, -2, 16, -2, -1],
        [0, -1, -2, -1,  0],
        [0,  0, -1,  0,  0]
    ], dtype=np.float32) / 4.0
    
    @staticmethod
    def apply_laplacian(image: np.ndarray) -> np.ndarray:
        """Apply Laplacian operator Δ"""
        if image.ndim == 2:
            return ndimage.convolve(image.astype(np.float32), 
                                   DifferentialOperators.LAPLACIAN, mode='reflect')
        elif image.ndim == 3:
            result = np.zeros_like(image, dtype=np.float32)
            for c in range(image.shape[2]):
                result[:,:,c] = ndimage.convolve(image[:,:,c].astype(np.float32), 
                                                 DifferentialOperators.LAPLACIAN, mode='reflect')
            return result
        else:
            raise ValueError(f"Expected 2D or 3D, got {image.ndim}D")
    
    @staticmethod
    def normalize_for_clip(response: np.ndarray) -> np.ndarray:
        """Normalize differential response to [0, 255] for CLIP"""
        # 取绝对值（我们关心的是响应强度）
        response = np.abs(response)
        
        # 归一化到 [0, 255]
        if response.max() > response.min():
            response = (response - response.min()) / (response.max() - response.min()) * 255
        
        return response.astype(np.uint8)


@register_criterion("clip_high_order")
class HighOrderCLIPCriterion(BaseCriterion):
    """
    High-Order Enhanced CLIP Criterion
    
    改进点：在 CLIP 特征空间显式引入高阶微分信息
    
    原文公式:
        C(x0) = ⟨-h/||h||, a·u_d - b·h + c·√d·x0⟩
    
    改进公式:
        x0_enhanced = CLIP(x0) + λ · CLIP(Δx0)
        h_enhanced = CLIP(h) + λ · CLIP(Δh)
        C(x0) = ⟨-h_enhanced/||h_enhanced||, a·u_d - b·h_enhanced + c·√d·x0_enhanced⟩
    
    理论依据:
        定理 (High-Order Feature Decoupling): 
        低阶特征提取器 Φ 满足 ||Φ(I) - Φ(I+ε)||→0 当 ||ε||_{L²}→0，
        即使 ||Δε||_{L²}→∞。显式引入 Δ 可捕捉这些高阶变化。
    """
    
    def __init__(self, config, model_manager):
        super().__init__(config, model_manager)
        
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        
        # CLIP dimension
        self.d_clip = 768
        self.sqrt_d_clip = self.d_clip ** 0.5
        
        # Manifold criterion coefficients
        self.a = 1.0
        self.b = 1.0
        self.c = 1.0
        
        # 高阶增强权重 λ
        self.lambda_ho = getattr(config, 'lambda_high_order', 0.3)
        
        self.image_processor = ImageProcessor(config)
        self.diff_ops = DifferentialOperators()
    
    def compute_clip_features(self, images: np.ndarray) -> torch.Tensor:
        """Compute CLIP image features"""
        clip_model = self.model_manager.clip
        processor = self.model_manager.clip_processor
        
        # Handle single image
        if images.ndim == 3:
            images = images[np.newaxis, ...]
        
        inputs = processor(images=list(images), return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            features = clip_model.get_image_features(**inputs)
        
        return features.detach().cpu()
    
    def compute_enhanced_features(self, images: np.ndarray) -> torch.Tensor:
        """
        计算增强的 CLIP 特征: CLIP(I) + λ·CLIP(ΔI)
        
        这是核心改动：显式引入高阶微分信息到特征空间
        """
        if images.ndim == 3:
            images = images[np.newaxis, ...]
        
        batch_size = images.shape[0]
        
        # 1. 原始图像的 CLIP 特征
        clip_original = self.compute_clip_features(images)
        
        # 2. Laplacian 响应的 CLIP 特征
        laplacian_images = []
        for i in range(batch_size):
            lap = self.diff_ops.apply_laplacian(images[i])
            lap_normalized = self.diff_ops.normalize_for_clip(lap)
            laplacian_images.append(lap_normalized)
        
        laplacian_images = np.stack(laplacian_images, axis=0)
        clip_laplacian = self.compute_clip_features(laplacian_images)
        
        # 3. 融合: enhanced = original + λ · laplacian
        # 注意：需要归一化以保持特征尺度
        clip_laplacian_normalized = F.normalize(clip_laplacian, p=2, dim=1)
        clip_original_normalized = F.normalize(clip_original, p=2, dim=1)
        
        enhanced = clip_original_normalized + self.lambda_ho * clip_laplacian_normalized
        
        return enhanced
    
    def postprocess_decoded(self, decoded: torch.Tensor, size: int, do_resize: bool = True) -> np.ndarray:
        return self.image_processor.postprocess(decoded, do_resize)
    
    def evaluate_batch(
        self,
        images: torch.Tensor,
        prompts: List[str],
        images_raw: Optional[List[torch.Tensor]] = None
    ) -> List[Dict]:
        """Evaluate batch with high-order enhanced features"""
        
        if images_raw is None:
            raise ValueError("HighOrderCLIPCriterion requires images_raw")
        
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
        
        # --------- Encode images ---------
        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
        latents = latents.repeat_interleave(num_noise, dim=0).half()
        
        # --------- Spherical noise ---------
        gauss_noise = torch.randn_like(latents, device=self.device).half()
        sqrt_d_latent = torch.prod(torch.tensor(latents.shape[1:])).float().sqrt()
        spherical_noise = self.normalize_batch(gauss_noise, self.config.epsilon_reg).half()
        spherical_noise = spherical_noise * sqrt_d_latent
        u_d_latent = spherical_noise.clone()
        
        # --------- Timestep ---------
        timestep = self.config.time_frac * scheduler.config.num_train_timesteps
        timestep = torch.full((latents.shape[0],), timestep, device=self.device, dtype=torch.long)
        
        # --------- Add noise & predict ---------
        noisy_latents = scheduler.add_noise(latents, spherical_noise, timestep).half()
        
        with torch.no_grad():
            noise_pred = unet(noisy_latents, timestep, encoder_hidden_states=text_emb)[0]
        
        noise_pred_scaled = noise_pred / vae.config.scaling_factor
        
        del noisy_latents, gauss_noise, text_emb
        
        # --------- Decode ---------
        sub_batch_size = 16
        
        def batch_decode(latent_batch):
            if latent_batch.size(0) <= sub_batch_size:
                with torch.no_grad():
                    return vae.decode(latent_batch, return_dict=False)[0]
            decoded_list = []
            for i in range((latent_batch.size(0) + sub_batch_size - 1) // sub_batch_size):
                start = i * sub_batch_size
                end = min((i+1) * sub_batch_size, latent_batch.size(0))
                torch.cuda.empty_cache()
                with torch.no_grad():
                    decoded_list.append(vae.decode(latent_batch[start:end], return_dict=False)[0])
            return torch.cat(decoded_list, dim=0)
        
        decoded_noise = batch_decode(noise_pred_scaled)
        decoded_u_d = batch_decode(u_d_latent / vae.config.scaling_factor)
        
        del noise_pred, noise_pred_scaled, spherical_noise, u_d_latent, latents
        
        # --------- Postprocess ---------
        siz = self.config.image_size
        decoded_noise_np = self.postprocess_decoded(decoded_noise, siz)
        decoded_u_d_np = self.postprocess_decoded(decoded_u_d, siz)
        
        del decoded_noise, decoded_u_d
        
        decoded_noise_chunks = numpy_chunk(decoded_noise_np, num_images)
        decoded_u_d_chunks = numpy_chunk(decoded_u_d_np, num_images)
        
        # --------- Compute criterion ---------
        results = []
        
        for i, (h_chunk, ud_chunk, cur_image_raw) in enumerate(
            zip(decoded_noise_chunks, decoded_u_d_chunks, images_raw)
        ):
            # Prepare original image
            img_np = cur_image_raw.float().cpu().numpy()
            if img_np.ndim == 4:
                img_np = img_np[0]
            if img_np.shape[0] == 3:
                img_np = np.transpose(img_np, (1, 2, 0))
            img_np_uint8 = (img_np * 255).astype(np.uint8) if img_np.max() <= 1 else img_np.astype(np.uint8)
            
            # ===== 核心改动：使用增强特征 =====
            # x0: 原图的增强特征
            x0_enhanced = self.compute_enhanced_features(img_np_uint8).squeeze(0)
            
            # h: 噪声预测的增强特征
            h_enhanced = self.compute_enhanced_features(h_chunk)
            
            # u_d: 球面噪声的特征（不需要增强，因为是噪声）
            ud_clip = self.compute_clip_features(ud_chunk)
            
            # ===== 计算 criterion =====
            s = h_enhanced.shape[0]
            
            # Normalize h: -h / ||h||
            h_norms = torch.norm(h_enhanced, p=2, dim=1, keepdim=True) + 1e-8
            h_normalized = -h_enhanced / h_norms
            
            # Expand x0
            x0_expanded = x0_enhanced.unsqueeze(0).expand(s, -1)
            
            # Combined: a·u_d - b·h + c·√d·x0
            combined = (self.a * ud_clip - 
                       self.b * h_enhanced + 
                       self.c * self.sqrt_d_clip * x0_expanded)
            
            # Inner product
            inner_products = self.cos(h_normalized, combined)
            
            # Criterion
            C = inner_products.mean().item()
            C_normalized = (C + 1) / (self.a + self.b + self.c + 1)
            
            result = {"criterion": float(C_normalized)}
            
            if self.config.return_terms:
                # 也计算原始特征的 criterion 用于对比
                x0_original = self.compute_clip_features(img_np_uint8).squeeze(0)
                h_original = self.compute_clip_features(h_chunk)
                
                h_orig_norms = torch.norm(h_original, p=2, dim=1, keepdim=True) + 1e-8
                h_orig_normalized = -h_original / h_orig_norms
                x0_orig_expanded = x0_original.unsqueeze(0).expand(s, -1)
                combined_orig = (self.a * ud_clip - self.b * h_original + 
                                self.c * self.sqrt_d_clip * x0_orig_expanded)
                C_orig = self.cos(h_orig_normalized, combined_orig).mean().item()
                C_orig_normalized = (C_orig + 1) / (self.a + self.b + self.c + 1)
                
                result.update({
                    "C_original": float(C_orig_normalized),
                    "C_enhanced": float(C_normalized),
                    "lambda_ho": float(self.lambda_ho),
                })
            
            results.append(result)
        
        return results