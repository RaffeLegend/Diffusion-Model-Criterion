"""
Enhanced CLIP-based Manifold Criterion with High-Order Differential Analysis

融合原文Manifold方法与高阶微分分析的实现。

理论基础：
1. 原文方法通过score function分析manifold的曲率
2. 曲率本质上涉及二阶微分结构
3. 生成模型的低阶优化不约束高阶结构
4. 显式加入高阶微分项可以增强检测能力

实现策略：
- 保留原文CLIP空间的计算框架（保证基础精度）
- 增加图像的Laplacian/Biharmonic响应
- 在CLIP空间融合低阶和高阶特征
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import ndimage

from criteria.base import BaseCriterion, register_criterion
from image_utils import ImageProcessor, numpy_chunk


class HighOrderOperators:
    """高阶微分算子"""
    
    # Laplacian kernel (3x3)
    LAPLACIAN_3x3 = np.array([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0]
    ], dtype=np.float32)
    
    # Laplacian kernel (5x5) - more stable
    LAPLACIAN_5x5 = np.array([
        [0,  0, -1,  0,  0],
        [0, -1, -2, -1,  0],
        [-1, -2, 16, -2, -1],
        [0, -1, -2, -1,  0],
        [0,  0, -1,  0,  0]
    ], dtype=np.float32) / 4.0
    
    # Biharmonic kernel (∆²) - 5x5 approximation
    # ∆² = ∂⁴/∂x⁴ + 2∂⁴/∂x²∂y² + ∂⁴/∂y⁴
    BIHARMONIC_5x5 = np.array([
        [0,  0,  1,  0,  0],
        [0,  2, -8,  2,  0],
        [1, -8, 20, -8,  1],
        [0,  2, -8,  2,  0],
        [0,  0,  1,  0,  0]
    ], dtype=np.float32)
    
    @staticmethod
    def apply_laplacian(image: np.ndarray, use_5x5: bool = True) -> np.ndarray:
        """
        Apply Laplacian operator to image.
        
        Args:
            image: (H, W, C) or (H, W) numpy array
            use_5x5: Use 5x5 kernel for better stability
        
        Returns:
            Laplacian response, same shape as input
        """
        kernel = HighOrderOperators.LAPLACIAN_5x5 if use_5x5 else HighOrderOperators.LAPLACIAN_3x3
        
        if image.ndim == 2:
            return ndimage.convolve(image.astype(np.float32), kernel, mode='reflect')
        elif image.ndim == 3:
            result = np.zeros_like(image, dtype=np.float32)
            for c in range(image.shape[2]):
                result[:, :, c] = ndimage.convolve(image[:, :, c].astype(np.float32), kernel, mode='reflect')
            return result
        else:
            raise ValueError(f"Expected 2D or 3D image, got {image.ndim}D")
    
    @staticmethod
    def apply_biharmonic(image: np.ndarray) -> np.ndarray:
        """
        Apply Biharmonic operator (∆²) to image.
        
        Args:
            image: (H, W, C) or (H, W) numpy array
        
        Returns:
            Biharmonic response, same shape as input
        """
        kernel = HighOrderOperators.BIHARMONIC_5x5
        
        if image.ndim == 2:
            return ndimage.convolve(image.astype(np.float32), kernel, mode='reflect')
        elif image.ndim == 3:
            result = np.zeros_like(image, dtype=np.float32)
            for c in range(image.shape[2]):
                result[:, :, c] = ndimage.convolve(image[:, :, c].astype(np.float32), kernel, mode='reflect')
            return result
        else:
            raise ValueError(f"Expected 2D or 3D image, got {image.ndim}D")
    
    @staticmethod
    def compute_gradient_magnitude(image: np.ndarray) -> np.ndarray:
        """Compute gradient magnitude |∇I|"""
        if image.ndim == 3:
            # Convert to grayscale for gradient
            gray = np.mean(image, axis=2)
        else:
            gray = image
        
        gx = ndimage.sobel(gray.astype(np.float32), axis=1)
        gy = ndimage.sobel(gray.astype(np.float32), axis=0)
        return np.sqrt(gx**2 + gy**2)
    
    @staticmethod
    def compute_dsir(image: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """
        Compute Differential Structural Invariant Residual (DSIR).
        
        DSIR = |∆²I| / (|∇I| + ε)
        
        This normalizes high-order response by gradient magnitude,
        isolating structural irregularities.
        """
        biharmonic = np.abs(HighOrderOperators.apply_biharmonic(image))
        grad_mag = HighOrderOperators.compute_gradient_magnitude(image)
        
        if image.ndim == 3:
            # Expand grad_mag to match biharmonic shape
            grad_mag = np.expand_dims(grad_mag, axis=2)
        
        dsir = biharmonic / (grad_mag + eps)
        return dsir
    
    @staticmethod
    def compute_high_order_energy(image: np.ndarray, order: str = 'biharmonic') -> float:
        """
        Compute high-order structural energy R(I) = ||∆²I||_L2
        """
        if order == 'laplacian':
            response = HighOrderOperators.apply_laplacian(image)
        elif order == 'biharmonic':
            response = HighOrderOperators.apply_biharmonic(image)
        else:
            raise ValueError(f"Unknown order: {order}")
        
        return np.sqrt(np.mean(response**2))


@register_criterion("clip_enhanced")
class EnhancedCLIPCriterion(BaseCriterion):
    """
    Enhanced CLIP criterion with high-order differential analysis.
    
    融合策略：
    1. 计算原文的manifold criterion (在CLIP空间)
    2. 增加高阶微分特征 (Laplacian/Biharmonic响应的CLIP特征)
    3. 组合两者得到增强criterion
    
    理论依据：
    - Manifold curvature κ 本质上涉及二阶微分
    - 显式加入高阶算子可以更直接地捕捉结构缺陷
    - 两者在CLIP空间融合可以获得互补信息
    """
    
    def __init__(self, config, model_manager):
        super().__init__(config, model_manager)
        
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        
        # CLIP feature dimension
        self.d_clip = 768
        self.sqrt_d_clip = self.d_clip ** 0.5
        
        # Manifold criterion coefficients (原文)
        self.a = 1.0
        self.b = 1.0
        self.c = 1.0
        
        # High-order enhancement weight
        # 这个参数控制高阶项的贡献，可以通过验证集调整
        self.lambda_high_order = getattr(config, 'lambda_high_order', 0.1)
        
        # High-order operator type
        self.high_order_type = getattr(config, 'high_order_type', 'laplacian')
        
        self.image_processor = ImageProcessor(config)
        self.high_order_ops = HighOrderOperators()
    
    def compute_clip_features(self, images: np.ndarray) -> torch.Tensor:
        """Compute CLIP image features."""
        clip_model = self.model_manager.clip
        processor = self.model_manager.clip_processor
        
        inputs = processor(images=images, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            features = clip_model.get_image_features(**inputs)
        
        return features.detach().cpu()
    
    def compute_high_order_features(self, images: np.ndarray) -> Tuple[torch.Tensor, float]:
        """
        Compute high-order differential features.
        
        Returns:
            - CLIP features of high-order response
            - High-order energy (scalar)
        """
        batch_size = images.shape[0] if images.ndim == 4 else 1
        
        if images.ndim == 3:
            images = images[np.newaxis, ...]
        
        high_order_images = []
        energies = []
        
        for i in range(batch_size):
            img = images[i]  # (H, W, C)
            
            # Apply high-order operator
            if self.high_order_type == 'laplacian':
                response = self.high_order_ops.apply_laplacian(img)
            elif self.high_order_type == 'biharmonic':
                response = self.high_order_ops.apply_biharmonic(img)
            elif self.high_order_type == 'dsir':
                response = self.high_order_ops.compute_dsir(img)
            else:
                raise ValueError(f"Unknown high_order_type: {self.high_order_type}")
            
            # Normalize to [0, 255] for CLIP processing
            response_normalized = response - response.min()
            if response_normalized.max() > 0:
                response_normalized = response_normalized / response_normalized.max() * 255
            response_normalized = response_normalized.astype(np.uint8)
            
            high_order_images.append(response_normalized)
            energies.append(np.sqrt(np.mean(response**2)))
        
        high_order_images = np.stack(high_order_images, axis=0)
        
        # Get CLIP features of high-order response
        high_order_clip = self.compute_clip_features(high_order_images)
        
        return high_order_clip, np.mean(energies)
    
    def compute_manifold_criterion(
        self,
        x0_clip: torch.Tensor,      # (dim,)
        h_clips: torch.Tensor,       # (s, dim)
        ud_clips: torch.Tensor,      # (s, dim)
    ) -> float:
        """
        计算原文的Manifold Criterion (在CLIP空间)
        
        公式: C(x0) = 1/s * Σ⟨-h/||h||, a*u_d - b*h + c*√d*x0⟩
        """
        s = h_clips.shape[0]
        
        # Normalize h: -h / ||h||
        h_norms = torch.norm(h_clips, p=2, dim=1, keepdim=True) + 1e-8
        h_normalized = -h_clips / h_norms
        
        # Expand x0
        x0_expanded = x0_clip.unsqueeze(0).expand(s, -1)
        
        # Combined term: a*u_d - b*h + c*√d*x0
        combined = (self.a * ud_clips - 
                    self.b * h_clips + 
                    self.c * self.sqrt_d_clip * x0_expanded)
        
        # Inner product (cosine similarity in CLIP space)
        inner_products = self.cos(h_normalized, combined)
        
        # Normalize to [0, 1]
        C = inner_products.mean().item()
        C_normalized = (C + 1) / (self.a + self.b + self.c + 1)
        
        return C_normalized
    
    def compute_high_order_criterion(
        self,
        x0_clip: torch.Tensor,           # (dim,)
        x0_high_order_clip: torch.Tensor, # (dim,)
        h_clips: torch.Tensor,            # (s, dim)
        h_high_order_clips: torch.Tensor, # (s, dim)
    ) -> float:
        """
        计算高阶微分增强项
        
        理论依据：
        - 生成图像的高阶响应与原图的关系与真实图像不同
        - 在CLIP空间度量这种差异
        
        计算方式：
        R = cos(CLIP(∆²x0), CLIP(∆²h)) - cos(CLIP(x0), CLIP(h))
        
        对于真实图像，高阶响应的相似性模式应该与低阶一致
        对于生成图像，高阶响应会有更大的偏差
        """
        s = h_clips.shape[0]
        
        # Low-order similarity: cos(x0, h)
        x0_expanded = x0_clip.unsqueeze(0).expand(s, -1)
        low_order_sim = self.cos(x0_expanded, h_clips).mean()
        
        # High-order similarity: cos(∆²x0, ∆²h)
        x0_ho_expanded = x0_high_order_clip.unsqueeze(0).expand(s, -1)
        high_order_sim = self.cos(x0_ho_expanded, h_high_order_clips).mean()
        
        # 差异：如果高阶相似性与低阶不一致，说明有结构缺陷
        # 对于生成图像，我们预期 high_order_sim 会相对更低
        R = (high_order_sim - low_order_sim).item()
        
        # 归一化到 [0, 1]，R越小说明越可能是生成图像
        R_normalized = (R + 1) / 2
        
        return R_normalized
    
    def postprocess_decoded(self, decoded: torch.Tensor, size: int, do_resize: bool = True) -> np.ndarray:
        return self.image_processor.postprocess(decoded, do_resize)
    
    def evaluate_batch(
        self,
        images: torch.Tensor,
        prompts: List[str],
        images_raw: Optional[List[torch.Tensor]] = None
    ) -> List[Dict]:
        """
        Evaluate images using enhanced CLIP criterion.
        """
        if images_raw is None:
            raise ValueError("EnhancedCLIPCriterion requires images_raw parameter")
        
        num_images = images.shape[0]
        num_noise = self.config.num_noise
        
        # Get model components
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
            expanded_prompts, 
            padding="max_length", 
            max_length=77, 
            truncation=True, 
            return_tensors="pt"
        )
        input_ids = text_tokens.input_ids.to(self.device)
        text_emb = text_encoder(input_ids).last_hidden_state
        
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
        
        # --------- Add noise ---------
        noisy_latents = scheduler.add_noise(
            original_samples=latents, 
            noise=spherical_noise, 
            timesteps=timestep
        ).half()
        
        # --------- Predict noise ---------
        with torch.no_grad():
            noise_pred = unet(noisy_latents, timestep, encoder_hidden_states=text_emb)[0]
        
        noise_pred_scaled = noise_pred / vae.config.scaling_factor
        
        del noisy_latents, timestep, gauss_noise, text_emb, input_ids
        
        # --------- Decode ---------
        sub_batch_size = 16
        
        def batch_decode(latent_batch):
            if latent_batch.size(0) <= sub_batch_size:
                with torch.no_grad():
                    return vae.decode(latent_batch, return_dict=False)[0]
            else:
                decoded_list = []
                num_sub_batches = (latent_batch.size(0) + sub_batch_size - 1) // sub_batch_size
                with torch.no_grad():
                    for i in range(num_sub_batches):
                        start_idx = i * sub_batch_size
                        end_idx = min((i + 1) * sub_batch_size, latent_batch.size(0))
                        torch.cuda.empty_cache()
                        decoded_sub = vae.decode(latent_batch[start_idx:end_idx], return_dict=False)[0]
                        decoded_list.append(decoded_sub)
                return torch.cat(decoded_list, dim=0)
        
        decoded_noise = batch_decode(noise_pred_scaled)
        u_d_for_decode = u_d_latent / vae.config.scaling_factor
        decoded_u_d = batch_decode(u_d_for_decode)
        
        del noise_pred, noise_pred_scaled, spherical_noise, u_d_latent, u_d_for_decode, latents
        
        # --------- Postprocess ---------
        siz = self.config.image_size
        decoded_noise_np = self.postprocess_decoded(decoded_noise, siz, do_resize=True)
        decoded_u_d_np = self.postprocess_decoded(decoded_u_d, siz, do_resize=True)
        
        del decoded_noise, decoded_u_d
        
        decoded_noise_chunks = numpy_chunk(decoded_noise_np, num_images)
        decoded_u_d_chunks = numpy_chunk(decoded_u_d_np, num_images)
        
        # --------- Compute criterion for each image ---------
        results = []
        for i, (h_chunk, ud_chunk, cur_image_raw) in enumerate(
            zip(decoded_noise_chunks, decoded_u_d_chunks, images_raw)
        ):
            # 1. Original image features
            img_np = cur_image_raw.float().cpu().numpy()
            if img_np.ndim == 4:
                img_np = img_np[0]  # Remove batch dim
            if img_np.shape[0] == 3:  # (C, H, W) -> (H, W, C)
                img_np = np.transpose(img_np, (1, 2, 0))
            img_np_uint8 = (img_np * 255).astype(np.uint8) if img_np.max() <= 1 else img_np.astype(np.uint8)
            
            x0_clip = self.compute_clip_features(img_np_uint8[np.newaxis, ...]).squeeze(0)
            
            # 2. High-order features of original image
            x0_high_order_clip, x0_energy = self.compute_high_order_features(img_np_uint8)
            x0_high_order_clip = x0_high_order_clip.squeeze(0)
            
            # 3. Noise prediction features
            h_clip = self.compute_clip_features(h_chunk)
            
            # 4. High-order features of noise prediction
            h_high_order_clip, h_energy = self.compute_high_order_features(h_chunk)
            
            # 5. Spherical noise features
            ud_clip = self.compute_clip_features(ud_chunk)
            
            # 6. Compute manifold criterion (原文方法)
            C_manifold = self.compute_manifold_criterion(x0_clip, h_clip, ud_clip)
            
            # 7. Compute high-order criterion (增强项)
            C_high_order = self.compute_high_order_criterion(
                x0_clip, x0_high_order_clip, 
                h_clip, h_high_order_clip
            )
            
            # 8. Combine (加权融合)
            criterion = C_manifold + self.lambda_high_order * C_high_order
            
            result = {"criterion": float(criterion)}
            
            if self.config.return_terms:
                result.update({
                    "C_manifold": float(C_manifold),
                    "C_high_order": float(C_high_order),
                    "x0_energy": float(x0_energy),
                    "h_energy": float(h_energy),
                    "lambda": float(self.lambda_high_order),
                })
            
            results.append(result)
        
        return results


@register_criterion("clip_with_dsir")
class CLIPWithDSIRCriterion(BaseCriterion):
    """
    简化版本：原文CLIP criterion + DSIR energy
    
    这是最小改动方案：
    - 完全保留原文的CLIP criterion计算
    - 仅添加DSIR能量作为额外特征
    - 不改变原文的核心逻辑
    
    适合快速验证想法。
    """
    
    def __init__(self, config, model_manager):
        super().__init__(config, model_manager)
        
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.d_clip = 768
        self.sqrt_d_clip = self.d_clip ** 0.5
        self.a = self.b = self.c = 1.0
        
        # DSIR weight
        self.dsir_weight = getattr(config, 'dsir_weight', 0.01)
        
        self.image_processor = ImageProcessor(config)
        self.high_order_ops = HighOrderOperators()
    
    def compute_clip_features(self, images: np.ndarray) -> torch.Tensor:
        clip_model = self.model_manager.clip
        processor = self.model_manager.clip_processor
        inputs = processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = clip_model.get_image_features(**inputs)
        return features.detach().cpu()
    
    def compute_dsir_energy(self, image: np.ndarray) -> float:
        """计算DSIR能量"""
        dsir = self.high_order_ops.compute_dsir(image)
        return float(np.mean(np.abs(dsir)))
    
    def postprocess_decoded(self, decoded: torch.Tensor, size: int, do_resize: bool = True) -> np.ndarray:
        return self.image_processor.postprocess(decoded, do_resize)
    
    def evaluate_batch(
        self,
        images: torch.Tensor,
        prompts: List[str],
        images_raw: Optional[List[torch.Tensor]] = None
    ) -> List[Dict]:
        """与原文方法基本相同，仅增加DSIR项"""
        if images_raw is None:
            raise ValueError("CLIPWithDSIRCriterion requires images_raw parameter")
        
        num_images = images.shape[0]
        num_noise = self.config.num_noise
        
        unet = self.model_manager.unet
        vae = self.model_manager.vae
        scheduler = self.model_manager.scheduler
        tokenizer = self.model_manager.tokenizer
        text_encoder = self.model_manager.text_encoder
        
        # Text embeddings
        expanded_prompts = []
        for prompt in prompts:
            expanded_prompts.extend([prompt] * num_noise)
        
        text_tokens = tokenizer(expanded_prompts, padding="max_length", max_length=77, 
                               truncation=True, return_tensors="pt")
        input_ids = text_tokens.input_ids.to(self.device)
        text_emb = text_encoder(input_ids).last_hidden_state
        
        # Encode
        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
        latents = latents.repeat_interleave(num_noise, dim=0).half()
        
        # Spherical noise
        gauss_noise = torch.randn_like(latents, device=self.device).half()
        sqrt_d_latent = torch.prod(torch.tensor(latents.shape[1:])).float().sqrt()
        spherical_noise = self.normalize_batch(gauss_noise, self.config.epsilon_reg).half()
        spherical_noise = spherical_noise * sqrt_d_latent
        u_d_latent = spherical_noise.clone()
        
        # Timestep
        timestep = self.config.time_frac * scheduler.config.num_train_timesteps
        timestep = torch.full((latents.shape[0],), timestep, device=self.device, dtype=torch.long)
        
        # Add noise & predict
        noisy_latents = scheduler.add_noise(latents, spherical_noise, timestep).half()
        with torch.no_grad():
            noise_pred = unet(noisy_latents, timestep, encoder_hidden_states=text_emb)[0]
        noise_pred_scaled = noise_pred / vae.config.scaling_factor
        
        del noisy_latents, timestep, gauss_noise, text_emb, input_ids
        
        # Decode
        sub_batch_size = 16
        def batch_decode(latent_batch):
            if latent_batch.size(0) <= sub_batch_size:
                with torch.no_grad():
                    return vae.decode(latent_batch, return_dict=False)[0]
            decoded_list = []
            for i in range((latent_batch.size(0) + sub_batch_size - 1) // sub_batch_size):
                start, end = i * sub_batch_size, min((i+1) * sub_batch_size, latent_batch.size(0))
                torch.cuda.empty_cache()
                with torch.no_grad():
                    decoded_list.append(vae.decode(latent_batch[start:end], return_dict=False)[0])
            return torch.cat(decoded_list, dim=0)
        
        decoded_noise = batch_decode(noise_pred_scaled)
        decoded_u_d = batch_decode(u_d_latent / vae.config.scaling_factor)
        
        del noise_pred, noise_pred_scaled, spherical_noise, u_d_latent, latents
        
        siz = self.config.image_size
        decoded_noise_np = self.postprocess_decoded(decoded_noise, siz, do_resize=True)
        decoded_u_d_np = self.postprocess_decoded(decoded_u_d, siz, do_resize=True)
        
        del decoded_noise, decoded_u_d
        
        decoded_noise_chunks = numpy_chunk(decoded_noise_np, num_images)
        decoded_u_d_chunks = numpy_chunk(decoded_u_d_np, num_images)
        
        # Compute criterion
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
            
            # CLIP features
            x0_clip = self.compute_clip_features(img_np_uint8[np.newaxis, ...]).squeeze(0)
            h_clip = self.compute_clip_features(h_chunk)
            ud_clip = self.compute_clip_features(ud_chunk)
            
            # Original manifold criterion
            s = h_clip.shape[0]
            h_norms = torch.norm(h_clip, p=2, dim=1, keepdim=True) + 1e-8
            h_normalized = -h_clip / h_norms
            x0_expanded = x0_clip.unsqueeze(0).expand(s, -1)
            combined = self.a * ud_clip - self.b * h_clip + self.c * self.sqrt_d_clip * x0_expanded
            inner_products = self.cos(h_normalized, combined)
            C_manifold = (inner_products.mean().item() + 1) / (self.a + self.b + self.c + 1)
            
            # DSIR energy (高阶项)
            dsir_x0 = self.compute_dsir_energy(img_np_uint8)
            dsir_h = np.mean([self.compute_dsir_energy(h_chunk[j]) for j in range(h_chunk.shape[0])])
            
            # 组合：原文criterion + DSIR差异
            # 生成图像的DSIR能量通常更高
            dsir_diff = dsir_h - dsir_x0
            criterion = C_manifold + self.dsir_weight * dsir_diff
            
            result = {"criterion": float(criterion)}
            
            if self.config.return_terms:
                result.update({
                    "C_manifold": float(C_manifold),
                    "dsir_x0": float(dsir_x0),
                    "dsir_h": float(dsir_h),
                    "dsir_diff": float(dsir_diff),
                })
            
            results.append(result)
        
        return results