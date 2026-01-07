"""
CLIP-based Manifold Criterion (Fixed Version)

根据论文 "Manifold Induced Biases for Zero-shot and Few-shot Detection of Generated Images"
修正的实现。

主要修正：
1. CLIP特征维度改为768 (clip-vit-large-patch14)
2. 按照论文Section 4.3的公式计算criterion
3. 将u_d, h, x0都映射到CLIP空间后再计算
4. 使用cosine similarity作为内积
"""

import torch
import numpy as np
from typing import Dict, List, Optional

from criteria.base import BaseCriterion, register_criterion
from image_utils import ImageProcessor, numpy_chunk


@register_criterion("clip")
class CLIPCriterion(BaseCriterion):
    """
    CLIP-based criterion for evaluating diffusion model outputs.
    
    按照论文公式:
    C(x0) = 1/s * Σ⟨-h(x̃)/||h(x̃)||, a*u_d - b*h(x̃) + c*√d*x0⟩
    
    其中 a=b=c=1，所有向量都先映射到CLIP空间
    """
    
    def __init__(self, config, model_manager):
        super().__init__(config, model_manager)
        
        # Cosine similarity
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        
        # CLIP feature dimension - clip-vit-large-patch14 输出是 768 维
        self.d_clip = 768
        self.sqrt_d_clip = self.d_clip ** 0.5
        
        # 论文中的系数 a, b, c
        self.a = 1.0
        self.b = 1.0
        self.c = 1.0
        
        self.image_processor = ImageProcessor(config)
    
    def compute_clip_features(self, images: np.ndarray) -> torch.Tensor:
        """
        Compute CLIP image features.
        
        Args:
            images: NumPy array of shape (B, H, W, C) with uint8 values
                   或 torch.Tensor
            
        Returns:
            CLIP features tensor of shape (B, dim)
        """
        clip_model = self.model_manager.clip
        processor = self.model_manager.clip_processor
        
        # Process images for CLIP
        inputs = processor(images=images, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            features = clip_model.get_image_features(**inputs)
        
        return features.detach().cpu()
    
    def postprocess_decoded(
        self, 
        decoded: torch.Tensor, 
        size: int, 
        do_resize: bool = True
    ) -> np.ndarray:
        """
        Postprocess decoded latents to uint8 numpy array.
        """
        return self.image_processor.postprocess(decoded, do_resize)
    
    def compute_single_criterion(
        self,
        x0_clip: torch.Tensor,      # (dim,) 原始图像的CLIP特征
        h_clips: torch.Tensor,       # (s, dim) decoded noise predictions的CLIP特征
        ud_clips: torch.Tensor,      # (s, dim) spherical noise的CLIP特征
    ) -> float:
        """
        按照论文Section 4.3计算单个图像的criterion
        
        公式: C(x0) = 1/s * Σ⟨-h/||h||, a*u_d - b*h + c*√d*x0⟩
        
        在CLIP空间中，使用cosine similarity作为内积
        """
        s = h_clips.shape[0]
        
        # 归一化 h: -h / ||h||
        h_norms = torch.norm(h_clips, p=2, dim=1, keepdim=True) + 1e-8
        h_normalized = -h_clips / h_norms  # (s, dim)
        
        # 扩展 x0_clip 到 (s, dim)
        x0_expanded = x0_clip.unsqueeze(0).expand(s, -1)  # (s, dim)
        
        # 计算 a*u_d - b*h + c*√d*x0
        # 注意：论文中说在CLIP空间用cosine similarity，所以这里的组合也应该在CLIP空间
        combined = (self.a * ud_clips - 
                    self.b * h_clips + 
                    self.c * self.sqrt_d_clip * x0_expanded)  # (s, dim)
        
        # 计算内积 (cosine similarity)
        # cos_sim 范围是 [-1, 1]
        inner_products = self.cos(h_normalized, combined)  # (s,)
        
        # 平均
        C = inner_products.mean().item()
        
        # 归一化到 [0, 1] 范围
        # 论文: "Dynamic range is adjusted to approximate [0, 1] by scaling with a + b + c and adding 1"
        C_normalized = (C + 1) / (self.a + self.b + self.c + 1)
        
        # 另一种归一化方式（如果上面的效果不好可以试试）：
        # C_normalized = (C + (self.a + self.b + self.c)) / (self.a + self.b + self.c)
        
        return C_normalized
    
    def evaluate_batch(
        self,
        images: torch.Tensor,
        prompts: List[str],
        images_raw: Optional[List[torch.Tensor]] = None
    ) -> List[Dict]:
        """
        Evaluate images using CLIP-based criterion.
        """
        if images_raw is None:
            raise ValueError("CLIPCriterion requires images_raw parameter")
        
        num_images = images.shape[0]
        num_noise = self.config.num_noise
        
        # Get model components
        unet = self.model_manager.unet
        vae = self.model_manager.vae
        scheduler = self.model_manager.scheduler
        tokenizer = self.model_manager.tokenizer
        text_encoder = self.model_manager.text_encoder
        
        # --------- text embeddings ---------
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
        
        # --------- encode images to latent space ---------
        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
        
        latents = latents.repeat_interleave(num_noise, dim=0).half()
        
        # --------- spherical noise (u_d) ---------
        # 论文 Eq.10: x̃ = √(1-α)*x0 + √α*u_d
        # u_d ~ Unif(S^{d-1}(√d)) - uniform on sphere with radius √d
        gauss_noise = torch.randn_like(latents, device=self.device).half()
        
        # 归一化到单位球面，然后乘以√d
        sqrt_d_latent = torch.prod(torch.tensor(latents.shape[1:])).float().sqrt()
        spherical_noise = self.normalize_batch(gauss_noise, self.config.epsilon_reg).half()
        spherical_noise = spherical_noise * sqrt_d_latent  # u_d with radius √d
        
        # 保存原始的 spherical_noise (u_d) 用于后续CLIP映射
        # 注意：这是在latent space的u_d
        u_d_latent = spherical_noise.clone()
        
        # --------- timestep ---------
        timestep = self.config.time_frac * scheduler.config.num_train_timesteps
        timestep = torch.full((latents.shape[0],), timestep, device=self.device, dtype=torch.long)
        
        # --------- add noise: x̃ = √(1-α)*x0 + √α*u_d ---------
        noisy_latents = scheduler.add_noise(
            original_samples=latents, 
            noise=spherical_noise, 
            timesteps=timestep
        ).half()
        
        # --------- predict noise: h(x̃) ---------
        with torch.no_grad():
            noise_pred = unet(noisy_latents, timestep, encoder_hidden_states=text_emb)[0]
        
        # 论文中说要除以 scaling_factor (这是 h(x̃))
        noise_pred_scaled = noise_pred / vae.config.scaling_factor
        
        # Clean up
        del noisy_latents, timestep, gauss_noise, text_emb, input_ids
        
        # --------- decode to image space ---------
        # 需要decode: noise_pred (h), spherical_noise (u_d)
        sub_batch_size = 16
        
        def batch_decode(latent_batch):
            """Helper to decode in sub-batches"""
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
        
        # Decode noise prediction (h)
        decoded_noise = batch_decode(noise_pred_scaled)
        
        # Decode spherical noise (u_d)
        # 注意：u_d 也需要除以 scaling_factor 才能正确decode
        u_d_for_decode = u_d_latent / vae.config.scaling_factor
        decoded_u_d = batch_decode(u_d_for_decode)
        
        del noise_pred, noise_pred_scaled, spherical_noise, u_d_latent, u_d_for_decode, latents
        
        # --------- postprocess to numpy ---------
        siz = self.config.image_size
        decoded_noise_np = self.postprocess_decoded(decoded_noise, siz, do_resize=True)
        decoded_u_d_np = self.postprocess_decoded(decoded_u_d, siz, do_resize=True)
        
        del decoded_noise, decoded_u_d
        
        # Split into per-image chunks
        decoded_noise_chunks = numpy_chunk(decoded_noise_np, num_images)  # List of (num_noise, H, W, C)
        decoded_u_d_chunks = numpy_chunk(decoded_u_d_np, num_images)
        
        # --------- compute CLIP criterion for each image ---------
        results = []
        for i, (h_chunk, ud_chunk, cur_image_raw) in enumerate(
            zip(decoded_noise_chunks, decoded_u_d_chunks, images_raw)
        ):
            # 1. CLIP features for original image x0
            img_for_clip = cur_image_raw.float().to(self.device)
            x0_clip = self.compute_clip_features(img_for_clip)  # (1, dim)
            x0_clip = x0_clip.squeeze(0)  # (dim,)
            
            # 2. CLIP features for decoded noise prediction h(x̃)
            h_clip = self.compute_clip_features(h_chunk)  # (num_noise, dim)
            
            # 3. CLIP features for decoded spherical noise u_d
            ud_clip = self.compute_clip_features(ud_chunk)  # (num_noise, dim)
            
            # 4. 计算 criterion
            criterion = self.compute_single_criterion(x0_clip, h_clip, ud_clip)
            
            result = {"criterion": float(criterion)}
            
            if self.config.return_terms:
                # 额外返回一些中间值用于调试
                h_norms = torch.norm(h_clip, p=2, dim=1)
                
                # bias: cos(x0, h) - 原始图像和noise prediction的相似度
                bias_vec = self.cos(x0_clip.unsqueeze(0).expand(h_clip.shape[0], -1), h_clip)
                
                # kappa: cos(h, u_d) - noise prediction和spherical noise的相似度  
                kappa_vec = self.cos(h_clip, ud_clip)
                
                result.update({
                    "bias": float(bias_vec.mean()),
                    "kappa": float(kappa_vec.mean()),
                    "D": float(h_norms.mean()),
                    "x0_norm": float(torch.norm(x0_clip)),
                })
            
            results.append(result)
        
        return results


@register_criterion("clip_v2")
class CLIPCriterionV2(BaseCriterion):
    """
    Alternative implementation that more closely follows the paper's description.
    
    这个版本尝试更严格地按照论文的数学公式实现：
    - 使用论文中的 κ(x0) - D(x0) 形式
    """
    
    def __init__(self, config, model_manager):
        super().__init__(config, model_manager)
        
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.d_clip = 768
        self.sqrt_d_clip = self.d_clip ** 0.5
        self.image_processor = ImageProcessor(config)
    
    def compute_clip_features(self, images: np.ndarray) -> torch.Tensor:
        clip_model = self.model_manager.clip
        processor = self.model_manager.clip_processor
        inputs = processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = clip_model.get_image_features(**inputs)
        return features.detach().cpu()
    
    def postprocess_decoded(self, decoded: torch.Tensor, size: int, do_resize: bool = True) -> np.ndarray:
        return self.image_processor.postprocess(decoded, do_resize)
    
    def compute_criterion_v2(
        self,
        x0_clip: torch.Tensor,      # (dim,)
        h_clips: torch.Tensor,       # (s, dim)
        ud_clips: torch.Tensor,      # (s, dim)
    ) -> Dict[str, float]:
        """
        按照论文 Corollary 2 的形式计算:
        
        κ(x0) ≈ curvature (高 = 生成图像)
        D(x0) ≈ gradient magnitude (低 = 生成图像)
        
        生成图像应该有: 高κ, 低D
        所以 criterion = κ - D 应该对生成图像更高
        """
        s = h_clips.shape[0]
        
        # 归一化 h
        h_norms = torch.norm(h_clips, p=2, dim=1, keepdim=True) + 1e-8
        h_normalized = h_clips / h_norms  # (s, dim)
        
        # 归一化 u_d
        ud_norms = torch.norm(ud_clips, p=2, dim=1, keepdim=True) + 1e-8
        ud_normalized = ud_clips / ud_norms  # (s, dim)
        
        x0_expanded = x0_clip.unsqueeze(0).expand(s, -1)
        
        # κ 项: -⟨h/||h||, u_d⟩ (Eq. 14)
        # 在CLIP空间用cosine similarity
        kappa_term = -self.cos(h_normalized, ud_normalized)  # (s,)
        kappa = kappa_term.mean().item()
        
        # D 项: ⟨h/||h||, h⟩ = ||h|| (Eq. 26)
        D = h_norms.squeeze().mean().item()
        
        # bias 项: ⟨h/||h||, x0⟩ (Corollary 3, Eq. 18)
        # 这个项对于unbiased predictor应该接近0
        bias_term = self.cos(h_normalized, x0_expanded)  # (s,)
        bias = bias_term.mean().item()
        
        # 组合 criterion
        # 论文: C ≈ c1/√α * κ - c2*D + c3*⟨b0, x0⟩
        # 简化版本 (a=b=c=1):
        criterion_raw = kappa - D / self.sqrt_d_clip + bias
        
        # 归一化
        criterion = (criterion_raw + 2) / 4  # 调整到大约 [0, 1]
        
        return {
            "criterion": criterion,
            "kappa": kappa,
            "D": D,
            "bias": bias,
            "criterion_raw": criterion_raw,
        }
    
    def evaluate_batch(
        self,
        images: torch.Tensor,
        prompts: List[str],
        images_raw: Optional[List[torch.Tensor]] = None
    ) -> List[Dict]:
        """与 CLIPCriterion 相同的接口"""
        if images_raw is None:
            raise ValueError("CLIPCriterionV2 requires images_raw parameter")
        
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
        
        text_tokens = tokenizer(
            expanded_prompts, 
            padding="max_length", 
            max_length=77, 
            truncation=True, 
            return_tensors="pt"
        )
        input_ids = text_tokens.input_ids.to(self.device)
        text_emb = text_encoder(input_ids).last_hidden_state
        
        # Encode images
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
        
        # Add noise
        noisy_latents = scheduler.add_noise(
            original_samples=latents, 
            noise=spherical_noise, 
            timesteps=timestep
        ).half()
        
        # Predict noise
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
        
        # Postprocess
        siz = self.config.image_size
        decoded_noise_np = self.postprocess_decoded(decoded_noise, siz, do_resize=True)
        decoded_u_d_np = self.postprocess_decoded(decoded_u_d, siz, do_resize=True)
        
        del decoded_noise, decoded_u_d
        
        decoded_noise_chunks = numpy_chunk(decoded_noise_np, num_images)
        decoded_u_d_chunks = numpy_chunk(decoded_u_d_np, num_images)
        
        # Compute criterion for each image
        results = []
        for i, (h_chunk, ud_chunk, cur_image_raw) in enumerate(
            zip(decoded_noise_chunks, decoded_u_d_chunks, images_raw)
        ):
            img_for_clip = cur_image_raw.float().to(self.device)
            x0_clip = self.compute_clip_features(img_for_clip).squeeze(0)
            h_clip = self.compute_clip_features(h_chunk)
            ud_clip = self.compute_clip_features(ud_chunk)
            
            criterion_dict = self.compute_criterion_v2(x0_clip, h_clip, ud_clip)
            
            result = {"criterion": criterion_dict["criterion"]}
            
            if self.config.return_terms:
                result.update({
                    "kappa": criterion_dict["kappa"],
                    "D": criterion_dict["D"],
                    "bias": criterion_dict["bias"],
                    "criterion_raw": criterion_dict["criterion_raw"],
                })
            
            results.append(result)
        
        return results