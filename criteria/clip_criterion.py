"""
CLIP-based Manifold Criterion

This criterion evaluates images by comparing CLIP features of:
1. Original image
2. Decoded noise prediction
3. Decoded spherical noise

The criterion measures how well the diffusion model's noise prediction
aligns with the image manifold in CLIP space.
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
    
    Computes similarity metrics between original image, noise prediction,
    and spherical noise in CLIP feature space.
    """
    
    def __init__(self, config, model_manager):
        super().__init__(config, model_manager)
        
        # Cosine similarity
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        
        # CLIP feature dimension
        self.sqrt_d_clip = np.sqrt(768)  # CLIP ViT-L/14 dimension
        
        self.image_processor = ImageProcessor(config)
    
    def compute_clip_features(self, images: np.ndarray) -> torch.Tensor:
        """
        Compute CLIP image features.
        
        Args:
            images: NumPy array of shape (B, H, W, C) with uint8 values
            
        Returns:
            CLIP features tensor of shape (B, 768)
        """
        clip = self.model_manager.clip
        processor = self.model_manager.clip_processor
        
        # Process images for CLIP
        inputs = processor(images=list(images), return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self.device)
        
        with torch.no_grad():
            features = clip.get_image_features(pixel_values)
        
        return features.cpu()
    
    def postprocess_decoded(
        self, 
        decoded: torch.Tensor, 
        size: int, 
        do_resize: bool = True
    ) -> np.ndarray:
        """
        Postprocess decoded latents to uint8 numpy array.
        
        Args:
            decoded: Tensor of shape (B, C, H, W) in [-1, 1]
            size: Target size for resize
            do_resize: Whether to resize
            
        Returns:
            NumPy array of shape (B, H, W, C) with uint8 values
        """
        return self.image_processor.postprocess(decoded, do_resize)
    
    def evaluate_batch(
        self,
        images: torch.Tensor,
        prompts: List[str],
        images_raw: Optional[List[torch.Tensor]] = None
    ) -> List[Dict]:
        """
        Evaluate images using CLIP-based criterion.
        
        The algorithm:
        1. Encode images to latent space
        2. Add spherical noise at timestep t
        3. Predict noise with UNet
        4. Decode both noise prediction and spherical noise
        5. Compute CLIP features for original, predicted, and noise
        6. Compute similarity metrics
        
        Args:
            images: Preprocessed images (B, C, H, W) in [-1, 1]
            prompts: Text prompts for each image
            images_raw: Raw images (H, W, C) in [0, 255] - REQUIRED for this criterion
            
        Returns:
            List of result dicts with 'criterion' key
        """
        if images_raw is None:
            raise ValueError("CLIPCriterion requires images_raw parameter")
        
        num_images = images.shape[0]
        num_noise = self.config.num_noise
        
        # Get model components
        unet = self.model_manager.unet
        scheduler = self.model_manager.scheduler
        
        # Compute text embeddings
        text_emb = self.compute_text_embeddings(prompts, num_noise)
        
        # Encode images to latent space
        latents = self.encode_images_to_latents(images)
        
        # Repeat for noise samples
        latents = latents.repeat_interleave(num_noise, dim=0).half()
        
        # Sample spherical noise
        gauss_noise = torch.randn_like(latents, device=self.device, dtype=torch.float16)
        spherical_noise = self.normalize_batch(gauss_noise, self.config.epsilon_reg).half()
        
        # Scale noise by sqrt(dimension)
        sqrt_d = torch.prod(torch.tensor(latents.shape[1:], device=self.device)).float().sqrt()
        spherical_noise = spherical_noise * sqrt_d
        
        # Get timestep
        timestep = self.config.time_frac * scheduler.config.num_train_timesteps
        timestep = torch.full((latents.shape[0],), timestep, device=self.device, dtype=torch.long)
        
        # Add noise to latents
        noisy_latents = scheduler.add_noise(
            original_samples=latents,
            noise=spherical_noise,
            timesteps=timestep
        ).half()
        
        # Predict noise with UNet
        with torch.no_grad():
            noise_pred = unet(noisy_latents, timestep, encoder_hidden_states=text_emb)[0]
        
        # Clean up intermediate tensors
        del noisy_latents, timestep, gauss_noise, text_emb
        
        # Decode noise predictions and spherical noise to image space
        decoded_noise = self.decode_latents(noise_pred)
        decoded_spherical_noise = self.decode_latents(spherical_noise)
        
        # Postprocess to numpy
        siz = self.config.image_size
        decoded_noise_np = self.postprocess_decoded(decoded_noise, siz, do_resize=True)
        decoded_spherical_noise_np = self.postprocess_decoded(decoded_spherical_noise, siz, do_resize=True)
        
        # Split into per-image chunks
        decoded_noise_chunks = numpy_chunk(decoded_noise_np, num_images)
        decoded_spherical_chunks = numpy_chunk(decoded_spherical_noise_np, num_images)
        
        # Compute CLIP-based criterion for each image
        results = []
        for i, (noise_chunk, spherical_chunk, img_raw) in enumerate(
            zip(decoded_noise_chunks, decoded_spherical_chunks, images_raw)
        ):
            # Get CLIP features for original image
            img_np = img_raw.float().cpu().numpy()
            img_clip = self.compute_clip_features(img_np[np.newaxis, ...])
            
            # Get CLIP features for decoded noise prediction
            img_d_clip = self.compute_clip_features(noise_chunk)
            
            # Get CLIP features for decoded spherical noise
            img_s_clip = self.compute_clip_features(spherical_chunk)
            
            # Compute similarity metrics
            bias_vec = self.cos(img_clip, img_d_clip).numpy()
            kappa_vec = self.cos(img_d_clip, img_s_clip).numpy()
            D_vec = torch.norm(
                img_d_clip.view(img_d_clip.size(0), -1),
                p=2,
                dim=1
            ).cpu().numpy()
            
            # Aggregate over noise samples
            bias_mean = bias_vec.mean()
            kappa_mean = kappa_vec.mean()
            D_mean = D_vec.mean()
            
            # Compute criterion using coefficients from paper
            criterion = 1 + (self.sqrt_d_clip * bias_mean - D_mean + kappa_mean) / (self.sqrt_d_clip + 2)
            
            result = {"criterion": float(criterion)}
            
            if self.config.return_terms:
                result.update({
                    "bias": float(bias_mean),
                    "kappa": float(kappa_mean),
                    "D": float(D_mean),
                    "bias_std": float(bias_vec.std()),
                    "kappa_std": float(kappa_vec.std()),
                    "D_std": float(D_vec.std()),
                })
            
            results.append(result)
        
        return results
