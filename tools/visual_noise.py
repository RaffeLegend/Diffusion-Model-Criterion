"""
Visualize Full Detection Pipeline
==================================
Shows every step of our method with decoded visualizations.

Pipeline:
1. x_0 - Original image
2. Decode(z_0) - VAE reconstruction (should ≈ x_0)
3. Decode(z_t) - Noisy latent decoded to image space
4. h^dec - Noise prediction decoded to image space
5. Δh^dec - Laplacian of noise prediction

Usage:
    python visualize_pipeline.py --image /path/to/image.jpg --output_dir output --label real

Requirements:
    pip install torch diffusers matplotlib numpy scipy pillow
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from scipy.ndimage import convolve, gaussian_filter


class PipelineVisualizer:
    """Visualize the full detection pipeline."""
    
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model_id = model_id
        self._load_model()
        
    def _load_model(self):
        """Load VAE, UNet, and scheduler."""
        from diffusers import AutoencoderKL, UNet2DConditionModel
        from diffusers.schedulers import DDPMScheduler
        
        print(f"Loading model: {self.model_id}")
        
        self.vae = AutoencoderKL.from_pretrained(
            self.model_id, subfolder="vae"
        ).to(self.device).eval()
        
        self.unet = UNet2DConditionModel.from_pretrained(
            self.model_id, subfolder="unet"
        ).to(self.device).eval()
        
        self.scheduler = DDPMScheduler.from_pretrained(
            self.model_id, subfolder="scheduler"
        )
        
        print("Model loaded!")
    
    def load_image(self, path, size=512):
        """Load and preprocess image."""
        img = Image.open(path).convert('RGB')
        img = img.resize((size, size), Image.LANCZOS)
        img_np = np.array(img).astype(np.float32) / 255.0
        
        # To tensor [-1, 1]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        img_tensor = (img_tensor - 0.5) * 2
        
        return img_tensor.to(self.device), img_np
    
    @torch.no_grad()
    def decode_latent(self, latent):
        """Decode latent to image space."""
        img = self.vae.decode(latent / self.vae.config.scaling_factor).sample
        img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = (img + 1) / 2  # [-1,1] -> [0,1]
        img = np.clip(img, 0, 1)
        return img
    
    @torch.no_grad()
    def run_pipeline(self, img_tensor, timestep_ratio=0.3, num_samples=4):
        """
        Run the full pipeline and return all intermediate results.
        
        Returns dict with:
            - z_0: original latent
            - z_0_decoded: VAE reconstruction
            - z_t: noisy latent
            - z_t_decoded: noisy image
            - h: noise prediction (latent)
            - h_dec: noise prediction (decoded)
            - laplacian: Laplacian of h_dec
            - alpha_bar: noise schedule value
        """
        results = {}
        
        # Step 1: Encode to latent space
        latent_dist = self.vae.encode(img_tensor).latent_dist
        z_0 = latent_dist.sample()
        z_0_scaled = z_0 * self.vae.config.scaling_factor
        results['z_0'] = z_0_scaled
        
        # Step 2: Decode z_0 (should reconstruct original)
        results['z_0_decoded'] = self.decode_latent(z_0_scaled)
        
        # Step 3: Add noise to get z_t
        num_timesteps = self.scheduler.config.num_train_timesteps
        t = int(timestep_ratio * num_timesteps)
        timestep = torch.tensor([t], device=self.device)
        
        # Store alpha_bar for reference
        alpha_bar = self.scheduler.alphas_cumprod[t].item()
        results['alpha_bar'] = alpha_bar
        results['timestep'] = t
        
        # Average over multiple noise samples
        h_list = []
        z_t_list = []
        
        for _ in range(num_samples):
            noise = torch.randn_like(z_0_scaled)
            z_t = self.scheduler.add_noise(z_0_scaled, noise, timestep)
            z_t_list.append(z_t)
            
            # Step 4: UNet predicts noise
            encoder_hidden_states = torch.zeros(
                1, 77, self.unet.config.cross_attention_dim,
                device=self.device
            )
            h = self.unet(z_t, timestep, encoder_hidden_states).sample
            h_list.append(h)
        
        # Average
        z_t_avg = torch.stack(z_t_list).mean(dim=0)
        h_avg = torch.stack(h_list).mean(dim=0)
        
        results['z_t'] = z_t_avg
        results['z_t_decoded'] = self.decode_latent(z_t_avg)
        
        results['h'] = h_avg
        results['h_dec'] = self.decode_latent(h_avg)
        
        # Step 5: Compute Laplacian
        results['laplacian'] = compute_laplacian(results['h_dec'])
        results['laplacian_mag'] = np.sqrt((results['laplacian'] ** 2).sum(axis=-1))
        
        return results
    
    def visualize_pipeline(self, image_path, output_dir, label='', timestep_ratio=0.3):
        """Visualize the full pipeline."""
        os.makedirs(output_dir, exist_ok=True)
        prefix = f"{label}_" if label else ""
        
        # Load image
        print(f"Processing: {image_path}")
        img_tensor, img_np = self.load_image(image_path)
        
        # Run pipeline
        print("Running pipeline...")
        results = self.run_pipeline(img_tensor, timestep_ratio=timestep_ratio)
        
        # Compute metrics
        laplacian_norm = results['laplacian_mag'].mean()
        
        # ===== Save individual images =====
        print("Saving visualizations...")
        
        # 1. Original image x_0
        save_image(img_np, os.path.join(output_dir, f'{prefix}1_x0_original.png'))
        
        # 2. VAE reconstruction Decode(z_0)
        save_image(results['z_0_decoded'], 
                   os.path.join(output_dir, f'{prefix}2_z0_decoded.png'))
        
        # 3. Noisy image Decode(z_t)
        save_image(results['z_t_decoded'],
                   os.path.join(output_dir, f'{prefix}3_zt_decoded.png'))
        
        # 4. Noise prediction h^dec
        h_dec_vis = normalize_for_vis(results['h_dec'])
        save_image(h_dec_vis, os.path.join(output_dir, f'{prefix}4_h_dec.png'))
        
        # 5. Laplacian magnitude
        save_image(results['laplacian_mag'],
                   os.path.join(output_dir, f'{prefix}5_laplacian.png'),
                   cmap='hot')
        
        # 6. Overlay on original
        overlay = create_overlay(img_np, results['laplacian_mag'])
        save_image(overlay, os.path.join(output_dir, f'{prefix}6_overlay.png'))
        
        # ===== Create pipeline figure =====
        self.create_pipeline_figure(
            img_np, results, output_dir, prefix, timestep_ratio
        )
        
        # ===== Save metrics =====
        metrics_path = os.path.join(output_dir, f'{prefix}metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write(f"timestep: {results['timestep']}\n")
            f.write(f"alpha_bar: {results['alpha_bar']:.4f}\n")
            f.write(f"signal_ratio: {np.sqrt(results['alpha_bar']):.4f}\n")
            f.write(f"noise_ratio: {np.sqrt(1-results['alpha_bar']):.4f}\n")
            f.write(f"laplacian_norm: {laplacian_norm:.6f}\n")
        
        # Print summary
        print(f"\n===== Results =====")
        print(f"Timestep: {results['timestep']} / 1000")
        print(f"α̅_t = {results['alpha_bar']:.4f}")
        print(f"Signal ratio: {np.sqrt(results['alpha_bar']):.2%}")
        print(f"Noise ratio: {np.sqrt(1-results['alpha_bar']):.2%}")
        print(f"||Δh^dec||: {laplacian_norm:.6f}")
        print(f"\nSaved to {output_dir}/")
        
    def create_pipeline_figure(self, img_np, results, output_dir, prefix, timestep_ratio):
        """Create a figure showing the full pipeline."""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        t = results['timestep']
        alpha_bar = results['alpha_bar']
        
        # Row 1: Image space
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title(r'(1) Input $x_0$', fontsize=12)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(results['z_0_decoded'])
        axes[0, 1].set_title(r'(2) VAE Recon $\mathcal{D}(\mathcal{E}(x_0))$', fontsize=12)
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(results['z_t_decoded'])
        axes[0, 2].set_title(f'(3) Noisy $\\mathcal{{D}}(z_t)$\nt={t}, $\\bar{{\\alpha}}_t$={alpha_bar:.2f}', fontsize=12)
        axes[0, 2].axis('off')
        
        # Row 2: Detection pipeline
        h_dec_vis = normalize_for_vis(results['h_dec'])
        axes[1, 0].imshow(h_dec_vis)
        axes[1, 0].set_title(r'(4) Noise Pred $\mathbf{h}^{dec}=\mathcal{D}(\epsilon_\theta(z_t,t))$', fontsize=12)
        axes[1, 0].axis('off')
        
        im = axes[1, 1].imshow(results['laplacian_mag'], cmap='hot')
        axes[1, 1].set_title(r'(5) Laplacian $\|\Delta \mathbf{h}^{dec}\|$', fontsize=12)
        axes[1, 1].axis('off')
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        overlay = create_overlay(img_np, results['laplacian_mag'])
        axes[1, 2].imshow(overlay)
        axes[1, 2].set_title('(6) High-Response Regions', fontsize=12)
        axes[1, 2].axis('off')
        
        # Add arrows to show flow
        fig.suptitle('Score Laplacian Criterion (SLC) Pipeline', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{prefix}pipeline.png'), 
                    dpi=200, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f'{prefix}pipeline.pdf'), 
                    bbox_inches='tight')
        plt.close()


def compute_laplacian(image):
    """Compute Laplacian using 3x3 kernel."""
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=np.float32)
    
    if len(image.shape) == 3:
        laplacian = np.zeros_like(image)
        for c in range(image.shape[2]):
            laplacian[:, :, c] = convolve(image[:, :, c], kernel, mode='reflect')
    else:
        laplacian = convolve(image, kernel, mode='reflect')
        
    return laplacian


def normalize_for_vis(img):
    """Normalize image to [0, 1] for visualization."""
    img_min, img_max = img.min(), img.max()
    if img_max - img_min > 1e-8:
        return (img - img_min) / (img_max - img_min)
    return img - img_min


def create_overlay(image, heatmap, alpha=0.5, threshold_percentile=50):
    """Create overlay of heatmap on original image."""
    heatmap_norm = normalize_for_vis(heatmap)
    
    threshold = np.percentile(heatmap_norm, threshold_percentile)
    heatmap_masked = np.where(heatmap_norm > threshold, heatmap_norm, 0)
    
    heatmap_smooth = gaussian_filter(heatmap_masked, sigma=2)
    heatmap_smooth = normalize_for_vis(heatmap_smooth)
    
    cmap = plt.cm.get_cmap('hot')
    heatmap_colored = cmap(heatmap_smooth)[:, :, :3]
    
    alpha_mask = heatmap_smooth[:, :, np.newaxis]
    alpha_mask = np.clip(alpha_mask * 2, 0, 1)
    
    overlay = image * (1 - alpha_mask * alpha) + heatmap_colored * alpha_mask * alpha
    overlay = np.clip(overlay, 0, 1)
    
    return overlay


def save_image(data, path, cmap=None, vmin=None, vmax=None):
    """Save a single image."""
    fig, ax = plt.subplots(figsize=(5, 5))
    if cmap:
        ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        ax.imshow(data)
    ax.axis('off')
    plt.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize full detection pipeline')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='pipeline_vis', help='Output directory')
    parser.add_argument('--label', type=str, default='', help='Label prefix (e.g., real, gen)')
    parser.add_argument('--model_id', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--timestep', type=float, default=0.3, help='Timestep ratio (0-1)')
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = PipelineVisualizer(model_id=args.model_id)
    
    # Visualize
    visualizer.visualize_pipeline(
        args.image, 
        args.output_dir, 
        label=args.label,
        timestep_ratio=args.timestep
    )


if __name__ == "__main__":
    main()