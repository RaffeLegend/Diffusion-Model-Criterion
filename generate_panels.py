"""
Generate Visualization Panels for Single Image
==============================================
Input: Single image
Output: Individual visualization panels with overlay

Usage:
    python generate_panels.py --image /path/to/image.png --output_dir output --label real

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


class DiffusionAnalyzer:
    """Analyze images using pretrained diffusion model."""
    
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model_id = model_id
        self._load_model()
        
    def _load_model(self):
        """Load VAE and UNet from pretrained diffusion model."""
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
        
        print("Model loaded successfully!")
    
    def load_image(self, path, size=512):
        """Load and preprocess a single image."""
        img = Image.open(path).convert('RGB')
        img = img.resize((size, size), Image.LANCZOS)
        img_np = np.array(img).astype(np.float32) / 255.0
        
        # Convert to tensor and normalize to [-1, 1]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        img_tensor = (img_tensor - 0.5) * 2
        
        return img_tensor.to(self.device), img_np
    
    @torch.no_grad()
    def get_noise_prediction(self, img_tensor, timestep_ratio=0.3, num_samples=4):
        """Get decoded noise prediction h^dec for an image."""
        # Encode to latent space
        latent = self.vae.encode(img_tensor).latent_dist.sample()
        latent = latent * self.vae.config.scaling_factor
        
        # Get timestep
        num_timesteps = self.scheduler.config.num_train_timesteps
        t = int(timestep_ratio * num_timesteps)
        timestep = torch.tensor([t], device=self.device)
        
        h_dec_list = []
        
        for _ in range(num_samples):
            # Add noise
            noise = torch.randn_like(latent)
            noisy_latent = self.scheduler.add_noise(latent, noise, timestep)
            
            # UNet needs text conditioning - use empty/unconditional
            encoder_hidden_states = torch.zeros(
                1, 77, self.unet.config.cross_attention_dim, 
                device=self.device
            )
            
            # Predict noise
            h = self.unet(noisy_latent, timestep, encoder_hidden_states).sample
            
            # Decode to image space
            h_dec = self.vae.decode(h / self.vae.config.scaling_factor).sample
            h_dec_list.append(h_dec)
        
        # Average over samples
        h_dec = torch.stack(h_dec_list).mean(dim=0)
        
        # Convert to numpy [H, W, 3]
        h_dec_np = h_dec.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        return h_dec_np


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


def compute_frequency_spectrum(image):
    """Compute 2D FFT magnitude spectrum and radial average."""
    if len(image.shape) == 3:
        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    else:
        gray = image
    
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)
    spectrum = np.log1p(magnitude)
    
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)
    
    max_r = min(cx, cy)
    radial_profile = np.zeros(max_r)
    for i in range(max_r):
        mask = (r == i)
        if mask.sum() > 0:
            radial_profile[i] = magnitude[mask].mean()
    
    return spectrum, radial_profile


def compute_high_freq_energy(image, K_ratio=0.25):
    """Compute high-frequency energy ratio."""
    if len(image.shape) == 3:
        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    else:
        gray = image
    
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    power = np.abs(f_shift) ** 2
    
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    K = K_ratio * min(cx, cy)
    
    high_freq_mask = r > K
    total_energy = power.sum()
    high_freq_energy = power[high_freq_mask].sum()
    
    return high_freq_energy / total_energy if total_energy > 0 else 0


def normalize_for_vis(img):
    """Normalize image to [0, 1] for visualization."""
    img_min, img_max = img.min(), img.max()
    if img_max - img_min > 1e-8:
        return (img - img_min) / (img_max - img_min)
    return img - img_min


def create_overlay_blend(image, heatmap, alpha=0.5, threshold_percentile=50):
    """
    Create overlay: blend heatmap color on original image.
    High response areas show through in hot colors.
    """
    heatmap_norm = normalize_for_vis(heatmap)
    
    # Apply threshold
    threshold = np.percentile(heatmap_norm, threshold_percentile)
    heatmap_masked = np.where(heatmap_norm > threshold, heatmap_norm, 0)
    
    # Smooth
    heatmap_smooth = gaussian_filter(heatmap_masked, sigma=2)
    heatmap_smooth = normalize_for_vis(heatmap_smooth)
    
    # Apply hot colormap
    cmap = plt.cm.get_cmap('hot')
    heatmap_colored = cmap(heatmap_smooth)[:, :, :3]
    
    # Create alpha mask
    alpha_mask = heatmap_smooth[:, :, np.newaxis]
    alpha_mask = np.clip(alpha_mask * 2, 0, 1)
    
    # Blend
    overlay = image * (1 - alpha_mask * alpha) + heatmap_colored * alpha_mask * alpha
    overlay = np.clip(overlay, 0, 1)
    
    return overlay


def create_overlay_highlight(image, heatmap, threshold_percentile=75):
    """
    Create overlay: highlight high-response regions with red color and boundary.
    """
    from scipy.ndimage import binary_dilation, binary_erosion
    
    heatmap_norm = normalize_for_vis(heatmap)
    heatmap_smooth = gaussian_filter(heatmap_norm, sigma=3)
    
    threshold = np.percentile(heatmap_smooth, threshold_percentile)
    mask = heatmap_smooth > threshold
    
    # Get boundary
    dilated = binary_dilation(mask, iterations=3)
    eroded = binary_erosion(mask, iterations=1)
    boundary = dilated & ~eroded
    
    # Create overlay
    overlay = image.copy()
    # Semi-transparent red fill
    overlay[mask] = overlay[mask] * 0.6 + np.array([1, 0, 0]) * 0.4
    # Solid red boundary
    overlay[boundary] = np.array([1, 0, 0])
    
    return overlay, mask


def create_overlay_contour(image, heatmap, output_path, levels=5):
    """
    Create overlay: draw contour lines on original image.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    
    heatmap_smooth = gaussian_filter(normalize_for_vis(heatmap), sigma=3)
    contour_levels = np.linspace(0.3, 0.9, levels)
    ax.contour(heatmap_smooth, levels=contour_levels, colors='red', linewidths=1.5, alpha=0.8)
    
    ax.axis('off')
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def save_image(data, path, cmap=None, vmin=None, vmax=None):
    """Save a single image without any borders or labels."""
    fig, ax = plt.subplots(figsize=(5, 5))
    if cmap:
        ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        ax.imshow(data)
    ax.axis('off')
    plt.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate visualization panels for single image')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--label', type=str, default='', help='Label prefix for output files')
    parser.add_argument('--model_id', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--timestep', type=float, default=0.3, help='Timestep ratio')
    parser.add_argument('--num_samples', type=int, default=4, help='Number of noise samples')
    parser.add_argument('--threshold', type=float, default=70, help='Percentile threshold for highlighting')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    prefix = f"{args.label}_" if args.label else ""
    
    # Initialize analyzer
    analyzer = DiffusionAnalyzer(model_id=args.model_id)
    
    # Load image
    print(f"Processing: {args.image}")
    img_tensor, img_np = analyzer.load_image(args.image)
    
    # Get noise prediction
    print("Computing noise prediction...")
    h_dec = analyzer.get_noise_prediction(img_tensor, args.timestep, args.num_samples)
    
    # Compute Laplacian
    print("Computing Laplacian...")
    laplacian = compute_laplacian(h_dec)
    laplacian_mag = np.sqrt((laplacian ** 2).sum(axis=-1))
    laplacian_norm = laplacian_mag.mean()
    
    # Compute frequency spectrum
    print("Computing frequency spectrum...")
    spectrum, radial = compute_frequency_spectrum(h_dec)
    hf_energy = compute_high_freq_energy(h_dec)
    
    # ===== Save basic outputs =====
    print("Saving outputs...")
    
    # 1. Original input image
    save_image(img_np, os.path.join(args.output_dir, f'{prefix}input.png'))
    
    # 2. h^dec
    h_dec_vis = normalize_for_vis(h_dec)
    save_image(h_dec_vis, os.path.join(args.output_dir, f'{prefix}hdec.png'))
    
    # 3. Laplacian magnitude
    save_image(laplacian_mag, os.path.join(args.output_dir, f'{prefix}laplacian.png'), cmap='hot')
    
    # 4. Frequency spectrum
    save_image(spectrum, os.path.join(args.output_dir, f'{prefix}spectrum.png'), cmap='viridis')
    
    # ===== Save overlay visualizations =====
    print("Creating overlay visualizations...")
    
    # 5. Heatmap overlay (blend)
    overlay_blend = create_overlay_blend(img_np, laplacian_mag, alpha=0.6, threshold_percentile=50)
    save_image(overlay_blend, os.path.join(args.output_dir, f'{prefix}overlay_blend.png'))
    
    # 6. Highlighted regions
    overlay_highlight, mask = create_overlay_highlight(img_np, laplacian_mag, 
                                                        threshold_percentile=args.threshold)
    save_image(overlay_highlight, os.path.join(args.output_dir, f'{prefix}overlay_highlight.png'))
    
    # 7. Contour overlay
    create_overlay_contour(img_np, laplacian_mag, 
                          os.path.join(args.output_dir, f'{prefix}overlay_contour.png'))
    
    # 8. Save radial spectrum
    np.save(os.path.join(args.output_dir, f'{prefix}radial.npy'), radial)
    
    # 9. Save metrics
    metrics_path = os.path.join(args.output_dir, f'{prefix}metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"laplacian_norm: {laplacian_norm}\n")
        f.write(f"hf_energy: {hf_energy}\n")
        f.write(f"high_response_ratio: {mask.mean()}\n")
    
    # Print summary
    print(f"\n===== Results =====")
    print(f"||Δh^dec||: {laplacian_norm:.6f}")
    print(f"HF Energy: {hf_energy:.6f}")
    print(f"High Response Area: {mask.mean()*100:.1f}%")
    print(f"\nSaved to {args.output_dir}/:")
    print(f"  {prefix}input.png              - original image")
    print(f"  {prefix}hdec.png               - noise prediction h^dec")
    print(f"  {prefix}laplacian.png          - ||Δh^dec|| heatmap")
    print(f"  {prefix}spectrum.png           - frequency spectrum")
    print(f"  {prefix}overlay_blend.png      - heatmap blended on image")
    print(f"  {prefix}overlay_highlight.png  - high-response regions highlighted")
    print(f"  {prefix}overlay_contour.png    - contour lines on image")
    print(f"  {prefix}radial.npy             - radial spectrum data")
    print(f"  {prefix}metrics.txt            - numerical metrics")


if __name__ == "__main__":
    main()