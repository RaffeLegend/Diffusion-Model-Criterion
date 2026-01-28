"""
Three-Column Comparison: Frequency Analysis vs Manifold vs Our Method (SLC)
============================================================================
Compares three approaches on 500+ real and 500+ fake images:
1. Naive frequency analysis of original image
2. Manifold method (without Laplacian enhancement)
3. Our method (with Laplacian subtractive enhancement)

Usage:
    python compare_three_methods.py --real_dir /path/to/real --gen_dir /path/to/gen --output_dir results

Adjustable parameters for tuning results:
    --K: frequency threshold for high-frequency energy (default: 0.25)
    --timestep: diffusion timestep ratio (default: 0.3)
    --lambda_enh: subtractive enhancement weight (default: 0.1)
    --num_samples: noise samples per image (default: 8)

Requirements:
    pip install torch diffusers transformers matplotlib numpy scipy scikit-learn tqdm
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from glob import glob
import json


class ThreeMethodComparison:
    """Compare frequency analysis, Manifold, and SLC methods."""
    
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self._load_models(model_id)
        
        # Laplacian kernel
        self.lap_kernel = torch.tensor([
            [0,  1,  0],
            [1, -4,  1],
            [0,  1,  0]
        ], dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1, 1, 1).to(self.device)
        
    def _load_models(self, model_id):
        from diffusers import AutoencoderKL, UNet2DConditionModel
        from diffusers.schedulers import DDPMScheduler
        from transformers import CLIPModel
        
        print("Loading models...")
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(self.device).eval()
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(self.device).eval()
        self.scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device).eval()
        print("Models loaded!")
    
    def load_image(self, path, size=512):
        try:
            img = Image.open(path).convert('RGB')
            img = img.resize((size, size), Image.LANCZOS)
            img_np = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
            img_tensor = (img_tensor - 0.5) * 2  # [-1, 1]
            return img_tensor.to(self.device), img_np
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None, None
    
    # ==================== Method 1: Frequency Analysis ====================
    
    def compute_high_freq_energy(self, img_np, K=0.25, normalize=True):
        """
        Compute high-frequency energy of original image.
        
        Args:
            img_np: numpy image [H, W, 3] in [0, 1]
            K: frequency threshold (0-0.5), higher K = more selective
            normalize: if True, return ratio E_high / E_total
        """
        # Convert to grayscale
        gray = 0.299 * img_np[:,:,0] + 0.587 * img_np[:,:,1] + 0.114 * img_np[:,:,2]
        
        # FFT
        f = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f)
        magnitude = np.abs(f_shift) ** 2
        
        # Create frequency mask
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        r_norm = r / max(h, w)  # Normalize to [0, 0.5]
        
        # High frequency mask
        hf_mask = r_norm > K
        
        # Compute energies
        total_energy = magnitude.sum()
        hf_energy = magnitude[hf_mask].sum()
        
        if normalize and total_energy > 0:
            return hf_energy / total_energy
        return hf_energy
    
    # ==================== Method 2 & 3: Manifold and Ours ====================
    
    @torch.no_grad()
    def apply_laplacian(self, imgs):
        if imgs.ndim == 3:
            imgs = imgs.unsqueeze(0)
        lap = F.conv2d(imgs.float(), self.lap_kernel.to(imgs.device), padding=1, groups=3)
        lap = torch.abs(lap)
        B = lap.shape[0]
        lap_flat = lap.view(B, -1)
        lap_min = lap_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        lap_max = lap_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        lap = (lap - lap_min) / (lap_max - lap_min + 1e-8)
        return lap
    
    @torch.no_grad()
    def get_clip_features(self, img_tensor):
        img_resized = F.interpolate(img_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(self.device)
        img_norm = (img_resized - mean) / std
        feats = self.clip.get_image_features(pixel_values=img_norm)
        return feats
    
    @torch.no_grad()
    def compute_manifold_and_slc(self, img_tensor, timestep_ratio=0.3, num_samples=8, lambda_enh=0.1):
        """
        Compute both Manifold criterion (lambda=0) and our SLC criterion (lambda>0).
        
        Returns:
            manifold_criterion: Manifold method score (no Laplacian enhancement)
            slc_criterion: Our method score (with Laplacian enhancement)
            laplacian_mean: Mean Laplacian magnitude (for reference)
        """
        # Hyperparameters (same as original)
        a, b, c = 1.0, 1.0, 1.0
        d_clip = 768
        sqrt_d_clip = d_clip ** 0.5
        
        # Encode
        z = self.vae.encode(img_tensor).latent_dist.sample()
        z = z * self.vae.config.scaling_factor
        
        # Timestep
        num_timesteps = self.scheduler.config.num_train_timesteps
        t = int(timestep_ratio * num_timesteps)
        timestep = torch.tensor([t], device=self.device)
        
        # x0 CLIP features
        x0_01 = (img_tensor + 1) / 2
        x0_feat = self.get_clip_features(x0_01)
        x0_feat_norm = F.normalize(x0_feat, p=2, dim=1)
        
        cos = torch.nn.CosineSimilarity(dim=1)
        
        manifold_scores = []
        slc_scores = []
        laplacian_means = []
        
        for _ in range(num_samples):
            # Sample noise
            noise = torch.randn_like(z)
            sqrt_d = torch.prod(torch.tensor(z.shape[1:])).float().sqrt()
            u = F.normalize(noise.view(1, -1), p=2, dim=1).view(z.shape) * sqrt_d
            
            # Add noise
            z_noisy = self.scheduler.add_noise(z, u, timestep)
            
            # UNet prediction
            encoder_hidden_states = torch.zeros(1, 77, self.unet.config.cross_attention_dim, device=self.device)
            h = self.unet(z_noisy, timestep, encoder_hidden_states).sample
            
            # Decode h
            h_dec = self.vae.decode(h / self.vae.config.scaling_factor).sample
            h_dec_01 = ((h_dec + 1) / 2).clamp(0, 1)
            
            # Laplacian
            h_lap = self.apply_laplacian(h_dec_01)
            laplacian_means.append(torch.sqrt((h_lap ** 2).sum(dim=1)).mean().item())
            
            # Decode u
            u_dec = self.vae.decode(u / self.vae.config.scaling_factor).sample
            u_dec_01 = ((u_dec + 1) / 2).clamp(0, 1)
            
            # CLIP features
            h_feat = self.get_clip_features(h_dec_01)
            h_lap_feat = self.get_clip_features(h_lap)
            u_feat = self.get_clip_features(u_dec_01)
            
            h_feat_n = F.normalize(h_feat, p=2, dim=1)
            h_lap_feat_n = F.normalize(h_lap_feat, p=2, dim=1)
            u_feat_n = F.normalize(u_feat, p=2, dim=1)
            
            # ===== Manifold (lambda = 0) =====
            phi_manifold = h_feat_n  # No enhancement
            h_norm_m = torch.norm(phi_manifold, p=2, dim=1, keepdim=True) + 1e-8
            h_dir_m = -phi_manifold / h_norm_m
            vec_m = a * u_feat_n - b * phi_manifold + c * sqrt_d_clip * x0_feat_norm
            C_m = cos(h_dir_m, vec_m).item()
            manifold_scores.append(C_m)
            
            # ===== Our SLC (lambda > 0) =====
            phi_slc = h_feat_n - lambda_enh * h_lap_feat_n  # Subtractive enhancement
            h_norm_s = torch.norm(phi_slc, p=2, dim=1, keepdim=True) + 1e-8
            h_dir_s = -phi_slc / h_norm_s
            vec_s = a * u_feat_n - b * phi_slc + c * sqrt_d_clip * x0_feat_norm
            C_s = cos(h_dir_s, vec_s).item()
            slc_scores.append(C_s)
        
        # Normalize scores (same as original)
        manifold_criterion = (np.mean(manifold_scores) + 1) / (a + b + c + 1)
        slc_criterion = (np.mean(slc_scores) + 1) / (a + b + c + 1)
        laplacian_mean = np.mean(laplacian_means)
        
        return manifold_criterion, slc_criterion, laplacian_mean
    
    # ==================== Main Analysis ====================
    
    def analyze_image(self, img_path, K=0.25, timestep_ratio=0.3, num_samples=8, lambda_enh=0.1):
        """Analyze single image with all three methods."""
        
        img_tensor, img_np = self.load_image(img_path)
        if img_tensor is None:
            return None
        
        try:
            # Method 1: Frequency analysis
            hf_energy = self.compute_high_freq_energy(img_np, K=K)
            
            # Method 2 & 3: Manifold and SLC
            manifold_score, slc_score, lap_mean = self.compute_manifold_and_slc(
                img_tensor, timestep_ratio, num_samples, lambda_enh
            )
            
            return {
                'freq_hf_energy': hf_energy,
                'manifold_criterion': manifold_score,
                'slc_criterion': slc_score,
                'laplacian_mean': lap_mean,
            }
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None
    
    def analyze_dataset(self, image_paths, label, K=0.25, timestep_ratio=0.3, 
                        num_samples=8, lambda_enh=0.1):
        """Analyze all images."""
        
        all_results = []
        
        for path in tqdm(image_paths, desc=f"Analyzing {label}"):
            result = self.analyze_image(path, K, timestep_ratio, num_samples, lambda_enh)
            if result is not None:
                result['path'] = path
                result['label'] = label
                all_results.append(result)
            
            # Clear cache periodically
            if len(all_results) % 50 == 0:
                torch.cuda.empty_cache()
        
        return all_results


def compute_auc(real_vals, gen_vals):
    """Compute AUC (try both directions, return best)."""
    labels = np.array([0] * len(real_vals) + [1] * len(gen_vals))
    scores = np.concatenate([real_vals, gen_vals])
    
    auc1 = roc_auc_score(labels, scores)
    auc2 = roc_auc_score(labels, -scores)
    
    if auc1 >= auc2:
        return auc1, "higher=gen"
    else:
        return auc2, "lower=gen"


def create_three_column_figure(real_results, gen_results, output_dir, 
                                bins=50, alpha=0.7, xlim_freq=None, 
                                xlim_manifold=None, xlim_slc=None,
                                figsize=(15, 5)):
    """
    Create the three-column comparison figure.
    
    Args:
        bins: number of histogram bins
        alpha: histogram transparency
        xlim_*: x-axis limits for each column (None for auto)
    """
    
    # Extract values
    real_freq = np.array([r['freq_hf_energy'] for r in real_results])
    gen_freq = np.array([r['freq_hf_energy'] for r in gen_results])
    
    real_manifold = np.array([r['manifold_criterion'] for r in real_results])
    gen_manifold = np.array([r['manifold_criterion'] for r in gen_results])
    
    real_slc = np.array([r['slc_criterion'] for r in real_results])
    gen_slc = np.array([r['slc_criterion'] for r in gen_results])
    
    # Compute AUCs
    auc_freq, dir_freq = compute_auc(real_freq, gen_freq)
    auc_manifold, dir_manifold = compute_auc(real_manifold, gen_manifold)
    auc_slc, dir_slc = compute_auc(real_slc, gen_slc)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # ===== Column 1: Frequency Analysis =====
    ax1 = axes[0]
    ax1.hist(real_freq, bins=bins, alpha=alpha, color='#2E86AB', label='Real', density=True)
    ax1.hist(gen_freq, bins=bins, alpha=alpha, color='#E94F37', label='Generated', density=True)
    ax1.axvline(real_freq.mean(), color='#2E86AB', linestyle='--', linewidth=2)
    ax1.axvline(gen_freq.mean(), color='#E94F37', linestyle='--', linewidth=2)
    ax1.set_xlabel('High-Frequency Energy Ratio', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title(f'(a) Frequency Analysis\nAUC = {auc_freq:.3f}', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    if xlim_freq:
        ax1.set_xlim(xlim_freq)
    
    # ===== Column 2: Manifold =====
    ax2 = axes[1]
    ax2.hist(real_manifold, bins=bins, alpha=alpha, color='#2E86AB', label='Real', density=True)
    ax2.hist(gen_manifold, bins=bins, alpha=alpha, color='#E94F37', label='Generated', density=True)
    ax2.axvline(real_manifold.mean(), color='#2E86AB', linestyle='--', linewidth=2)
    ax2.axvline(gen_manifold.mean(), color='#E94F37', linestyle='--', linewidth=2)
    ax2.set_xlabel('Criterion Score', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title(f'(b) Manifold (λ=0)\nAUC = {auc_manifold:.3f}', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    if xlim_manifold:
        ax2.set_xlim(xlim_manifold)
    
    # ===== Column 3: Our SLC =====
    ax3 = axes[2]
    ax3.hist(real_slc, bins=bins, alpha=alpha, color='#2E86AB', label='Real', density=True)
    ax3.hist(gen_slc, bins=bins, alpha=alpha, color='#E94F37', label='Generated', density=True)
    ax3.axvline(real_slc.mean(), color='#2E86AB', linestyle='--', linewidth=2)
    ax3.axvline(gen_slc.mean(), color='#E94F37', linestyle='--', linewidth=2)
    ax3.set_xlabel('Criterion Score', fontsize=11)
    ax3.set_ylabel('Density', fontsize=11)
    ax3.set_title(f'(c) Ours (SLC, λ>0)\nAUC = {auc_slc:.3f}', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)
    if xlim_slc:
        ax3.set_xlim(xlim_slc)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'three_column_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'three_column_comparison.pdf'), bbox_inches='tight')
    plt.close()
    
    # Print summary
    print(f"\n{'='*60}")
    print("THREE-METHOD COMPARISON")
    print(f"{'='*60}")
    print(f"{'Method':<25} {'Real μ±σ':>15} {'Gen μ±σ':>15} {'AUC':>8}")
    print("-"*60)
    print(f"{'Frequency Analysis':<25} {real_freq.mean():.4f}±{real_freq.std():.4f}   {gen_freq.mean():.4f}±{gen_freq.std():.4f}   {auc_freq:.3f}")
    print(f"{'Manifold (λ=0)':<25} {real_manifold.mean():.4f}±{real_manifold.std():.4f}   {gen_manifold.mean():.4f}±{gen_manifold.std():.4f}   {auc_manifold:.3f}")
    print(f"{'Ours SLC (λ>0)':<25} {real_slc.mean():.4f}±{real_slc.std():.4f}   {gen_slc.mean():.4f}±{gen_slc.std():.4f}   {auc_slc:.3f}")
    print(f"{'='*60}")
    
    return {
        'freq': {'auc': auc_freq, 'real_mean': real_freq.mean(), 'gen_mean': gen_freq.mean()},
        'manifold': {'auc': auc_manifold, 'real_mean': real_manifold.mean(), 'gen_mean': gen_manifold.mean()},
        'slc': {'auc': auc_slc, 'real_mean': real_slc.mean(), 'gen_mean': gen_slc.mean()},
    }


def create_detailed_figure(real_results, gen_results, output_dir, bins=50, alpha=0.7):
    """Create additional detailed figures: ROC curves and bar chart."""
    
    # Extract values
    real_freq = np.array([r['freq_hf_energy'] for r in real_results])
    gen_freq = np.array([r['freq_hf_energy'] for r in gen_results])
    real_manifold = np.array([r['manifold_criterion'] for r in real_results])
    gen_manifold = np.array([r['manifold_criterion'] for r in gen_results])
    real_slc = np.array([r['slc_criterion'] for r in real_results])
    gen_slc = np.array([r['slc_criterion'] for r in gen_results])
    
    # Compute AUCs and ROC curves
    def get_roc(real_vals, gen_vals):
        labels = np.array([0] * len(real_vals) + [1] * len(gen_vals))
        scores = np.concatenate([real_vals, gen_vals])
        auc1 = roc_auc_score(labels, scores)
        auc2 = roc_auc_score(labels, -scores)
        if auc1 >= auc2:
            fpr, tpr, _ = roc_curve(labels, scores)
            return fpr, tpr, auc1
        else:
            fpr, tpr, _ = roc_curve(labels, -scores)
            return fpr, tpr, auc2
    
    fpr_freq, tpr_freq, auc_freq = get_roc(real_freq, gen_freq)
    fpr_manifold, tpr_manifold, auc_manifold = get_roc(real_manifold, gen_manifold)
    fpr_slc, tpr_slc, auc_slc = get_roc(real_slc, gen_slc)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC curves
    ax1 = axes[0]
    ax1.plot(fpr_freq, tpr_freq, 'gray', linewidth=2, linestyle='--', label=f'Frequency (AUC={auc_freq:.3f})')
    ax1.plot(fpr_manifold, tpr_manifold, '#FFA500', linewidth=2, label=f'Manifold (AUC={auc_manifold:.3f})')
    ax1.plot(fpr_slc, tpr_slc, '#E94F37', linewidth=2.5, label=f'Ours SLC (AUC={auc_slc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k:', linewidth=1)
    ax1.set_xlabel('False Positive Rate', fontsize=11)
    ax1.set_ylabel('True Positive Rate', fontsize=11)
    ax1.set_title('ROC Curves', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # AUC bar chart
    ax2 = axes[1]
    methods = ['Frequency\nAnalysis', 'Manifold\n(λ=0)', 'Ours SLC\n(λ>0)']
    aucs = [auc_freq, auc_manifold, auc_slc]
    colors = ['gray', '#FFA500', '#E94F37']
    
    bars = ax2.bar(methods, aucs, color=colors, alpha=0.8)
    ax2.axhline(0.5, color='black', linestyle='--', linewidth=1, label='Random')
    ax2.set_ylabel('AUC', fontsize=11)
    ax2.set_title('AUC Comparison', fontsize=12, fontweight='bold')
    ax2.set_ylim([0.4, 1.0])
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, auc in zip(bars, aucs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{auc:.3f}', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_and_auc.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'roc_and_auc.pdf'), bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare three methods for AI image detection')
    
    # Data paths
    parser.add_argument('--real_dir', type=str, required=True, help='Directory containing real images')
    parser.add_argument('--gen_dir', type=str, required=True, help='Directory containing generated images')
    parser.add_argument('--output_dir', type=str, default='three_method_comparison')
    
    # Adjustable parameters
    parser.add_argument('--K', type=float, default=0.25, help='Frequency threshold (0-0.5), higher = more selective')
    parser.add_argument('--timestep', type=float, default=0.3, help='Diffusion timestep ratio')
    parser.add_argument('--lambda_enh', type=float, default=0.1, help='Subtractive enhancement weight')
    parser.add_argument('--num_samples', type=int, default=8, help='Noise samples per image')
    
    # Dataset options
    parser.add_argument('--max_images', type=int, default=500, help='Max images per class')
    parser.add_argument('--extensions', type=str, default='jpg,JPEG,png,webp')
    
    # Visualization options
    parser.add_argument('--bins', type=int, default=50, help='Histogram bins')
    parser.add_argument('--alpha', type=float, default=0.7, help='Histogram transparency')
    parser.add_argument('--xlim_freq', type=float, nargs=2, default=None, help='X-axis limits for frequency plot')
    parser.add_argument('--xlim_manifold', type=float, nargs=2, default=None, help='X-axis limits for manifold plot')
    parser.add_argument('--xlim_slc', type=float, nargs=2, default=None, help='X-axis limits for SLC plot')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save config
    config = vars(args)
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Collect image paths
    extensions = args.extensions.split(',')
    real_paths = []
    gen_paths = []
    
    for ext in extensions:
        real_paths.extend(glob(os.path.join(args.real_dir, f'*.{ext}')))
        real_paths.extend(glob(os.path.join(args.real_dir, f'**/*.{ext}'), recursive=True))
        gen_paths.extend(glob(os.path.join(args.gen_dir, f'*.{ext}')))
        gen_paths.extend(glob(os.path.join(args.gen_dir, f'**/*.{ext}'), recursive=True))
    
    real_paths = list(set(real_paths))[:args.max_images]
    gen_paths = list(set(gen_paths))[:args.max_images]
    
    print(f"Found {len(real_paths)} real images and {len(gen_paths)} generated images")
    
    if len(real_paths) == 0 or len(gen_paths) == 0:
        print("Error: No images found!")
        return
    
    # Initialize analyzer
    analyzer = ThreeMethodComparison()
    
    # Analyze datasets
    print(f"\nParameters: K={args.K}, timestep={args.timestep}, lambda={args.lambda_enh}, samples={args.num_samples}")
    
    print("\nAnalyzing real images...")
    real_results = analyzer.analyze_dataset(
        real_paths, 'real', 
        K=args.K, timestep_ratio=args.timestep, 
        num_samples=args.num_samples, lambda_enh=args.lambda_enh
    )
    
    print("\nAnalyzing generated images...")
    gen_results = analyzer.analyze_dataset(
        gen_paths, 'gen',
        K=args.K, timestep_ratio=args.timestep,
        num_samples=args.num_samples, lambda_enh=args.lambda_enh
    )
    
    print(f"\nSuccessfully analyzed {len(real_results)} real and {len(gen_results)} generated images")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    summary = create_three_column_figure(
        real_results, gen_results, args.output_dir,
        bins=args.bins, alpha=args.alpha,
        xlim_freq=args.xlim_freq, xlim_manifold=args.xlim_manifold, xlim_slc=args.xlim_slc
    )
    
    create_detailed_figure(real_results, gen_results, args.output_dir, bins=args.bins, alpha=args.alpha)
    
    # Save raw results
    np.save(os.path.join(args.output_dir, 'real_results.npy'), real_results)
    np.save(os.path.join(args.output_dir, 'gen_results.npy'), gen_results)
    
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Results saved to {args.output_dir}/")
    print(f"   - three_column_comparison.png: Main comparison figure")
    print(f"   - roc_and_auc.png: ROC curves and AUC bar chart")
    print(f"   - config.json: Parameters used")
    print(f"   - summary.json: AUC summary")


if __name__ == "__main__":
    main()