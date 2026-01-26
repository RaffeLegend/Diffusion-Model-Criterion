"""
Analyze All SLC Components
==========================
Computes every intermediate quantity in the SLC method to find
which signal has the largest Real vs Generated gap.

Usage:
    python analyze_slc_components.py --real real.jpg --gen gen.png --output_dir output

Requirements:
    pip install torch diffusers transformers matplotlib numpy
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F


class SLCAnalyzer:
    """Analyze all components of SLC method."""
    
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self._load_models(model_id)
        
        # Hyperparameters
        self.lam = 0.1
        self.a = 1.0
        self.b = 1.0
        self.c = 1.0
        self.d_clip = 768
        self.sqrt_d_clip = self.d_clip ** 0.5
        
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
        img = Image.open(path).convert('RGB')
        img = img.resize((size, size), Image.LANCZOS)
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        img_tensor = (img_tensor - 0.5) * 2
        return img_tensor.to(self.device), img_np
    
    @torch.no_grad()
    def apply_laplacian(self, imgs):
        if imgs.ndim == 3:
            imgs = imgs.unsqueeze(0)
        lap = F.conv2d(imgs.float(), self.lap_kernel.to(imgs.device), padding=1, groups=3)
        # Normalize
        lap = torch.abs(lap)
        B = lap.shape[0]
        lap_flat = lap.view(B, -1)
        lap_min = lap_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        lap_max = lap_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        lap = (lap - lap_min) / (lap_max - lap_min + 1e-8)
        return lap
    
    @torch.no_grad()
    def get_clip_features(self, img_tensor):
        """img_tensor: [B, 3, H, W] in [0, 1]"""
        img_resized = F.interpolate(img_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(self.device)
        img_norm = (img_resized - mean) / std
        feats = self.clip.get_image_features(pixel_values=img_norm)
        return feats  # Don't normalize yet, keep raw
    
    @torch.no_grad()
    def analyze_image(self, img_tensor, num_samples=16, timestep_ratio=0.3):
        """
        Compute ALL intermediate quantities for analysis.
        """
        results = {}
        
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
        
        results['x0_feat_norm'] = torch.norm(x0_feat, p=2).item()
        
        # Collect over multiple noise samples
        all_metrics = {
            'h_feat_norm': [],
            'h_lap_feat_norm': [],
            'h_enh_feat_norm': [],
            'u_feat_norm': [],
            'cos_h_hlap': [],           # cos(Φ(h), Φ(Δh))
            'cos_h_u': [],              # cos(Φ(h), Φ(u))
            'cos_h_x0': [],             # cos(Φ(h), Φ(x0))
            'cos_henh_x0': [],          # cos(Φ_enh, Φ(x0))
            'cos_henh_u': [],           # cos(Φ_enh, Φ(u))
            'cos_hlap_x0': [],          # cos(Φ(Δh), Φ(x0))
            'cos_hlap_u': [],           # cos(Φ(Δh), Φ(u))
            'criterion_raw': [],
            'criterion_norm': [],
            'term_a_u': [],             # a * Φ(u) contribution
            'term_b_henh': [],          # b * Φ_enh contribution  
            'term_c_x0': [],            # c * √d * Φ(x0) contribution
            'laplacian_pixel_mean': [], # ||Δh^dec|| in pixel space
        }
        
        cos = torch.nn.CosineSimilarity(dim=1)
        
        for i in range(num_samples):
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
            
            # Laplacian of h_dec
            h_lap = self.apply_laplacian(h_dec_01)
            
            # Pixel-space Laplacian magnitude
            h_lap_mag = torch.sqrt((h_lap ** 2).sum(dim=1))
            all_metrics['laplacian_pixel_mean'].append(h_lap_mag.mean().item())
            
            # Decode u
            u_dec = self.vae.decode(u / self.vae.config.scaling_factor).sample
            u_dec_01 = ((u_dec + 1) / 2).clamp(0, 1)
            
            # CLIP features (raw, not normalized)
            h_feat = self.get_clip_features(h_dec_01)
            h_lap_feat = self.get_clip_features(h_lap)
            u_feat = self.get_clip_features(u_dec_01)
            
            # Norms
            all_metrics['h_feat_norm'].append(torch.norm(h_feat, p=2).item())
            all_metrics['h_lap_feat_norm'].append(torch.norm(h_lap_feat, p=2).item())
            all_metrics['u_feat_norm'].append(torch.norm(u_feat, p=2).item())
            
            # Normalize for cosine
            h_feat_n = F.normalize(h_feat, p=2, dim=1)
            h_lap_feat_n = F.normalize(h_lap_feat, p=2, dim=1)
            u_feat_n = F.normalize(u_feat, p=2, dim=1)
            
            # Subtractive enhancement
            h_enh = h_feat_n - self.lam * h_lap_feat_n
            h_enh_n = F.normalize(h_enh, p=2, dim=1)
            
            all_metrics['h_enh_feat_norm'].append(torch.norm(h_enh, p=2).item())
            
            # Cosine similarities
            all_metrics['cos_h_hlap'].append(cos(h_feat_n, h_lap_feat_n).item())
            all_metrics['cos_h_u'].append(cos(h_feat_n, u_feat_n).item())
            all_metrics['cos_h_x0'].append(cos(h_feat_n, x0_feat_norm).item())
            all_metrics['cos_henh_x0'].append(cos(h_enh_n, x0_feat_norm).item())
            all_metrics['cos_henh_u'].append(cos(h_enh_n, u_feat_n).item())
            all_metrics['cos_hlap_x0'].append(cos(h_lap_feat_n, x0_feat_norm).item())
            all_metrics['cos_hlap_u'].append(cos(h_lap_feat_n, u_feat_n).item())
            
            # Criterion calculation (from your code)
            h_norm = torch.norm(h_enh, p=2, dim=1, keepdim=True) + 1e-8
            h_dir = -h_enh / h_norm
            
            vec = self.a * u_feat_n - self.b * h_enh + self.c * self.sqrt_d_clip * x0_feat_norm
            
            C = cos(h_dir, vec).item()
            C_norm = (C + 1) / (self.a + self.b + self.c + 1)
            
            all_metrics['criterion_raw'].append(C)
            all_metrics['criterion_norm'].append(C_norm)
            
            # Individual term contributions (dot products with h_dir)
            all_metrics['term_a_u'].append((h_dir * self.a * u_feat_n).sum().item())
            all_metrics['term_b_henh'].append((h_dir * (-self.b * h_enh)).sum().item())
            all_metrics['term_c_x0'].append((h_dir * self.c * self.sqrt_d_clip * x0_feat_norm).sum().item())
        
        # Aggregate results (mean and std)
        for key, values in all_metrics.items():
            results[f'{key}_mean'] = np.mean(values)
            results[f'{key}_std'] = np.std(values)
            results[f'{key}_all'] = values
        
        return results


def compute_gaps(real_results, gen_results):
    """Compute gaps between real and generated for all metrics."""
    gaps = {}
    
    metrics = [
        'h_feat_norm',
        'h_lap_feat_norm', 
        'h_enh_feat_norm',
        'u_feat_norm',
        'cos_h_hlap',
        'cos_h_u',
        'cos_h_x0',
        'cos_henh_x0',
        'cos_henh_u',
        'cos_hlap_x0',
        'cos_hlap_u',
        'criterion_raw',
        'criterion_norm',
        'term_a_u',
        'term_b_henh',
        'term_c_x0',
        'laplacian_pixel_mean',
    ]
    
    for m in metrics:
        real_val = real_results[f'{m}_mean']
        gen_val = gen_results[f'{m}_mean']
        gap = gen_val - real_val
        ratio = gen_val / (real_val + 1e-8) if real_val != 0 else float('inf')
        
        gaps[m] = {
            'real': real_val,
            'gen': gen_val,
            'gap': gap,
            'ratio': ratio,
            'abs_gap': abs(gap)
        }
    
    return gaps


def visualize_gaps(gaps, output_dir):
    """Visualize the gaps for all metrics."""
    
    # Sort by absolute gap
    sorted_metrics = sorted(gaps.keys(), key=lambda x: gaps[x]['abs_gap'], reverse=True)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ===== Plot 1: Top gaps bar chart =====
    ax1 = axes[0, 0]
    
    top_n = 10
    top_metrics = sorted_metrics[:top_n]
    
    x = np.arange(len(top_metrics))
    width = 0.35
    
    real_vals = [gaps[m]['real'] for m in top_metrics]
    gen_vals = [gaps[m]['gen'] for m in top_metrics]
    
    bars1 = ax1.bar(x - width/2, real_vals, width, label='Real', color='#2E86AB', alpha=0.8)
    bars2 = ax1.bar(x + width/2, gen_vals, width, label='Generated', color='#E94F37', alpha=0.8)
    
    ax1.set_ylabel('Value', fontsize=11)
    ax1.set_title(f'Top {top_n} Metrics by Gap (Real vs Generated)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(top_metrics, rotation=45, ha='right', fontsize=9)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # ===== Plot 2: Gap magnitudes =====
    ax2 = axes[0, 1]
    
    gap_vals = [gaps[m]['gap'] for m in top_metrics]
    colors = ['#E94F37' if g > 0 else '#2E86AB' for g in gap_vals]
    
    ax2.barh(top_metrics, gap_vals, color=colors, alpha=0.8)
    ax2.axvline(0, color='black', linewidth=1)
    ax2.set_xlabel('Gap (Gen - Real)', fontsize=11)
    ax2.set_title('Gap Direction and Magnitude', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # ===== Plot 3: Cosine similarities comparison =====
    ax3 = axes[1, 0]
    
    cos_metrics = [m for m in sorted_metrics if m.startswith('cos_')]
    
    x = np.arange(len(cos_metrics))
    real_cos = [gaps[m]['real'] for m in cos_metrics]
    gen_cos = [gaps[m]['gen'] for m in cos_metrics]
    
    ax3.bar(x - width/2, real_cos, width, label='Real', color='#2E86AB', alpha=0.8)
    ax3.bar(x + width/2, gen_cos, width, label='Generated', color='#E94F37', alpha=0.8)
    
    ax3.set_ylabel('Cosine Similarity', fontsize=11)
    ax3.set_title('CLIP Feature Space: Cosine Similarities', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([m.replace('cos_', '') for m in cos_metrics], rotation=45, ha='right', fontsize=9)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim([-1, 1])
    
    # ===== Plot 4: Criterion terms breakdown =====
    ax4 = axes[1, 1]
    
    term_metrics = ['term_a_u', 'term_b_henh', 'term_c_x0', 'criterion_raw']
    
    x = np.arange(len(term_metrics))
    real_terms = [gaps[m]['real'] for m in term_metrics]
    gen_terms = [gaps[m]['gen'] for m in term_metrics]
    
    ax4.bar(x - width/2, real_terms, width, label='Real', color='#2E86AB', alpha=0.8)
    ax4.bar(x + width/2, gen_terms, width, label='Generated', color='#E94F37', alpha=0.8)
    
    ax4.set_ylabel('Value', fontsize=11)
    ax4.set_title('Criterion Components Breakdown', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['a·Φ(u)', '-b·Φ_enh', 'c·√d·Φ(x₀)', 'Criterion'], fontsize=10)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'slc_components_analysis.png'), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'slc_components_analysis.pdf'), bbox_inches='tight')
    plt.close()


def print_report(gaps):
    """Print a detailed report of all gaps."""
    
    print("\n" + "="*80)
    print("SLC COMPONENTS ANALYSIS REPORT")
    print("="*80)
    
    # Sort by absolute gap
    sorted_metrics = sorted(gaps.keys(), key=lambda x: gaps[x]['abs_gap'], reverse=True)
    
    print(f"\n{'Metric':<25} {'Real':>12} {'Gen':>12} {'Gap':>12} {'Ratio':>10}")
    print("-"*80)
    
    for m in sorted_metrics:
        g = gaps[m]
        ratio_str = f"{g['ratio']:.3f}x" if abs(g['ratio']) < 100 else "inf"
        print(f"{m:<25} {g['real']:>12.4f} {g['gen']:>12.4f} {g['gap']:>+12.4f} {ratio_str:>10}")
    
    print("-"*80)
    
    # Highlight top discriminative metrics
    print("\n🔥 TOP 5 MOST DISCRIMINATIVE METRICS:")
    for i, m in enumerate(sorted_metrics[:5]):
        g = gaps[m]
        direction = "Gen > Real" if g['gap'] > 0 else "Real > Gen"
        print(f"  {i+1}. {m}: gap = {g['gap']:+.4f} ({direction})")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real', type=str, required=True)
    parser.add_argument('--gen', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='slc_analysis')
    parser.add_argument('--num_samples', type=int, default=16)
    parser.add_argument('--timestep', type=float, default=0.3)
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize
    analyzer = SLCAnalyzer()
    
    # Analyze real image
    print(f"\nAnalyzing real image: {args.real}")
    real_tensor, real_img = analyzer.load_image(args.real)
    real_results = analyzer.analyze_image(real_tensor, args.num_samples, args.timestep)
    
    # Analyze generated image
    print(f"Analyzing generated image: {args.gen}")
    gen_tensor, gen_img = analyzer.load_image(args.gen)
    gen_results = analyzer.analyze_image(gen_tensor, args.num_samples, args.timestep)
    
    # Compute gaps
    gaps = compute_gaps(real_results, gen_results)
    
    # Print report
    print_report(gaps)
    
    # Visualize
    print("\nCreating visualizations...")
    visualize_gaps(gaps, args.output_dir)
    
    # Save raw results
    np.save(os.path.join(args.output_dir, 'real_results.npy'), real_results)
    np.save(os.path.join(args.output_dir, 'gen_results.npy'), gen_results)
    np.save(os.path.join(args.output_dir, 'gaps.npy'), gaps)
    
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()