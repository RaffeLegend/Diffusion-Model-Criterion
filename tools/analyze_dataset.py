"""
SLC Large-Scale Statistical Analysis
=====================================
Analyzes 1000+ real and 1000+ generated images to find statistically
significant differences in SLC-related metrics.

Usage:
    python analyze_dataset.py --real_dir /path/to/real --gen_dir /path/to/gen --output_dir results

Outputs:
    - Distribution plots for all metrics
    - Statistical tests (t-test, KS test, AUC)
    - Summary report identifying most discriminative metrics

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
import json
from glob import glob


class SLCDatasetAnalyzer:
    """Analyze SLC metrics on large dataset."""
    
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
        try:
            img = Image.open(path).convert('RGB')
            img = img.resize((size, size), Image.LANCZOS)
            img_np = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
            img_tensor = (img_tensor - 0.5) * 2
            return img_tensor.to(self.device)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None
    
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
    def analyze_single_image(self, img_tensor, num_samples=8, timestep_ratio=0.3):
        """Compute all SLC metrics for a single image."""
        
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
        
        # Collect metrics over noise samples
        metrics_list = []
        
        for _ in range(num_samples):
            noise = torch.randn_like(z)
            sqrt_d = torch.prod(torch.tensor(z.shape[1:])).float().sqrt()
            u = F.normalize(noise.view(1, -1), p=2, dim=1).view(z.shape) * sqrt_d
            
            z_noisy = self.scheduler.add_noise(z, u, timestep)
            
            encoder_hidden_states = torch.zeros(1, 77, self.unet.config.cross_attention_dim, device=self.device)
            h = self.unet(z_noisy, timestep, encoder_hidden_states).sample
            
            h_dec = self.vae.decode(h / self.vae.config.scaling_factor).sample
            h_dec_01 = ((h_dec + 1) / 2).clamp(0, 1)
            
            h_lap = self.apply_laplacian(h_dec_01)
            h_lap_mag = torch.sqrt((h_lap ** 2).sum(dim=1)).mean().item()
            
            u_dec = self.vae.decode(u / self.vae.config.scaling_factor).sample
            u_dec_01 = ((u_dec + 1) / 2).clamp(0, 1)
            
            h_feat = self.get_clip_features(h_dec_01)
            h_lap_feat = self.get_clip_features(h_lap)
            u_feat = self.get_clip_features(u_dec_01)
            
            h_feat_n = F.normalize(h_feat, p=2, dim=1)
            h_lap_feat_n = F.normalize(h_lap_feat, p=2, dim=1)
            u_feat_n = F.normalize(u_feat, p=2, dim=1)
            
            h_enh = h_feat_n - self.lam * h_lap_feat_n
            h_enh_n = F.normalize(h_enh, p=2, dim=1)
            
            # Criterion
            h_norm = torch.norm(h_enh, p=2, dim=1, keepdim=True) + 1e-8
            h_dir = -h_enh / h_norm
            vec = self.a * u_feat_n - self.b * h_enh + self.c * self.sqrt_d_clip * x0_feat_norm
            C = cos(h_dir, vec).item()
            C_norm = (C + 1) / (self.a + self.b + self.c + 1)
            
            metrics_list.append({
                'cos_h_x0': cos(h_feat_n, x0_feat_norm).item(),
                'cos_henh_x0': cos(h_enh_n, x0_feat_norm).item(),
                'cos_hlap_x0': cos(h_lap_feat_n, x0_feat_norm).item(),
                'cos_h_u': cos(h_feat_n, u_feat_n).item(),
                'cos_henh_u': cos(h_enh_n, u_feat_n).item(),
                'cos_hlap_u': cos(h_lap_feat_n, u_feat_n).item(),
                'cos_h_hlap': cos(h_feat_n, h_lap_feat_n).item(),
                'h_feat_norm': torch.norm(h_feat, p=2).item(),
                'h_lap_feat_norm': torch.norm(h_lap_feat, p=2).item(),
                'h_enh_feat_norm': torch.norm(h_enh, p=2).item(),
                'laplacian_pixel_mean': h_lap_mag,
                'criterion_raw': C,
                'criterion_norm': C_norm,
            })
        
        # Average over samples
        result = {}
        for key in metrics_list[0].keys():
            result[key] = np.mean([m[key] for m in metrics_list])
        
        return result
    
    def analyze_dataset(self, image_paths, label, num_samples=8, timestep_ratio=0.3):
        """Analyze all images in a list."""
        all_results = []
        
        for path in tqdm(image_paths, desc=f"Analyzing {label}"):
            img_tensor = self.load_image(path)
            if img_tensor is None:
                continue
            
            try:
                result = self.analyze_single_image(img_tensor, num_samples, timestep_ratio)
                result['path'] = path
                result['label'] = label
                all_results.append(result)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
            
            # Clear cache periodically
            if len(all_results) % 50 == 0:
                torch.cuda.empty_cache()
        
        return all_results


def compute_statistics(real_results, gen_results, metric_name):
    """Compute statistical tests for a single metric."""
    real_vals = np.array([r[metric_name] for r in real_results])
    gen_vals = np.array([r[metric_name] for r in gen_results])
    
    # Basic stats
    real_mean, real_std = real_vals.mean(), real_vals.std()
    gen_mean, gen_std = gen_vals.mean(), gen_vals.std()
    
    # Gap
    gap = gen_mean - real_mean
    
    # T-test
    t_stat, t_pval = stats.ttest_ind(real_vals, gen_vals)
    
    # KS test
    ks_stat, ks_pval = stats.ks_2samp(real_vals, gen_vals)
    
    # AUC (treat as binary classification: 0=real, 1=gen)
    labels = np.array([0] * len(real_vals) + [1] * len(gen_vals))
    scores = np.concatenate([real_vals, gen_vals])
    
    # Try both directions for AUC
    auc1 = roc_auc_score(labels, scores)
    auc2 = roc_auc_score(labels, -scores)
    auc = max(auc1, auc2)
    auc_direction = "higher=gen" if auc1 > auc2 else "lower=gen"
    
    # Cohen's d (effect size)
    pooled_std = np.sqrt(((len(real_vals)-1)*real_std**2 + (len(gen_vals)-1)*gen_std**2) / 
                         (len(real_vals) + len(gen_vals) - 2))
    cohens_d = abs(gap) / (pooled_std + 1e-8)
    
    return {
        'metric': metric_name,
        'real_mean': real_mean,
        'real_std': real_std,
        'gen_mean': gen_mean,
        'gen_std': gen_std,
        'gap': gap,
        't_stat': t_stat,
        't_pval': t_pval,
        'ks_stat': ks_stat,
        'ks_pval': ks_pval,
        'auc': auc,
        'auc_direction': auc_direction,
        'cohens_d': cohens_d,
        'real_vals': real_vals,
        'gen_vals': gen_vals,
    }


def create_distribution_plots(stats_dict, output_dir):
    """Create distribution plots for all metrics."""
    
    metrics = list(stats_dict.keys())
    n_metrics = len(metrics)
    
    # Sort by AUC
    sorted_metrics = sorted(metrics, key=lambda x: stats_dict[x]['auc'], reverse=True)
    
    # Create figure with subplots
    n_cols = 4
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3*n_rows))
    axes = axes.flatten()
    
    for i, metric in enumerate(sorted_metrics):
        ax = axes[i]
        s = stats_dict[metric]
        
        # Plot histograms
        ax.hist(s['real_vals'], bins=30, alpha=0.6, color='#2E86AB', label='Real', density=True)
        ax.hist(s['gen_vals'], bins=30, alpha=0.6, color='#E94F37', label='Gen', density=True)
        
        # Add mean lines
        ax.axvline(s['real_mean'], color='#2E86AB', linestyle='--', linewidth=2)
        ax.axvline(s['gen_mean'], color='#E94F37', linestyle='--', linewidth=2)
        
        ax.set_title(f"{metric}\nAUC={s['auc']:.3f}, d={s['cohens_d']:.2f}", fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_metrics, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_distributions.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'all_distributions.pdf'), bbox_inches='tight')
    plt.close()


def create_top_metrics_plot(stats_dict, output_dir, top_n=6):
    """Create detailed plot for top N metrics by AUC."""
    
    sorted_metrics = sorted(stats_dict.keys(), key=lambda x: stats_dict[x]['auc'], reverse=True)
    top_metrics = sorted_metrics[:top_n]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(top_metrics):
        ax = axes[i]
        s = stats_dict[metric]
        
        # Histogram
        ax.hist(s['real_vals'], bins=40, alpha=0.7, color='#2E86AB', label=f"Real (μ={s['real_mean']:.3f})", density=True)
        ax.hist(s['gen_vals'], bins=40, alpha=0.7, color='#E94F37', label=f"Gen (μ={s['gen_mean']:.3f})", density=True)
        
        ax.axvline(s['real_mean'], color='#2E86AB', linestyle='--', linewidth=2)
        ax.axvline(s['gen_mean'], color='#E94F37', linestyle='--', linewidth=2)
        
        ax.set_xlabel(metric, fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f"AUC = {s['auc']:.3f} | Cohen's d = {s['cohens_d']:.2f} | KS = {s['ks_stat']:.3f}", 
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    
    plt.suptitle(f'Top {top_n} Most Discriminative Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_metrics.png'), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'top_metrics.pdf'), bbox_inches='tight')
    plt.close()


def create_roc_curves(stats_dict, output_dir, top_n=6):
    """Create ROC curves for top metrics."""
    
    sorted_metrics = sorted(stats_dict.keys(), key=lambda x: stats_dict[x]['auc'], reverse=True)
    top_metrics = sorted_metrics[:top_n]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, top_n))
    
    for i, metric in enumerate(top_metrics):
        s = stats_dict[metric]
        
        labels = np.array([0] * len(s['real_vals']) + [1] * len(s['gen_vals']))
        scores = np.concatenate([s['real_vals'], s['gen_vals']])
        
        # Use correct direction
        if s['auc_direction'] == "lower=gen":
            scores = -scores
        
        fpr, tpr, _ = roc_curve(labels, scores)
        ax.plot(fpr, tpr, color=colors[i], linewidth=2, label=f"{metric} (AUC={s['auc']:.3f})")
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves for Top Metrics', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'roc_curves.pdf'), bbox_inches='tight')
    plt.close()


def create_summary_bar_chart(stats_dict, output_dir):
    """Create bar chart comparing AUC of all metrics."""
    
    sorted_metrics = sorted(stats_dict.keys(), key=lambda x: stats_dict[x]['auc'], reverse=True)
    
    aucs = [stats_dict[m]['auc'] for m in sorted_metrics]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#E94F37' if auc > 0.7 else '#FFA500' if auc > 0.6 else '#2E86AB' for auc in aucs]
    
    bars = ax.bar(range(len(sorted_metrics)), aucs, color=colors, alpha=0.8)
    
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Random (AUC=0.5)')
    ax.axhline(0.7, color='orange', linestyle='--', linewidth=1, label='Acceptable (AUC=0.7)')
    ax.axhline(0.9, color='green', linestyle='--', linewidth=1, label='Excellent (AUC=0.9)')
    
    ax.set_xticks(range(len(sorted_metrics)))
    ax.set_xticklabels(sorted_metrics, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('AUC Comparison Across All Metrics', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim([0.4, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{auc:.2f}', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'auc_comparison.png'), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'auc_comparison.pdf'), bbox_inches='tight')
    plt.close()


def print_report(stats_dict):
    """Print summary report."""
    
    sorted_metrics = sorted(stats_dict.keys(), key=lambda x: stats_dict[x]['auc'], reverse=True)
    
    print("\n" + "="*100)
    print("SLC DATASET ANALYSIS REPORT")
    print("="*100)
    
    print(f"\n{'Metric':<20} {'Real μ±σ':>15} {'Gen μ±σ':>15} {'Gap':>10} {'AUC':>8} {'Cohen d':>10} {'KS stat':>10}")
    print("-"*100)
    
    for m in sorted_metrics:
        s = stats_dict[m]
        real_str = f"{s['real_mean']:.3f}±{s['real_std']:.3f}"
        gen_str = f"{s['gen_mean']:.3f}±{s['gen_std']:.3f}"
        print(f"{m:<20} {real_str:>15} {gen_str:>15} {s['gap']:>+10.4f} {s['auc']:>8.3f} {s['cohens_d']:>10.2f} {s['ks_stat']:>10.3f}")
    
    print("-"*100)
    
    print("\n🔥 TOP 5 MOST DISCRIMINATIVE METRICS (by AUC):")
    for i, m in enumerate(sorted_metrics[:5]):
        s = stats_dict[m]
        print(f"  {i+1}. {m}: AUC={s['auc']:.3f}, Cohen's d={s['cohens_d']:.2f}, {s['auc_direction']}")
    
    print("\n" + "="*100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_dir', type=str, required=True, help='Directory containing real images')
    parser.add_argument('--gen_dir', type=str, required=True, help='Directory containing generated images')
    parser.add_argument('--output_dir', type=str, default='dataset_analysis')
    parser.add_argument('--num_samples', type=int, default=8, help='Number of noise samples per image')
    parser.add_argument('--timestep', type=float, default=0.3)
    parser.add_argument('--max_images', type=int, default=None, help='Max images per class (for testing)')
    parser.add_argument('--extensions', type=str, default='jpg,jpeg,png,webp,JPEG', help='Image extensions')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Collect image paths
    extensions = args.extensions.split(',')
    real_paths = []
    gen_paths = []
    
    for ext in extensions:
        real_paths.extend(glob(os.path.join(args.real_dir, f'*.{ext}')))
        real_paths.extend(glob(os.path.join(args.real_dir, f'**/*.{ext}'), recursive=True))
        gen_paths.extend(glob(os.path.join(args.gen_dir, f'*.{ext}')))
        gen_paths.extend(glob(os.path.join(args.gen_dir, f'**/*.{ext}'), recursive=True))
    
    real_paths = list(set(real_paths))
    gen_paths = list(set(gen_paths))
    
    if args.max_images:
        real_paths = real_paths[:args.max_images]
        gen_paths = gen_paths[:args.max_images]
    
    print(f"Found {len(real_paths)} real images and {len(gen_paths)} generated images")
    
    if len(real_paths) == 0 or len(gen_paths) == 0:
        print("Error: No images found!")
        return
    
    # Initialize analyzer
    analyzer = SLCDatasetAnalyzer()
    
    # Analyze datasets
    print("\nAnalyzing real images...")
    real_results = analyzer.analyze_dataset(real_paths, 'real', args.num_samples, args.timestep)
    
    print("\nAnalyzing generated images...")
    gen_results = analyzer.analyze_dataset(gen_paths, 'gen', args.num_samples, args.timestep)
    
    print(f"\nSuccessfully analyzed {len(real_results)} real and {len(gen_results)} generated images")
    
    # Compute statistics for each metric
    metrics = ['cos_h_x0', 'cos_henh_x0', 'cos_hlap_x0', 'cos_h_u', 'cos_henh_u', 
               'cos_hlap_u', 'cos_h_hlap', 'h_feat_norm', 'h_lap_feat_norm', 
               'h_enh_feat_norm', 'laplacian_pixel_mean', 'criterion_raw', 'criterion_norm']
    
    stats_dict = {}
    for metric in metrics:
        stats_dict[metric] = compute_statistics(real_results, gen_results, metric)
    
    # Print report
    print_report(stats_dict)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_distribution_plots(stats_dict, args.output_dir)
    create_top_metrics_plot(stats_dict, args.output_dir)
    create_roc_curves(stats_dict, args.output_dir)
    create_summary_bar_chart(stats_dict, args.output_dir)
    
    # Save raw results
    print("Saving raw results...")
    
    # Convert numpy arrays to lists for JSON serialization
    stats_json = {}
    for m, s in stats_dict.items():
        stats_json[m] = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                         for k, v in s.items()}
    
    with open(os.path.join(args.output_dir, 'statistics.json'), 'w') as f:
        json.dump(stats_json, f, indent=2)
    
    # Save raw results as numpy
    np.save(os.path.join(args.output_dir, 'real_results.npy'), real_results)
    np.save(os.path.join(args.output_dir, 'gen_results.npy'), gen_results)
    
    print(f"\n✅ All results saved to {args.output_dir}/")
    print(f"   - all_distributions.png: Distribution plots for all metrics")
    print(f"   - top_metrics.png: Detailed view of top 6 metrics")
    print(f"   - roc_curves.png: ROC curves")
    print(f"   - auc_comparison.png: AUC bar chart")
    print(f"   - statistics.json: All statistics")


if __name__ == "__main__":
    main()