"""
Experiment: Validate Laplacian Norm Gap (Prediction 2)
======================================================

Verifies: E[||Δh^dec_G||^2] > E[||Δh^dec_R||^2]

This script:
1. Loads real and generated images
2. Computes h^dec = D(ε_θ(z_t, t)) for each image
3. Computes Laplacian norm ||Δh^dec||
4. Generates Figure 2: histogram + box plot

Usage:
    python validate_prediction2.py --real_dir /path/to/real --gen_dirs /path/to/gen1 /path/to/gen2 ...
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
import json
from scipy import stats

# ============================================================
# Configuration
# ============================================================

class Config:
    # Diffusion settings
    timestep_ratio = 0.3
    num_inference_steps = 1000
    
    # Processing
    image_size = 512
    batch_size = 4
    num_samples = 8
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Dataset
# ============================================================

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None, max_images=None):
        self.image_dir = image_dir
        self.transform = transform
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        self.image_paths = []
        
        for root, _, files in os.walk(image_dir):
            for f in files:
                if os.path.splitext(f)[1].lower() in valid_extensions:
                    self.image_paths.append(os.path.join(root, f))
        
        self.image_paths.sort()
        if max_images:
            self.image_paths = self.image_paths[:max_images]
        
        print(f"Found {len(self.image_paths)} images in {image_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, img_path
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(3, Config.image_size, Config.image_size), img_path


# ============================================================
# Laplacian Computation
# ============================================================

def compute_laplacian_norm(x):
    """
    Compute mean absolute Laplacian: (1/HWC) * sum |Δx|
    
    This is more interpretable and matches the paper's laplacian_pixel_mean metric.
    
    Args:
        x: tensor of shape (B, C, H, W)
    
    Returns:
        norms: tensor of shape (B,)
    """
    x = x.float()
    
    laplacian_kernel = torch.tensor([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    
    B, C, H, W = x.shape
    x_reshaped = x.view(B * C, 1, H, W)
    laplacian = F.conv2d(x_reshaped, laplacian_kernel, padding=1)
    laplacian = laplacian.view(B, C, H, W)
    
    # Mean absolute value (pixel-wise mean)
    norms = laplacian.abs().mean(dim=(1, 2, 3))
    
    return norms


def compute_laplacian_map(x):
    """
    Compute pixel-wise Laplacian magnitude for visualization.
    
    Args:
        x: tensor of shape (B, C, H, W)
    
    Returns:
        lap_map: tensor of shape (B, H, W) - magnitude per pixel
    """
    x = x.float()
    
    laplacian_kernel = torch.tensor([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    
    B, C, H, W = x.shape
    x_reshaped = x.view(B * C, 1, H, W)
    laplacian = F.conv2d(x_reshaped, laplacian_kernel, padding=1)
    laplacian = laplacian.view(B, C, H, W)
    
    # Per-pixel magnitude (averaged over channels)
    lap_map = torch.sqrt((laplacian ** 2).mean(dim=1))
    
    return lap_map


# ============================================================
# Diffusion Model
# ============================================================

class DiffusionProbe:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device="cuda"):
        from diffusers import StableDiffusionPipeline, DDPMScheduler
        
        print(f"Loading diffusion model: {model_id}")
        self.device = device
        
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
        )
        self.pipe = self.pipe.to(device)
        
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer
        self.scheduler = DDPMScheduler.from_config(self.pipe.scheduler.config)
        
        self.vae.eval()
        self.unet.eval()
        self.text_encoder.eval()
        
        self.scheduler.set_timesteps(Config.num_inference_steps)
        t_idx = int(Config.timestep_ratio * Config.num_inference_steps)
        self.timestep = self.scheduler.timesteps[t_idx]
        
        self._precompute_empty_embedding()
        
        print(f"Using timestep: {self.timestep.item()} (t/T = {Config.timestep_ratio})")
    
    @torch.no_grad()
    def _precompute_empty_embedding(self):
        text_input = self.tokenizer(
            [""],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        self.empty_embedding = self.text_encoder(text_input.input_ids.to(self.device))[0]
    
    @torch.no_grad()
    def get_h_dec(self, images, num_samples=1):
        B = images.shape[0]
        
        images = images * 2 - 1
        images = images.to(self.device, dtype=self.vae.dtype)
        
        latent_dist = self.vae.encode(images).latent_dist
        z0 = latent_dist.sample() * self.vae.config.scaling_factor
        
        encoder_hidden_states = self.empty_embedding.expand(B, -1, -1)
        
        h_dec_list = []
        
        for _ in range(num_samples):
            noise = torch.randn_like(z0)
            noise = noise / noise.view(B, -1).norm(dim=1, keepdim=True).view(B, 1, 1, 1)
            noise = noise * np.sqrt(np.prod(z0.shape[1:]))
            
            alpha_t = self.scheduler.alphas_cumprod[self.timestep]
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            
            z_t = sqrt_alpha_t * z0 + sqrt_one_minus_alpha_t * noise
            
            t_batch = self.timestep.expand(B).to(self.device)
            h = self.unet(z_t, t_batch, encoder_hidden_states=encoder_hidden_states).sample
            
            h_dec = self.vae.decode(h / self.vae.config.scaling_factor).sample
            h_dec = (h_dec + 1) / 2
            h_dec = h_dec.clamp(0, 1)
            
            h_dec_list.append(h_dec)
        
        h_dec = torch.stack(h_dec_list, dim=1)
        
        return h_dec


# ============================================================
# Visualization
# ============================================================

COLORS = {
    'real': '#4A90A4',
    'gen': '#C76B6B',
    'real_light': '#A8D0E0',
    'gen_light': '#E8B8B8',
    'text': '#2C3E50',
    'subtext': '#7F8C8D',
}


def create_figure2(real_norms, gen_norms_dict, output_path):
    """
    Create Figure 2: Laplacian norm distribution.
    (a) Histogram with Real vs Generated (aggregated)
    (b) Box plot by generator
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Aggregate all generated
    all_gen_norms = np.concatenate(list(gen_norms_dict.values()))
    
    # === (a) Histogram ===
    ax = axes[0]
    
    # Compute statistics
    real_mean = np.mean(real_norms)
    gen_mean = np.mean(all_gen_norms)
    ratio = gen_mean / real_mean
    
    # KDE-style histogram
    bins = np.linspace(
        min(real_norms.min(), all_gen_norms.min()),
        max(real_norms.max(), all_gen_norms.max()),
        50
    )
    
    ax.hist(real_norms, bins=bins, density=True, alpha=0.6, color=COLORS['real'], 
            label=f'Real (μ={real_mean:.1f})', edgecolor='white', linewidth=0.5)
    ax.hist(all_gen_norms, bins=bins, density=True, alpha=0.6, color=COLORS['gen'],
            label=f'Generated (μ={gen_mean:.1f})', edgecolor='white', linewidth=0.5)
    
    # Mean lines
    ax.axvline(real_mean, color=COLORS['real'], linestyle='--', linewidth=2, alpha=0.8)
    ax.axvline(gen_mean, color=COLORS['gen'], linestyle='--', linewidth=2, alpha=0.8)
    
    # Ratio annotation
    ax.annotate(f'{ratio:.2f}×', 
                xy=(gen_mean, ax.get_ylim()[1] * 0.9),
                fontsize=12, fontweight='bold', color=COLORS['text'],
                ha='center')
    
    ax.set_xlabel(r'$\|\Delta \mathbf{h}^{\mathrm{dec}}\|$', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('(a) Distribution of Laplacian Norm', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # === (b) Box plot by generator ===
    ax = axes[1]
    
    # Prepare data
    labels = ['Real'] + list(gen_norms_dict.keys())
    data = [real_norms] + list(gen_norms_dict.values())
    
    # Box plot
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
    
    # Color boxes
    colors = [COLORS['real']] + [COLORS['gen']] * len(gen_norms_dict)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    # Median line color
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(1.5)
    
    ax.set_ylabel(r'$\|\Delta \mathbf{h}^{\mathrm{dec}}\|$', fontsize=12)
    ax.set_title('(b) Laplacian Norm by Generator', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Figure saved to {output_path}")


def create_qualitative_figure(probe, real_image_path, gen_image_path, output_path, transform):
    """
    Create Figure 6: Qualitative comparison of Laplacian response.
    Shows input image, h^dec, Laplacian map, and histogram.
    """
    
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    
    for row, (img_path, label) in enumerate([(real_image_path, 'Real'), (gen_image_path, 'Generated')]):
        # Load image
        image = Image.open(img_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        
        # Get h^dec
        h_dec = probe.get_h_dec(image_tensor, num_samples=1)
        h_dec = h_dec[:, 0]  # (1, 3, H, W)
        
        # Compute Laplacian map
        lap_map = compute_laplacian_map(h_dec.cpu())[0].numpy()  # (H, W)
        lap_norm = compute_laplacian_norm(h_dec.cpu())[0].item()
        
        # Convert tensors to numpy for display (ensure float32)
        image_np = image_tensor[0].permute(1, 2, 0).numpy().astype(np.float32)
        h_dec_np = h_dec[0].cpu().float().permute(1, 2, 0).numpy().astype(np.float32)
        lap_map = lap_map.astype(np.float32)
        
        # (a) Input image
        axes[row, 0].imshow(image_np)
        axes[row, 0].set_title(f'{label}: Input Image', fontsize=11)
        axes[row, 0].axis('off')
        
        # (b) h^dec
        axes[row, 1].imshow(h_dec_np.clip(0, 1))
        axes[row, 1].set_title(r'$\mathbf{h}^{\mathrm{dec}}$', fontsize=11)
        axes[row, 1].axis('off')
        
        # (c) Laplacian response map
        im = axes[row, 2].imshow(lap_map, cmap='hot')
        axes[row, 2].set_title(r'$|\Delta \mathbf{h}^{\mathrm{dec}}|$', fontsize=11)
        axes[row, 2].axis('off')
        plt.colorbar(im, ax=axes[row, 2], fraction=0.046, pad=0.04)
        
        # (d) Histogram of Laplacian values
        axes[row, 3].hist(lap_map.flatten(), bins=50, density=True, 
                         color=COLORS['real'] if label == 'Real' else COLORS['gen'],
                         alpha=0.7, edgecolor='white')
        axes[row, 3].axvline(lap_map.mean(), color='black', linestyle='--', linewidth=1.5)
        axes[row, 3].set_xlabel('Laplacian Magnitude', fontsize=10)
        axes[row, 3].set_ylabel('Density', fontsize=10)
        axes[row, 3].set_title(f'μ = {lap_norm:.2f}', fontsize=11)
        axes[row, 3].spines['top'].set_visible(False)
        axes[row, 3].spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Qualitative figure saved to {output_path}")


# ============================================================
# Main Experiment
# ============================================================

def run_experiment(args):
    device = Config.device
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((Config.image_size, Config.image_size)),
        transforms.ToTensor(),
    ])
    
    probe = DiffusionProbe(device=device)
    
    results = defaultdict(list)
    
    # Process real images
    print("\n" + "=" * 60)
    print("Processing Real Images")
    print("=" * 60)
    
    real_dataset = ImageDataset(args.real_dir, transform=transform, max_images=args.max_images)
    real_loader = DataLoader(real_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=0)
    
    real_image_example = None
    
    for images, paths in tqdm(real_loader, desc="Real"):
        if images.sum() == 0:
            continue
        
        if real_image_example is None:
            real_image_example = paths[0]
        
        h_dec = probe.get_h_dec(images, num_samples=Config.num_samples)
        h_dec_mean = h_dec.mean(dim=1)
        
        lap_norms = compute_laplacian_norm(h_dec_mean.cpu())
        results['Real'].extend(lap_norms.tolist())
    
    # Process generated images
    gen_image_example = None
    
    for gen_dir in args.gen_dirs:
        gen_name = os.path.basename(gen_dir.rstrip('/'))
        
        print("\n" + "=" * 60)
        print(f"Processing Generated Images: {gen_name}")
        print("=" * 60)
        
        gen_dataset = ImageDataset(gen_dir, transform=transform, max_images=args.max_images)
        gen_loader = DataLoader(gen_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=0)
        
        for images, paths in tqdm(gen_loader, desc=gen_name):
            if images.sum() == 0:
                continue
            
            if gen_image_example is None:
                gen_image_example = paths[0]
            
            h_dec = probe.get_h_dec(images, num_samples=Config.num_samples)
            h_dec_mean = h_dec.mean(dim=1)
            
            lap_norms = compute_laplacian_norm(h_dec_mean.cpu())
            results[gen_name].extend(lap_norms.tolist())
    
    # ============================================================
    # Print Results
    # ============================================================
    
    print("\n" + "=" * 60)
    print("Results: Laplacian Norm ||Δh^dec||")
    print("=" * 60)
    
    real_norms = np.array(results['Real'])
    print(f"\nReal: mean={real_norms.mean():.2f}, std={real_norms.std():.2f}")
    
    gen_norms_dict = {}
    for source in results:
        if source == 'Real':
            continue
        gen_norms = np.array(results[source])
        gen_norms_dict[source] = gen_norms
        
        ratio = gen_norms.mean() / real_norms.mean()
        t_stat, p_value = stats.ttest_ind(gen_norms, real_norms)
        
        # Cohen's d
        pooled_std = np.sqrt((real_norms.std()**2 + gen_norms.std()**2) / 2)
        cohens_d = (gen_norms.mean() - real_norms.mean()) / pooled_std
        
        print(f"{source}: mean={gen_norms.mean():.2f}, std={gen_norms.std():.2f}, "
              f"ratio={ratio:.2f}×, Cohen's d={cohens_d:.2f}, p={p_value:.2e}")
    
    # ============================================================
    # Generate Figures
    # ============================================================
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Figure 2: Histogram + Box plot
    fig2_path = os.path.join(args.output_dir, 'figure2_laplacian_norm.png')
    create_figure2(real_norms, gen_norms_dict, fig2_path)
    
    # Figure 6: Qualitative comparison (if we have example images)
    if real_image_example and gen_image_example:
        fig6_path = os.path.join(args.output_dir, 'figure6_qualitative.png')
        create_qualitative_figure(probe, real_image_example, gen_image_example, fig6_path, transform)
    
    # ============================================================
    # Save Results
    # ============================================================
    
    output_path = os.path.join(args.output_dir, 'prediction2_results.json')
    save_results = {k: v if isinstance(v, list) else v.tolist() for k, v in results.items()}
    
    with open(output_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Prediction 2: Laplacian Norm Gap")
    
    parser.add_argument("--real_dir", type=str, required=True,
                        help="Directory containing real images")
    parser.add_argument("--gen_dirs", type=str, nargs='+', required=True,
                        help="Directories containing generated images")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Output directory for results")
    parser.add_argument("--max_images", type=int, default=500,
                        help="Maximum number of images per source")
    
    args = parser.parse_args()
    
    run_experiment(args)