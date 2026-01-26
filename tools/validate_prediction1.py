"""
Experiment: Validate High-Frequency Energy Dominance (Prediction 1)
====================================================================

Verifies: E^high_K(h^dec_G) > E^high_K(h^dec_R)

This script:
1. Loads real and generated images
2. Computes h^dec = D(ε_θ(z_t, t)) for each image
3. Computes high-frequency energy ratio E^high_K / ||f||^2 for different K values
4. Reports statistics and generates Table 1

Usage:
    python validate_prediction1.py --real_dir /path/to/real --gen_dirs /path/to/gen1 /path/to/gen2 ...
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
import pandas as pd
from collections import defaultdict
import json

# ============================================================
# Configuration
# ============================================================

class Config:
    # Diffusion settings
    timestep_ratio = 0.3  # t/T
    num_inference_steps = 1000
    
    # Frequency analysis settings
    K_ratios = [1/8, 1/6, 1/4]  # K = N * ratio, where N is image size
    
    # Processing
    image_size = 512
    batch_size = 4
    num_samples = 8  # noise samples per image
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Dataset
# ============================================================

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None, max_images=None):
        self.image_dir = image_dir
        self.transform = transform
        
        # Collect image paths
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
            # Return a blank image on error
            return torch.zeros(3, Config.image_size, Config.image_size), img_path


# ============================================================
# Frequency Analysis Functions
# ============================================================

def compute_fft_energy(x):
    """
    Compute 2D FFT energy spectrum.
    
    Args:
        x: tensor of shape (B, C, H, W)
    
    Returns:
        energy: tensor of shape (B, H, W) - radially averaged energy
        freq_map: tensor of shape (H, W) - frequency magnitude at each point
    """
    B, C, H, W = x.shape
    
    # Convert to float32 for FFT (doesn't support float16)
    x = x.float()
    
    # Compute 2D FFT for each channel
    fft = torch.fft.fft2(x, dim=(-2, -1))
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
    
    # Compute energy (magnitude squared), average over channels
    energy = (fft_shifted.abs() ** 2).mean(dim=1)  # (B, H, W)
    
    # Create frequency magnitude map
    cy, cx = H // 2, W // 2
    y, x_coord = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    freq_map = torch.sqrt((y - cy).float() ** 2 + (x_coord - cx).float() ** 2)
    freq_map = freq_map.to(x.device)
    
    return energy, freq_map


def compute_high_freq_energy_ratio(x, K_ratio):
    """
    Compute E^high_K / ||f||^2.
    
    Args:
        x: tensor of shape (B, C, H, W)
        K_ratio: K = N * K_ratio, where N = min(H, W) / 2
    
    Returns:
        ratios: tensor of shape (B,) - high-freq energy ratio for each sample
    """
    B, C, H, W = x.shape
    N = min(H, W) // 2  # Nyquist frequency
    K = N * K_ratio
    
    energy, freq_map = compute_fft_energy(x)
    
    # Total energy
    total_energy = energy.sum(dim=(-2, -1))  # (B,)
    
    # High-frequency energy (frequencies > K)
    high_freq_mask = (freq_map > K).float()
    high_freq_energy = (energy * high_freq_mask).sum(dim=(-2, -1))  # (B,)
    
    # Ratio
    ratios = high_freq_energy / (total_energy + 1e-8)
    
    return ratios


def compute_laplacian_norm(x):
    """
    Compute mean absolute Laplacian: (1/HWC) * sum |Δx|
    
    Args:
        x: tensor of shape (B, C, H, W)
    
    Returns:
        norms: tensor of shape (B,)
    """
    # Convert to float32
    x = x.float()
    
    # 3x3 Laplacian kernel
    laplacian_kernel = torch.tensor([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    
    # Apply to each channel
    B, C, H, W = x.shape
    x_reshaped = x.view(B * C, 1, H, W)
    laplacian = F.conv2d(x_reshaped, laplacian_kernel, padding=1)
    laplacian = laplacian.view(B, C, H, W)
    
    # Mean absolute value (pixel-wise mean)
    norms = laplacian.abs().mean(dim=(1, 2, 3))
    
    return norms


# ============================================================
# Diffusion Model Wrapper
# ============================================================

class DiffusionProbe:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device="cuda"):
        from diffusers import StableDiffusionPipeline, DDPMScheduler
        
        print(f"Loading diffusion model: {model_id}")
        self.device = device
        
        # Load pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
        )
        self.pipe = self.pipe.to(device)
        
        # Extract components
        self.vae = self.pipe.vae
        self.unet = self.pipe.unet
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer
        self.scheduler = DDPMScheduler.from_config(self.pipe.scheduler.config)
        
        # Set to eval mode
        self.vae.eval()
        self.unet.eval()
        self.text_encoder.eval()
        
        # Precompute timestep
        self.scheduler.set_timesteps(Config.num_inference_steps)
        t_idx = int(Config.timestep_ratio * Config.num_inference_steps)
        self.timestep = self.scheduler.timesteps[t_idx]
        
        # Precompute empty text embedding (unconditional)
        self._precompute_empty_embedding()
        
        print(f"Using timestep: {self.timestep.item()} (t/T = {Config.timestep_ratio})")
    
    @torch.no_grad()
    def _precompute_empty_embedding(self):
        """Precompute empty text embedding for unconditional generation."""
        text_input = self.tokenizer(
            [""],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        self.empty_embedding = self.text_encoder(text_input.input_ids.to(self.device))[0]
        # Shape: (1, 77, 768)
    
    @torch.no_grad()
    def get_h_dec(self, images, num_samples=1):
        """
        Compute h^dec = D(ε_θ(z_t, t)) for input images.
        
        Args:
            images: tensor of shape (B, 3, H, W), values in [0, 1]
            num_samples: number of noise samples per image
        
        Returns:
            h_dec: tensor of shape (B, num_samples, 3, H, W)
        """
        B = images.shape[0]
        
        # Normalize to [-1, 1] for VAE
        images = images * 2 - 1
        images = images.to(self.device, dtype=self.vae.dtype)
        
        # Encode to latent space
        latent_dist = self.vae.encode(images).latent_dist
        z0 = latent_dist.sample() * self.vae.config.scaling_factor
        
        # Expand empty embedding to batch size
        encoder_hidden_states = self.empty_embedding.expand(B, -1, -1)
        
        h_dec_list = []
        
        for _ in range(num_samples):
            # Add noise
            noise = torch.randn_like(z0)
            # Normalize to spherical
            noise = noise / noise.view(B, -1).norm(dim=1, keepdim=True).view(B, 1, 1, 1)
            noise = noise * np.sqrt(np.prod(z0.shape[1:]))
            
            # Get alpha values
            alpha_t = self.scheduler.alphas_cumprod[self.timestep]
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            
            # Create noisy latent
            z_t = sqrt_alpha_t * z0 + sqrt_one_minus_alpha_t * noise
            
            # UNet prediction
            t_batch = self.timestep.expand(B).to(self.device)
            h = self.unet(z_t, t_batch, encoder_hidden_states=encoder_hidden_states).sample
            
            # Decode
            h_dec = self.vae.decode(h / self.vae.config.scaling_factor).sample
            h_dec = (h_dec + 1) / 2  # Back to [0, 1]
            h_dec = h_dec.clamp(0, 1)
            
            h_dec_list.append(h_dec)
        
        # Stack: (B, num_samples, 3, H, W)
        h_dec = torch.stack(h_dec_list, dim=1)
        
        return h_dec


# ============================================================
# Main Experiment
# ============================================================

def run_experiment(args):
    """Run the high-frequency energy experiment."""
    
    # Setup
    device = Config.device
    print(f"Using device: {device}")
    
    # Image transform
    transform = transforms.Compose([
        transforms.Resize((Config.image_size, Config.image_size)),
        transforms.ToTensor(),
    ])
    
    # Load diffusion model
    probe = DiffusionProbe(device=device)
    
    # Prepare results storage
    results = defaultdict(lambda: defaultdict(list))
    
    # Process real images
    print("\n" + "=" * 60)
    print("Processing Real Images")
    print("=" * 60)
    
    real_dataset = ImageDataset(args.real_dir, transform=transform, max_images=args.max_images)
    real_loader = DataLoader(real_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=0)
    
    for images, paths in tqdm(real_loader, desc="Real"):
        if images.sum() == 0:  # Skip blank images
            continue
        
        # Get h^dec
        h_dec = probe.get_h_dec(images, num_samples=Config.num_samples)
        
        # Average over noise samples
        h_dec_mean = h_dec.mean(dim=1)  # (B, 3, H, W)
        
        # Compute metrics for each K
        for K_ratio in Config.K_ratios:
            ratios = compute_high_freq_energy_ratio(h_dec_mean.cpu(), K_ratio)
            results['Real'][f'K=N/{int(1/K_ratio)}'].extend(ratios.tolist())
        
        # Laplacian norm
        lap_norms = compute_laplacian_norm(h_dec_mean.cpu())
        results['Real']['laplacian_norm'].extend(lap_norms.tolist())
    
    # Process generated images from each source
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
            
            # Get h^dec
            h_dec = probe.get_h_dec(images, num_samples=Config.num_samples)
            h_dec_mean = h_dec.mean(dim=1)
            
            # Compute metrics
            for K_ratio in Config.K_ratios:
                ratios = compute_high_freq_energy_ratio(h_dec_mean.cpu(), K_ratio)
                results[gen_name][f'K=N/{int(1/K_ratio)}'].extend(ratios.tolist())
            
            lap_norms = compute_laplacian_norm(h_dec_mean.cpu())
            results[gen_name]['laplacian_norm'].extend(lap_norms.tolist())
    
    # ============================================================
    # Generate Table 1
    # ============================================================
    
    print("\n" + "=" * 60)
    print("Results: High-Frequency Energy Ratio E^high_K / ||f||^2")
    print("=" * 60)
    
    # Build table data
    table_data = []
    
    for source in ['Real'] + [os.path.basename(d.rstrip('/')) for d in args.gen_dirs]:
        row = {'Source': source}
        for K_ratio in Config.K_ratios:
            key = f'K=N/{int(1/K_ratio)}'
            values = results[source][key]
            if values:
                row[key] = f"{np.mean(values):.4f}"
            else:
                row[key] = "N/A"
        table_data.append(row)
    
    # Add average of generated
    gen_sources = [os.path.basename(d.rstrip('/')) for d in args.gen_dirs]
    if gen_sources:
        avg_row = {'Source': 'Avg (Generated)'}
        for K_ratio in Config.K_ratios:
            key = f'K=N/{int(1/K_ratio)}'
            all_gen_values = []
            for source in gen_sources:
                all_gen_values.extend(results[source][key])
            if all_gen_values:
                avg_row[key] = f"{np.mean(all_gen_values):.4f}"
            else:
                avg_row[key] = "N/A"
        table_data.append(avg_row)
    
    # Print table
    df = pd.DataFrame(table_data)
    print("\n" + df.to_string(index=False))
    
    # ============================================================
    # Statistical Comparison
    # ============================================================
    
    print("\n" + "=" * 60)
    print("Statistical Comparison: Real vs Generated")
    print("=" * 60)
    
    real_values = {}
    for K_ratio in Config.K_ratios:
        key = f'K=N/{int(1/K_ratio)}'
        real_values[key] = np.array(results['Real'][key])
    
    for source in gen_sources:
        print(f"\n{source}:")
        for K_ratio in Config.K_ratios:
            key = f'K=N/{int(1/K_ratio)}'
            gen_vals = np.array(results[source][key])
            real_vals = real_values[key]
            
            if len(gen_vals) > 0 and len(real_vals) > 0:
                # Cohen's d
                pooled_std = np.sqrt((real_vals.std()**2 + gen_vals.std()**2) / 2)
                cohens_d = (gen_vals.mean() - real_vals.mean()) / pooled_std
                
                # t-test
                from scipy import stats
                t_stat, p_value = stats.ttest_ind(gen_vals, real_vals)
                
                print(f"  {key}: Gen={gen_vals.mean():.4f}, Real={real_vals.mean():.4f}, "
                      f"Δ={gen_vals.mean()-real_vals.mean():.4f}, Cohen's d={cohens_d:.3f}, p={p_value:.2e}")
    
    # ============================================================
    # Laplacian Norm Comparison (Prediction 2)
    # ============================================================
    
    print("\n" + "=" * 60)
    print("Laplacian Norm ||Δh^dec||^2 Comparison (Prediction 2)")
    print("=" * 60)
    
    real_lap = np.array(results['Real']['laplacian_norm'])
    print(f"\nReal: mean={real_lap.mean():.4f}, std={real_lap.std():.4f}")
    
    for source in gen_sources:
        gen_lap = np.array(results[source]['laplacian_norm'])
        ratio = gen_lap.mean() / real_lap.mean()
        
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(gen_lap, real_lap)
        
        print(f"{source}: mean={gen_lap.mean():.4f}, std={gen_lap.std():.4f}, "
              f"ratio={ratio:.2f}x, p={p_value:.2e}")
    
    # ============================================================
    # Save Results
    # ============================================================
    
    output_path = os.path.join(args.output_dir, 'prediction1_results.json')
    
    # Convert to serializable format
    save_results = {}
    for source, metrics in results.items():
        save_results[source] = {k: v for k, v in metrics.items()}
    
    with open(output_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    # Save LaTeX table
    latex_path = os.path.join(args.output_dir, 'table1_hf_energy.tex')
    with open(latex_path, 'w') as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\caption{High-frequency energy ratio $E^{high}_K / \\|f\\|^2$ of decoded noise predictions. ")
        f.write("Generated images consistently exhibit higher ratios than real images.}\n")
        f.write("\\label{tab:hf_energy}\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("Source & $K=N/8$ & $K=N/6$ & $K=N/4$ \\\\\n")
        f.write("\\midrule\n")
        
        for row in table_data:
            if row['Source'] == 'Avg (Generated)':
                f.write("\\midrule\n")
            f.write(f"{row['Source']} & {row.get('K=N/8', 'N/A')} & {row.get('K=N/6', 'N/A')} & {row.get('K=N/4', 'N/A')} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX table saved to {latex_path}")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Prediction 1: High-Frequency Energy Dominance")
    
    parser.add_argument("--real_dir", type=str, required=True,
                        help="Directory containing real images")
    parser.add_argument("--gen_dirs", type=str, nargs='+', required=True,
                        help="Directories containing generated images (one per generator)")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Output directory for results")
    parser.add_argument("--max_images", type=int, default=500,
                        help="Maximum number of images per source")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    run_experiment(args)