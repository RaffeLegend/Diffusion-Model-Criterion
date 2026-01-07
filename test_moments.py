#!/usr/bin/env python
"""
测试高阶统计量 (Kurtosis, Skewness 等)

Kurtosis 在 DALLE 上达到 72.31% AUC，是目前最好的方法
"""

import os
import sys
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classification_experiment import load_dataset
from config import EvalConfig
from models import ModelManager
from image_utils import ImageProcessor
from tqdm import tqdm


class HighOrderStatsCriterion:
    """基于高阶统计量的 criterion"""
    
    def __init__(self, config, model_manager):
        self.config = config
        self.model_manager = model_manager
        self.device = config.device
        
        self._lap_kernel = torch.tensor(
            [[0.0, 1.0, 0.0],
             [1.0, -4.0, 1.0],
             [0.0, 1.0, 0.0]], 
            device=self.device, dtype=torch.float32
        ).view(1, 1, 3, 3)
    
    def laplacian(self, x):
        C = x.shape[1]
        kernel = self._lap_kernel.repeat(C, 1, 1, 1)
        return F.conv2d(x.float(), kernel, padding=1, groups=C)
    
    def normalize_batch(self, x, epsilon_reg=0.0):
        x_flat = x.view(x.shape[0], -1)
        norms = x_flat.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return (x_flat / norms).view_as(x)
    
    def compute_moments(self, x):
        """计算各阶矩"""
        x_flat = x.view(x.shape[0], -1).float()
        mean = x_flat.mean(dim=1, keepdim=True)
        centered = x_flat - mean
        std = centered.std(dim=1, keepdim=True).clamp_min(1e-8)
        z = centered / std  # standardized
        
        # 各阶矩
        m2 = (z ** 2).mean(dim=1)  # should be ~1
        m3 = (z ** 3).mean(dim=1)  # skewness
        m4 = (z ** 4).mean(dim=1)  # kurtosis + 3
        m5 = (z ** 5).mean(dim=1)  # 5th moment
        m6 = (z ** 6).mean(dim=1)  # 6th moment
        
        return {
            'variance': std.squeeze() ** 2,
            'skewness': m3,
            'kurtosis': m4 - 3,  # excess kurtosis
            'moment5': m5,
            'moment6': m6 - 15,  # excess 6th moment (Gaussian has 15)
        }
    
    def evaluate(self, images, prompts, method="kurtosis"):
        num_images = images.shape[0]
        num_noise = self.config.num_noise
        
        unet = self.model_manager.unet
        scheduler = self.model_manager.scheduler
        tokenizer = self.model_manager.tokenizer
        text_encoder = self.model_manager.text_encoder
        vae = self.model_manager.vae
        
        # Text embeddings
        expanded_prompts = []
        for p in prompts:
            expanded_prompts.extend([p] * num_noise)
        
        text_tokens = tokenizer(
            expanded_prompts, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt",
        )
        with torch.no_grad():
            text_emb = text_encoder(text_tokens.input_ids.to(self.device)).last_hidden_state
        
        # Encode
        with torch.no_grad():
            z0 = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
        
        z0_rep = z0.repeat_interleave(num_noise, dim=0).half()
        
        # Noise
        eps = torch.randn_like(z0_rep).half()
        eps = self.normalize_batch(eps, self.config.epsilon_reg).half()
        sqrt_d = torch.prod(torch.tensor(z0_rep.shape[1:], device=self.device)).float().sqrt()
        eps = eps * sqrt_d
        
        # Timestep
        t = int(self.config.time_frac * scheduler.config.num_train_timesteps)
        t = max(1, min(t, scheduler.config.num_train_timesteps - 1))
        t_tensor = torch.full((z0_rep.shape[0],), t, device=self.device, dtype=torch.long)
        
        # z_t
        zt = scheduler.add_noise(z0_rep, eps, t_tensor).half()
        
        # Predict
        with torch.no_grad():
            eps_pred = unet(zt, t_tensor, encoder_hidden_states=text_emb)[0].half()
        
        # Error
        err = eps_pred.float() - eps.float()
        
        # 计算高阶统计量
        moments = self.compute_moments(err)
        
        if method == "kurtosis":
            stat = moments['kurtosis']
        elif method == "skewness":
            stat = moments['skewness'].abs()  # 取绝对值
        elif method == "moment5":
            stat = moments['moment5'].abs()
        elif method == "moment6":
            stat = moments['moment6']
        elif method == "kurtosis_hf":
            # 高频误差的 kurtosis
            err_hf = self.laplacian(err)
            moments_hf = self.compute_moments(err_hf)
            stat = moments_hf['kurtosis']
        elif method == "combined_moments":
            # 组合多个统计量
            k = moments['kurtosis']
            s = moments['skewness'].abs()
            # 归一化后相加
            k_norm = (k - k.mean()) / (k.std() + 1e-8)
            s_norm = (s - s.mean()) / (s.std() + 1e-8)
            stat = k_norm + 0.5 * s_norm
        elif method == "tail_ratio":
            # 尾部比例: 超过 2σ 的比例
            err_flat = err.view(err.shape[0], -1)
            mean = err_flat.mean(dim=1, keepdim=True)
            std = err_flat.std(dim=1, keepdim=True).clamp_min(1e-8)
            z = ((err_flat - mean) / std).abs()
            tail_ratio = (z > 2).float().mean(dim=1)
            stat = tail_ratio
        elif method == "tail_ratio_3sigma":
            # 超过 3σ 的比例
            err_flat = err.view(err.shape[0], -1)
            mean = err_flat.mean(dim=1, keepdim=True)
            std = err_flat.std(dim=1, keepdim=True).clamp_min(1e-8)
            z = ((err_flat - mean) / std).abs()
            stat = (z > 3).float().mean(dim=1)
        elif method == "percentile_range":
            # 95th - 5th percentile range (normalized)
            err_flat = err.view(err.shape[0], -1)
            p95 = torch.quantile(err_flat, 0.95, dim=1)
            p05 = torch.quantile(err_flat, 0.05, dim=1)
            std = err_flat.std(dim=1).clamp_min(1e-8)
            stat = (p95 - p05) / std
        elif method == "entropy_approx":
            # 基于直方图的熵近似
            err_flat = err.view(err.shape[0], -1)
            # 简化：用 log(std) 作为熵的 proxy
            std = err_flat.std(dim=1).clamp_min(1e-8)
            stat = torch.log(std)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # 聚合
        stat = stat.view(num_images, num_noise).mean(dim=1)
        return stat.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-noise", type=int, default=8)
    parser.add_argument("--time-frac", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--methods", type=str, nargs='+', 
        default=["kurtosis", "skewness", "moment5", "moment6", 
                 "kurtosis_hf", "combined_moments", "tail_ratio", 
                 "tail_ratio_3sigma", "percentile_range"])
    args = parser.parse_args()
    
    print("=" * 60)
    print("High-Order Statistics Test")
    print("=" * 60)
    
    image_paths, labels = load_dataset(dataset_dir=args.dataset_dir, max_samples=args.max_samples)
    labels = np.array(labels)
    print(f"Total: {len(image_paths)} images\n")
    
    config = EvalConfig(
        device=args.device,
        batch_size=args.batch_size,
        num_noise=args.num_noise,
        time_frac=args.time_frac,
    )
    
    model_manager = ModelManager(config)
    criterion = HighOrderStatsCriterion(config, model_manager)
    image_processor = ImageProcessor(config)
    
    results = []
    
    for method in args.methods:
        print(f"Testing: {method}...")
        
        all_scores = []
        for i in tqdm(range(0, len(image_paths), config.batch_size), desc=method):
            batch_paths = image_paths[i:i + config.batch_size]
            raw_images = image_processor.load_image_batch(batch_paths)
            processed = image_processor.preprocess(raw_images).to(config.device)
            prompts = ["a photo"] * len(batch_paths)
            
            try:
                scores = criterion.evaluate(processed, prompts, method=method)
                all_scores.extend(scores)
            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()
                all_scores.extend([float('nan')] * len(batch_paths))
        
        scores = np.array(all_scores)
        valid = ~np.isnan(scores)
        scores = scores[valid]
        labs = labels[valid]
        
        if len(scores) > 0:
            auc_h = roc_auc_score(labs, scores)
            auc_l = roc_auc_score(labs, -scores)
            best_auc = max(auc_h, auc_l)
            direction = "higher=fake" if auc_h > auc_l else "lower=fake"
            
            results.append({
                'method': method,
                'auc': best_auc,
                'direction': direction,
                'real_mean': scores[labs == 0].mean(),
                'fake_mean': scores[labs == 1].mean(),
            })
            print(f"  AUC: {best_auc:.4f} ({direction})")
    
    print("\n" + "=" * 80)
    print("SUMMARY - High-Order Statistics")
    print("=" * 80)
    print(f"{'Method':<20} {'AUC':<8} {'Direction':<15} {'Real Mean':<14} {'Fake Mean':<14}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: -x['auc']):
        print(f"{r['method']:<20} {r['auc']:<8.4f} {r['direction']:<15} {r['real_mean']:<14.6f} {r['fake_mean']:<14.6f}")
    
    if results:
        best = max(results, key=lambda x: x['auc'])
        print(f"\n最佳: {best['method']} (AUC = {best['auc']:.4f})")


if __name__ == "__main__":
    main()