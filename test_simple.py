#!/usr/bin/env python
"""
简化版测试 - 专注于有效方法

在 DALLE 数据集上，目前最好的是 cosine_latent (69%)
让我们尝试更多 latent 空间的变体
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


class SimpleLatentCriterion:
    """简化的 Latent 空间 criterion"""
    
    def __init__(self, config, model_manager):
        self.config = config
        self.model_manager = model_manager
        self.device = config.device
        
        # Laplacian kernel
        self._lap_kernel = torch.tensor(
            [[0.0, 1.0, 0.0],
             [1.0, -4.0, 1.0],
             [0.0, 1.0, 0.0]], 
            device=self.device, 
            dtype=torch.float32
        ).view(1, 1, 3, 3)
    
    def laplacian(self, x):
        C = x.shape[1]
        kernel = self._lap_kernel.repeat(C, 1, 1, 1)
        return F.conv2d(x.float(), kernel, padding=1, groups=C)
    
    def normalize_batch(self, x, epsilon_reg=0.0):
        """Normalize to unit sphere"""
        x_flat = x.view(x.shape[0], -1)
        norms = x_flat.norm(dim=1, keepdim=True).clamp_min(1e-8)
        x_normalized = x_flat / norms
        return x_normalized.view_as(x)
    
    def evaluate(self, images, prompts, method="cosine"):
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
        
        # 计算各种统计量
        err = eps_pred.float() - eps.float()
        
        if method == "cosine":
            # Cosine similarity (flatten)
            pred_flat = eps_pred.float().view(eps_pred.shape[0], -1)
            ref_flat = eps.float().view(eps.shape[0], -1)
            stat = F.cosine_similarity(pred_flat, ref_flat, dim=1)
            
        elif method == "rmse":
            # RMSE
            stat = err.pow(2).mean(dim=(1,2,3)).sqrt()
            
        elif method == "projection":
            # 高频投影比例
            err_hf = self.laplacian(err)
            hf_norm = err_hf.pow(2).sum(dim=(1,2,3)).sqrt()
            total_norm = err.pow(2).sum(dim=(1,2,3)).sqrt().clamp_min(1e-8)
            stat = hf_norm / total_norm
            
        elif method == "cosine_per_channel":
            # 每个 channel 的 cosine，然后平均
            B, C, H, W = eps_pred.shape
            cos_per_ch = []
            for c in range(C):
                pred_ch = eps_pred[:, c].float().view(B, -1)
                ref_ch = eps[:, c].float().view(B, -1)
                cos_per_ch.append(F.cosine_similarity(pred_ch, ref_ch, dim=1))
            stat = torch.stack(cos_per_ch, dim=1).mean(dim=1)
            
        elif method == "l1":
            # L1 误差
            stat = err.abs().mean(dim=(1,2,3))
            
        elif method == "max_err":
            # 最大误差
            stat = err.abs().view(err.shape[0], -1).max(dim=1)[0]
            
        elif method == "std_err":
            # 误差的标准差
            stat = err.view(err.shape[0], -1).std(dim=1)
            
        elif method == "kurtosis":
            # 误差的峰度 (检测异常分布)
            err_flat = err.view(err.shape[0], -1)
            mean = err_flat.mean(dim=1, keepdim=True)
            std = err_flat.std(dim=1, keepdim=True).clamp_min(1e-8)
            z = (err_flat - mean) / std
            stat = z.pow(4).mean(dim=1) - 3  # excess kurtosis
            
        elif method == "entropy":
            # 误差直方图的熵 (近似)
            err_flat = err.view(err.shape[0], -1)
            # 简化：用标准差作为 entropy proxy
            stat = err_flat.std(dim=1)
            
        elif method == "snr":
            # Signal-to-noise ratio: ||eps|| / ||err||
            eps_norm = eps.float().pow(2).sum(dim=(1,2,3)).sqrt()
            err_norm = err.pow(2).sum(dim=(1,2,3)).sqrt().clamp_min(1e-8)
            stat = eps_norm / err_norm
            
        elif method == "relative_err":
            # 相对误差: ||err|| / ||eps||
            eps_norm = eps.float().pow(2).sum(dim=(1,2,3)).sqrt().clamp_min(1e-8)
            err_norm = err.pow(2).sum(dim=(1,2,3)).sqrt()
            stat = err_norm / eps_norm
            
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
        default=["cosine", "rmse", "projection", "cosine_per_channel", 
                 "l1", "max_err", "std_err", "kurtosis", "snr", "relative_err"])
    args = parser.parse_args()
    
    print("=" * 60)
    print("Simple Latent Criterion Test")
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
    criterion = SimpleLatentCriterion(config, model_manager)
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
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Method':<18} {'AUC':<8} {'Direction':<15} {'Real Mean':<14} {'Fake Mean':<14}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: -x['auc']):
        print(f"{r['method']:<18} {r['auc']:<8.4f} {r['direction']:<15} {r['real_mean']:<14.6f} {r['fake_mean']:<14.6f}")
    
    if results:
        best = max(results, key=lambda x: x['auc'])
        print(f"\n最佳: {best['method']} (AUC = {best['auc']:.4f})")


if __name__ == "__main__":
    main()