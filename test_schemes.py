#!/usr/bin/env python
"""
快速测试不同 PDE scheme 的 AUC

用法:
    python test_schemes.py --dataset-dir ./dataset --max-samples 100
"""

import os
import sys
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classification_experiment import load_dataset, find_images
from config import EvalConfig


def test_scheme(scheme, image_paths, labels, config_base):
    """测试单个 scheme"""
    from criteria.pde_grounded import PDEGroundedCriterion
    from models import ModelManager
    from image_utils import ImageProcessor
    from tqdm import tqdm
    import torch
    
    # 创建 config
    config = EvalConfig(
        criterion_name="pde",
        device=config_base.device,
        batch_size=config_base.batch_size,
        num_noise=config_base.num_noise,
        time_frac=config_base.time_frac,
        return_terms=True,
    )
    config.method = scheme  # 设置具体方法
    
    # 初始化
    model_manager = ModelManager(config)
    criterion = PDEGroundedCriterion(config, model_manager)
    image_processor = ImageProcessor(config)
    
    scores = []
    
    # 分批处理
    for i in tqdm(range(0, len(image_paths), config.batch_size), desc=f"Testing {scheme}"):
        batch_paths = image_paths[i:i + config.batch_size]
        
        # 加载图片
        raw_images = image_processor.load_image_batch(batch_paths)
        processed = image_processor.preprocess(raw_images).to(config.device)
        
        # 生成 caption（简化：使用空 prompt）
        prompts = ["a photo"] * len(batch_paths)
        
        # 评估
        results = criterion.evaluate_batch(processed, prompts, raw_images)
        
        for r in results:
            scores.append(r['criterion'])
    
    scores = np.array(scores)
    labels = np.array(labels)
    
    # 计算 AUC（两个方向都试）
    auc_higher = roc_auc_score(labels, scores)
    auc_lower = roc_auc_score(labels, -scores)
    
    best_auc = max(auc_higher, auc_lower)
    direction = "higher=fake" if auc_higher > auc_lower else "lower=fake"
    
    # 清理（如果有 cleanup 方法）
    if hasattr(model_manager, 'cleanup'):
        model_manager.cleanup()
    
    return {
        'scheme': scheme,
        'auc': best_auc,
        'direction': direction,
        'real_mean': scores[labels == 0].mean(),
        'fake_mean': scores[labels == 1].mean(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-noise", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--schemes", type=str, nargs='+', 
                       default=["ser", "nsr", "projection", "ser_hf", "combined"])
    args = parser.parse_args()
    
    # 加载数据
    print("Loading dataset...")
    image_paths, labels = load_dataset(
        dataset_dir=args.dataset_dir,
        max_samples=args.max_samples
    )
    
    # 基础 config
    config_base = EvalConfig(
        device=args.device,
        batch_size=args.batch_size,
        num_noise=args.num_noise,
    )
    
    # 测试每个 scheme
    results = []
    for scheme in args.schemes:
        print(f"\n{'='*50}")
        print(f"Testing scheme: {scheme}")
        print(f"{'='*50}")
        
        try:
            result = test_scheme(scheme, image_paths, labels, config_base)
            results.append(result)
            print(f"  AUC: {result['auc']:.4f} ({result['direction']})")
            print(f"  Real mean: {result['real_mean']:.6f}")
            print(f"  Fake mean: {result['fake_mean']:.6f}")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    # 汇总
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Scheme':<15} {'AUC':<8} {'Direction':<15} {'Real Mean':<12} {'Fake Mean':<12}")
    print("-"*60)
    for r in sorted(results, key=lambda x: -x['auc']):
        print(f"{r['scheme']:<15} {r['auc']:<8.4f} {r['direction']:<15} {r['real_mean']:<12.6f} {r['fake_mean']:<12.6f}")


if __name__ == "__main__":
    main()