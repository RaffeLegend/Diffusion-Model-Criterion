#!/usr/bin/env python
"""
测试改进版 HOSC v2

用法:
    python test_hosc_v2.py --dataset-dir ./dataset --max-samples 100
"""

import os
import sys
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classification_experiment import load_dataset
from config import EvalConfig


def test_method(method, image_paths, labels, config_base, model_manager=None):
    """测试单个方法"""
    from criteria.hosc_criterion import ImprovedHOSCCriterion
    from models import ModelManager
    from image_utils import ImageProcessor
    from tqdm import tqdm
    import torch
    
    config = EvalConfig(
        criterion_name="hosc_v2",
        device=config_base.device,
        batch_size=config_base.batch_size,
        num_noise=config_base.num_noise,
        time_frac=config_base.time_frac,
        return_terms=True,
    )
    config.method = method
    config.time_fracs = [0.005, 0.01, 0.02, 0.05]  # 多尺度
    
    if model_manager is None:
        model_manager = ModelManager(config)
        created_manager = True
    else:
        created_manager = False
    
    criterion = ImprovedHOSCCriterion(config, model_manager)
    image_processor = ImageProcessor(config)
    
    scores = []
    
    for i in tqdm(range(0, len(image_paths), config.batch_size), desc=f"Testing {method}"):
        batch_paths = image_paths[i:i + config.batch_size]
        raw_images = image_processor.load_image_batch(batch_paths)
        processed = image_processor.preprocess(raw_images).to(config.device)
        prompts = ["a photo"] * len(batch_paths)
        results = criterion.evaluate_batch(processed, prompts, raw_images)
        for r in results:
            scores.append(r['criterion'])
    
    scores = np.array(scores)
    labels = np.array(labels)
    
    auc_higher = roc_auc_score(labels, scores)
    auc_lower = roc_auc_score(labels, -scores)
    
    best_auc = max(auc_higher, auc_lower)
    direction = "higher=fake" if auc_higher > auc_lower else "lower=fake"
    
    return {
        'method': method,
        'auc': best_auc,
        'direction': direction,
        'real_mean': scores[labels == 0].mean(),
        'fake_mean': scores[labels == 1].mean(),
    }, model_manager if created_manager else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--real-dir", type=str)
    parser.add_argument("--fake-dir", type=str)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-noise", type=int, default=8)
    parser.add_argument("--time-frac", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--methods", type=str, nargs='+', 
                       default=["multiscale", "relative", "grad_align", "cosine", "smoothness", "ensemble"])
    args = parser.parse_args()
    
    print("=" * 60)
    print("Loading dataset...")
    print("=" * 60)
    
    image_paths, labels = load_dataset(
        dataset_dir=args.dataset_dir,
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        max_samples=args.max_samples
    )
    
    print(f"Total: {len(image_paths)} images")
    print()
    
    config_base = EvalConfig(
        device=args.device,
        batch_size=args.batch_size,
        num_noise=args.num_noise,
        time_frac=args.time_frac,
    )
    
    results = []
    model_manager = None
    
    for method in args.methods:
        print(f"\n{'='*50}")
        print(f"Testing: {method}")
        print(f"{'='*50}")
        
        try:
            result, manager = test_method(method, image_paths, labels, config_base, model_manager)
            if manager is not None:
                model_manager = manager
            
            results.append(result)
            print(f"  AUC: {result['auc']:.4f} ({result['direction']})")
            print(f"  Real: {result['real_mean']:.6f}, Fake: {result['fake_mean']:.6f}")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("SUMMARY - Improved HOSC v2")
    print("=" * 80)
    print(f"{'Method':<15} {'AUC':<8} {'Direction':<15} {'Real Mean':<14} {'Fake Mean':<14}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: -x['auc']):
        print(f"{r['method']:<15} {r['auc']:<8.4f} {r['direction']:<15} {r['real_mean']:<14.6f} {r['fake_mean']:<14.6f}")
    
    if results:
        best = max(results, key=lambda x: x['auc'])
        print(f"\n最佳方法: {best['method']} (AUC = {best['auc']:.4f})")


if __name__ == "__main__":
    main()