#!/usr/bin/env python
"""
分类实验脚本 - Real vs Fake 图片分类 (支持多方法评估版)

支持按不同生成方法分别输出结果。

用法:
    # 数据目录结构:
    # test_dir/
    #   ├── 0_real/           (真实图像)
    #   └── 1_fake/
    #       ├── stylegan2/    (方法1)
    #       ├── ddpm/         (方法2)
    #       ├── ldm/          (方法3)
    #       └── ...
    
    python classification_experiment_multimethod.py \
        --calibration-dir /path/to/calibration/real \
        --test-dir /path/to/test/dataset \
        --mode zero-shot
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ==================== 数据加载 ====================

def find_images(directory: str, recursive: bool = False) -> List[str]:
    """查找目录中的所有图片"""
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.PNG', '.JPG', '.JPEG')
    directory = Path(directory)
    image_paths = []
    
    if not directory.exists():
        print(f"警告: 目录不存在 {directory}")
        return []
    
    if recursive:
        for f in directory.rglob("*"):
            if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                image_paths.append(str(f))
    else:
        for f in directory.iterdir():
            if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                image_paths.append(str(f))
    
    return sorted(image_paths)


def load_dataset_with_methods(
    dataset_dir: str,
    max_samples_per_method: int = None,
) -> Tuple[List[str], List[int], List[str]]:
    """
    加载数据集，返回每个图像对应的生成方法
    
    支持两种目录结构:
    
    结构1 (直接):
    dataset_dir/
      ├── biggan/
      │   ├── 0_real/
      │   └── 1_fake/
      ├── stylegan/
      │   ├── 0_real/
      │   └── 1_fake/
      └── ...
    
    结构2 (带场景):
    dataset_dir/
      ├── cyclegan/
      │   ├── apple/
      │   │   ├── 0_real/
      │   │   └── 1_fake/
      │   ├── horse/
      │   │   ├── 0_real/
      │   │   └── 1_fake/
      │   └── ...
      └── ...
    
    Returns:
        image_paths: 所有图像路径
        labels: 标签 (0=real, 1=fake)
        methods: 每个图像的生成方法
    """
    dataset_path = Path(dataset_dir)
    
    image_paths = []
    labels = []
    methods = []
    
    real_names = {'0_real', 'real', '0', 'Real', '0_Real'}
    fake_names = {'1_fake', 'fake', '1', 'Fake', '1_Fake'}
    
    # 遍历每个方法目录
    method_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    
    for method_dir in method_dirs:
        method_name = method_dir.name
        method_real_paths = []
        method_fake_paths = []
        
        # 检查是 结构1 还是 结构2
        subdirs = [d.name for d in method_dir.iterdir() if d.is_dir()]
        
        has_real_fake_directly = bool(set(subdirs) & (real_names | fake_names))
        
        if has_real_fake_directly:
            # 结构1: method/0_real, method/1_fake
            for subdir in method_dir.iterdir():
                if subdir.is_dir():
                    if subdir.name in real_names:
                        method_real_paths.extend(find_images(str(subdir), recursive=True))
                    elif subdir.name in fake_names:
                        method_fake_paths.extend(find_images(str(subdir), recursive=True))
        else:
            # 结构2: method/scene/0_real, method/scene/1_fake
            for scene_dir in method_dir.iterdir():
                if scene_dir.is_dir():
                    for subdir in scene_dir.iterdir():
                        if subdir.is_dir():
                            if subdir.name in real_names:
                                method_real_paths.extend(find_images(str(subdir), recursive=True))
                            elif subdir.name in fake_names:
                                method_fake_paths.extend(find_images(str(subdir), recursive=True))
        
        # 限制每个方法的样本数
        if max_samples_per_method:
            if len(method_real_paths) > max_samples_per_method:
                np.random.seed(42)
                method_real_paths = list(np.random.choice(method_real_paths, max_samples_per_method, replace=False))
            if len(method_fake_paths) > max_samples_per_method:
                np.random.seed(42)
                method_fake_paths = list(np.random.choice(method_fake_paths, max_samples_per_method, replace=False))
        
        # 添加到总列表
        if method_real_paths or method_fake_paths:
            image_paths.extend(method_real_paths)
            labels.extend([0] * len(method_real_paths))
            methods.extend([method_name] * len(method_real_paths))
            
            image_paths.extend(method_fake_paths)
            labels.extend([1] * len(method_fake_paths))
            methods.extend([method_name] * len(method_fake_paths))
            
            print(f"[{method_name}] Real: {len(method_real_paths)}, Fake: {len(method_fake_paths)}")
    
    print(f"\n总计: {len(image_paths)} 张图片 ({(np.array(labels)==0).sum()} real, {(np.array(labels)==1).sum()} fake)")
    
    return image_paths, labels, methods


def load_calibration_set(
    calibration_dir: str,
    n_calibration: int = 1000,
) -> List[str]:
    """
    加载校准集 (仅 Real 图片)
    
    支持多种方式:
    1. 直接指定一个包含 real 图片的目录 (如 .../0_real)
    2. 指定与 test_dir 相同结构的目录，自动收集所有方法的 0_real
    """
    calibration_path = Path(calibration_dir)
    real_paths = []
    
    real_names = {'0_real', 'real', '0', 'Real', '0_Real'}
    
    # 首先尝试直接在当前目录找图片
    direct_images = find_images(str(calibration_path), recursive=False)
    if direct_images:
        real_paths = direct_images
        print(f"直接从 {calibration_dir} 加载图片")
    else:
        # 检查子目录
        subdirs = [d for d in calibration_path.iterdir() if d.is_dir()]
        
        if not subdirs:
            # 没有子目录也没有图片，尝试递归搜索
            real_paths = find_images(str(calibration_path), recursive=True)
        else:
            subdir_names = {d.name for d in subdirs}
            
            # 如果子目录不是 real/fake 命名，说明是方法目录结构
            if not (subdir_names & real_names):
                # 遍历每个方法目录收集 real 图片
                for method_dir in subdirs:
                    method_subdirs_list = list(method_dir.iterdir())
                    method_subdirs = [d.name for d in method_subdirs_list if d.is_dir()]
                    
                    if set(method_subdirs) & real_names:
                        # 结构1: method/0_real
                        for subdir in method_subdirs_list:
                            if subdir.is_dir() and subdir.name in real_names:
                                real_paths.extend(find_images(str(subdir), recursive=True))
                    else:
                        # 结构2: method/scene/0_real
                        for scene_dir in method_subdirs_list:
                            if scene_dir.is_dir():
                                for subdir in scene_dir.iterdir():
                                    if subdir.is_dir() and subdir.name in real_names:
                                        real_paths.extend(find_images(str(subdir), recursive=True))
            else:
                # 子目录是 real/fake，找 real 目录
                for subdir in subdirs:
                    if subdir.name in real_names:
                        real_paths.extend(find_images(str(subdir), recursive=True))
    
    if len(real_paths) == 0:
        # 最后尝试递归搜索
        real_paths = find_images(str(calibration_path), recursive=True)
    
    if len(real_paths) == 0:
        raise ValueError(f"在 {calibration_dir} 中找不到图片")
    
    print(f"找到 {len(real_paths)} 张图片")
    
    if len(real_paths) < n_calibration:
        print(f"警告: 校准集只有 {len(real_paths)} 张图片")
        return real_paths
    
    np.random.seed(42)
    selected = np.random.choice(real_paths, size=n_calibration, replace=False)
    
    print(f"校准集: {len(selected)} 张 Real 图片 (从 {len(real_paths)} 张中选取)")
    return list(selected)


# ==================== 评估计算 ====================

def compute_scores(
    image_paths: List[str],
    criterion_name: str = "clip",
    batch_size: int = 4,
    num_noise: int = 64,
    device: str = "cuda",
    output_dir: str = "results"
) -> List[Dict]:
    """计算所有图片的 criterion 分数"""
    from config import EvalConfig
    from evaluator import DiffusionEvaluator
    
    config = EvalConfig(
        criterion_name=criterion_name,
        device=device,
        batch_size=batch_size,
        num_noise=num_noise,
        output_dir=output_dir,
        return_terms=True,
    )
    
    evaluator = DiffusionEvaluator(config)
    results = evaluator.evaluate_images(image_paths)
    evaluator.cleanup()
    
    return results


# ==================== 评估指标 ====================

def compute_metrics_for_subset(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    higher_is_fake: bool
) -> Dict:
    """计算一个子集的评估指标"""
    if len(scores) == 0 or len(np.unique(labels)) < 2:
        return None
    
    if higher_is_fake:
        predictions = (scores >= threshold).astype(int)
        scores_for_roc = scores
    else:
        predictions = (scores <= threshold).astype(int)
        scores_for_roc = -scores
    
    try:
        auc = roc_auc_score(labels, scores_for_roc)
        ap = average_precision_score(labels, scores_for_roc)
    except:
        auc = 0.5
        ap = 0.5
    
    return {
        "n_samples": len(scores),
        "n_real": int((labels == 0).sum()),
        "n_fake": int((labels == 1).sum()),
        "auc": float(auc),
        "ap": float(ap),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "real_mean": float(scores[labels == 0].mean()) if (labels == 0).sum() > 0 else None,
        "fake_mean": float(scores[labels == 1].mean()) if (labels == 1).sum() > 0 else None,
    }


def evaluate_per_method(
    scores: np.ndarray,
    labels: np.ndarray,
    methods: np.ndarray,
    threshold: float,
    higher_is_fake: bool
) -> Dict[str, Dict]:
    """按生成方法分别评估（每个方法有自己的 real 和 fake）"""
    results = {}
    
    # 获取所有方法
    unique_methods = np.unique(methods)
    
    for method in unique_methods:
        # 当前方法的所有样本（包括 real 和 fake）
        method_idx = methods == method
        method_scores = scores[method_idx]
        method_labels = labels[method_idx]
        
        metrics = compute_metrics_for_subset(
            method_scores, method_labels, threshold, higher_is_fake
        )
        
        if metrics:
            results[method] = metrics
    
    return results


# ==================== 主实验函数 ====================

def run_zeroshot_experiment_multimethod(
    calibration_dir: str,
    test_dir: str,
    criterion_name: str = "clip",
    batch_size: int = 4,
    num_noise: int = 64,
    device: str = "cuda",
    output_dir: str = "results/zeroshot",
    n_calibration: int = 1000,
    max_samples_per_method: int = None,
):
    """运行 Zero-shot 实验，按方法分别输出结果"""
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 70)
    print("Zero-shot 分类实验 (多方法评估)")
    print("=" * 70)
    print(f"方法: {criterion_name}")
    print(f"num_noise: {num_noise}")
    print()
    
    # 1. 加载校准集
    print("[1/4] 加载校准集...")
    calibration_paths = load_calibration_set(calibration_dir, n_calibration)
    print()
    
    # 2. 加载测试集（带方法信息）
    print("[2/4] 加载测试集...")
    test_paths, test_labels, test_methods = load_dataset_with_methods(
        test_dir, max_samples_per_method
    )
    test_labels = np.array(test_labels)
    test_methods = np.array(test_methods)
    print()
    
    # 3. 计算分数
    print("[3/4] 计算校准集分数...")
    cal_results = compute_scores(
        calibration_paths, criterion_name, batch_size, num_noise, device, output_dir
    )
    calibration_scores = np.array([r['criterion'] for r in cal_results])
    print()
    
    print("[4/4] 计算测试集分数...")
    test_results = compute_scores(
        test_paths, criterion_name, batch_size, num_noise, device, output_dir
    )
    test_scores = np.array([r['criterion'] for r in test_results])
    print()
    
    # 4. 校准阈值
    cal_mean = calibration_scores.mean()
    cal_std = calibration_scores.std()
    
    # 确定方向
    real_test_mean = test_scores[test_labels == 0].mean()
    fake_test_mean = test_scores[test_labels == 1].mean()
    higher_is_fake = fake_test_mean > real_test_mean
    
    if higher_is_fake:
        threshold = cal_mean + cal_std
    else:
        threshold = cal_mean - cal_std
    
    # 5. 总体评估
    overall_metrics = compute_metrics_for_subset(
        test_scores, test_labels, threshold, higher_is_fake
    )
    
    # 6. 按方法评估
    per_method_metrics = evaluate_per_method(
        test_scores, test_labels, test_methods, threshold, higher_is_fake
    )
    
    # 7. 生成报告
    print()
    print("=" * 70)
    print(f"实验结果 - {criterion_name}")
    print("=" * 70)
    print()
    print("【校准设置】")
    print(f"  校准集大小: {len(calibration_scores)}")
    print(f"  校准集 Mean: {cal_mean:.6f}")
    print(f"  校准集 Std:  {cal_std:.6f}")
    print(f"  方向: {'higher_is_fake' if higher_is_fake else 'lower_is_fake'}")
    print(f"  阈值: {threshold:.6f}")
    print()
    
    print("【总体结果】")
    print(f"  AUC:      {overall_metrics['auc']:.4f}")
    print(f"  AP:       {overall_metrics['ap']:.4f}")
    print(f"  Accuracy: {overall_metrics['accuracy']:.4f}")
    print(f"  F1:       {overall_metrics['f1']:.4f}")
    print()
    
    print("【按生成方法的结果】")
    print("-" * 70)
    print(f"{'Method':<20} {'N_Fake':>8} {'AUC':>8} {'AP':>8} {'Acc':>8} {'F1':>8}")
    print("-" * 70)
    
    # 按 AUC 排序
    sorted_methods = sorted(per_method_metrics.items(), key=lambda x: x[1]['auc'], reverse=True)
    
    for method, metrics in sorted_methods:
        print(f"{method:<20} {metrics['n_fake']:>8} {metrics['auc']:>8.4f} {metrics['ap']:>8.4f} {metrics['accuracy']:>8.4f} {metrics['f1']:>8.4f}")
    
    print("-" * 70)
    
    # 计算平均值
    avg_auc = np.mean([m['auc'] for m in per_method_metrics.values()])
    avg_ap = np.mean([m['ap'] for m in per_method_metrics.values()])
    avg_acc = np.mean([m['accuracy'] for m in per_method_metrics.values()])
    avg_f1 = np.mean([m['f1'] for m in per_method_metrics.values()])
    
    print(f"{'Average':<20} {'':<8} {avg_auc:>8.4f} {avg_ap:>8.4f} {avg_acc:>8.4f} {avg_f1:>8.4f}")
    print("=" * 70)
    
    # 8. 保存结果
    results_to_save = {
        "criterion": criterion_name,
        "timestamp": timestamp,
        "num_noise": num_noise,
        "calibration": {
            "n_samples": len(calibration_scores),
            "mean": float(cal_mean),
            "std": float(cal_std),
            "threshold": float(threshold),
            "higher_is_fake": higher_is_fake,
        },
        "overall": overall_metrics,
        "per_method": per_method_metrics,
        "average": {
            "auc": float(avg_auc),
            "ap": float(avg_ap),
            "accuracy": float(avg_acc),
            "f1": float(avg_f1),
        }
    }
    
    # 保存 JSON
    json_path = os.path.join(output_dir, f'results_{criterion_name}_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    print(f"\n结果已保存到: {json_path}")
    
    # 保存 CSV（方便查看）
    csv_data = []
    for method, metrics in sorted_methods:
        csv_data.append({
            "method": method,
            "n_fake": metrics['n_fake'],
            "auc": metrics['auc'],
            "ap": metrics['ap'],
            "accuracy": metrics['accuracy'],
            "precision": metrics['precision'],
            "recall": metrics['recall'],
            "f1": metrics['f1'],
            "fake_mean": metrics['fake_mean'],
        })
    
    df = pd.DataFrame(csv_data)
    csv_path = os.path.join(output_dir, f'per_method_{criterion_name}_{timestamp}.csv')
    df.to_csv(csv_path, index=False)
    print(f"CSV 已保存到: {csv_path}")
    
    return results_to_save


# ==================== 命令行接口 ====================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Real vs Fake 分类实验 (多方法评估)",
    )
    
    parser.add_argument("--mode", type=str, default="zero-shot",
                       choices=["zero-shot"])
    
    # 目录
    parser.add_argument("--calibration-dir", type=str, required=True,
                       help="校准集目录 (仅 Real 图片)")
    parser.add_argument("--test-dir", type=str, required=True,
                       help="测试集目录")
    parser.add_argument("--n-calibration", type=int, default=1000)
    parser.add_argument("--max-samples-per-method", type=int, default=None,
                       help="每个方法最多使用的样本数")
    
    # 评估参数
    parser.add_argument("--criterion", type=str, default="clip",
                       choices=["clip", "clip_v2", "pde", "clip_with_dsir", 
                               "clip_enhanced", "clip_high_order", "latent", 
                               "pde_criterion", "fpde", "score_laplacian"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-noise", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="results")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    run_zeroshot_experiment_multimethod(
        calibration_dir=args.calibration_dir,
        test_dir=args.test_dir,
        criterion_name=args.criterion,
        batch_size=args.batch_size,
        num_noise=args.num_noise,
        device=args.device,
        output_dir=args.output_dir,
        n_calibration=args.n_calibration,
        max_samples_per_method=args.max_samples_per_method,
    )


if __name__ == "__main__":
    main()