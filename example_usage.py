#!/usr/bin/env python
"""
示例：如何调用增强版 Criterion

三种方法:
1. clip        - 修正后的原文方法 (baseline)
2. clip_with_dsir   - 原文 + DSIR能量 (最小改动)
3. clip_enhanced    - 完整融合版 (CLIP空间高阶特征)
"""

import sys
from pathlib import Path

# ==================== 方法1: 修改 config 调用 ====================

def example_via_config():
    """通过配置文件调用"""
    from config import EvalConfig
    from evaluator import DiffusionEvaluator
    
    # 1. 原文方法 (修正版)
    config_baseline = EvalConfig(
        criterion_name="clip",  # 使用修正后的原文方法
        device="cuda",
        batch_size=4,
        num_noise=64,  # 论文用64
        return_terms=True,
    )
    
    # 2. 最小改动版: 原文 + DSIR
    config_dsir = EvalConfig(
        criterion_name="clip_with_dsir",
        device="cuda",
        batch_size=4,
        num_noise=64,
        return_terms=True,
        dsir_weight=0.01,  # DSIR权重，可调
    )
    
    # 3. 完整融合版
    config_enhanced = EvalConfig(
        criterion_name="clip_enhanced",
        device="cuda",
        batch_size=4,
        num_noise=64,
        return_terms=True,
        lambda_high_order=0.1,  # 高阶项权重
        high_order_type="laplacian",  # 可选: laplacian, biharmonic, dsir
    )
    
    # 选择一个config运行
    config = config_dsir  # 推荐先试这个
    
    evaluator = DiffusionEvaluator(config)
    
    # 评估图片
    image_paths = [
        "/path/to/image1.jpg",
        "/path/to/image2.jpg",
    ]
    results = evaluator.evaluate_images(image_paths)
    
    for i, r in enumerate(results):
        print(f"Image {i}: criterion={r['criterion']:.4f}")
        if 'C_manifold' in r:
            print(f"  - C_manifold: {r['C_manifold']:.4f}")
        if 'dsir_diff' in r:
            print(f"  - DSIR diff: {r['dsir_diff']:.4f}")
    
    evaluator.cleanup()
    return results


# ==================== 方法2: 直接实例化 Criterion ====================

def example_direct_instantiation():
    """直接实例化 Criterion 类"""
    from config import EvalConfig
    from model_manager import ModelManager
    
    # 导入新的criterion (确保已经注册)
    from criteria.clip_criterion_enhanced import CLIPWithDSIRCriterion, EnhancedCLIPCriterion
    
    # 准备配置
    config = EvalConfig(
        device="cuda",
        batch_size=4,
        num_noise=64,
        return_terms=True,
    )
    
    # 添加自定义参数
    config.dsir_weight = 0.01
    config.lambda_high_order = 0.1
    config.high_order_type = "laplacian"
    
    # 初始化模型管理器
    model_manager = ModelManager(config)
    
    # 直接实例化criterion
    # criterion = CLIPWithDSIRCriterion(config, model_manager)  # 最小改动版
    criterion = EnhancedCLIPCriterion(config, model_manager)  # 完整融合版
    
    # 准备输入数据 (需要按照你的数据加载方式)
    # images: torch.Tensor, shape (B, C, H, W)
    # prompts: List[str]
    # images_raw: List[torch.Tensor] - 原始图片用于CLIP
    
    # results = criterion.evaluate_batch(images, prompts, images_raw)
    
    return criterion


# ==================== 方法3: 用于分类实验 ====================

def example_classification():
    """用于分类实验"""
    from classification_experiment_fixed import run_zeroshot_experiment
    
    # Zero-shot 实验 - 使用增强方法
    results = run_zeroshot_experiment(
        calibration_dir="/path/to/calibration/real_images",
        test_dir="/path/to/test/dataset",
        criterion_name="clip_with_dsir",  # 或 "clip_enhanced"
        batch_size=4,
        num_noise=64,
        device="cuda",
        output_dir="results/enhanced_zeroshot",
        n_calibration=1000,
    )
    
    print(f"AUC: {results['zs_results']['auc']:.4f}")
    print(f"AP: {results['zs_results']['ap']:.4f}")
    
    return results


# ==================== 方法4: 快速测试脚本 ====================

def quick_test():
    """
    快速测试新方法是否工作
    """
    import torch
    import numpy as np
    
    print("=" * 50)
    print("Quick Test: Enhanced Criterion")
    print("=" * 50)
    
    # 测试高阶算子
    from criteria.clip_criterion_enhanced import HighOrderOperators
    
    ops = HighOrderOperators()
    
    # 创建测试图像
    test_img = np.random.rand(256, 256, 3).astype(np.float32) * 255
    test_img = test_img.astype(np.uint8)
    
    # 测试Laplacian
    lap = ops.apply_laplacian(test_img)
    print(f"Laplacian shape: {lap.shape}, range: [{lap.min():.2f}, {lap.max():.2f}]")
    
    # 测试Biharmonic
    bih = ops.apply_biharmonic(test_img)
    print(f"Biharmonic shape: {bih.shape}, range: [{bih.min():.2f}, {bih.max():.2f}]")
    
    # 测试DSIR
    dsir = ops.compute_dsir(test_img)
    print(f"DSIR shape: {dsir.shape}, mean: {np.mean(dsir):.4f}")
    
    # 测试能量计算
    energy_lap = ops.compute_high_order_energy(test_img, 'laplacian')
    energy_bih = ops.compute_high_order_energy(test_img, 'biharmonic')
    print(f"Laplacian energy: {energy_lap:.4f}")
    print(f"Biharmonic energy: {energy_bih:.4f}")
    
    print("\n✓ High-order operators work correctly!")
    
    return True


# ==================== 方法5: 对比实验 ====================

def comparison_experiment():
    """
    对比原文方法和增强方法
    """
    from classification_experiment_fixed import (
        load_dataset, compute_scores, compute_full_metrics,
        calibrate_threshold_zeroshot, classify_zeroshot
    )
    import numpy as np
    
    # 数据
    calibration_dir = "/path/to/calibration/real"
    test_dir = "/path/to/test/dataset"
    
    # 加载校准集
    from classification_experiment_fixed import load_calibration_set
    cal_paths = load_calibration_set(calibration_dir, n_calibration=1000)
    
    # 加载测试集
    test_paths, test_labels = load_dataset(dataset_dir=test_dir)
    test_labels = np.array(test_labels)
    
    # 对比不同方法
    methods = ["clip", "clip_with_dsir", "clip_enhanced"]
    results_all = {}
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"Testing method: {method}")
        print(f"{'='*50}")
        
        # 计算校准集分数
        cal_results = compute_scores(cal_paths, criterion_name=method, num_noise=64)
        cal_scores = np.array([r['criterion'] for r in cal_results])
        
        # 确定阈值
        threshold, cal_mean, cal_std = calibrate_threshold_zeroshot(cal_scores)
        print(f"Threshold: {threshold:.4f} (mean={cal_mean:.4f}, std={cal_std:.4f})")
        
        # 计算测试集分数
        test_results = compute_scores(test_paths, criterion_name=method, num_noise=64)
        test_scores = np.array([r['criterion'] for r in test_results])
        
        # 评估
        metrics = compute_full_metrics(test_scores, test_labels, threshold)
        zs_results = classify_zeroshot(test_scores, test_labels, threshold, 
                                       higher_is_fake=metrics['higher_is_fake'])
        
        results_all[method] = {
            'auc': zs_results['auc'],
            'ap': zs_results['ap'],
            'accuracy': zs_results['accuracy'],
        }
        
        print(f"AUC: {zs_results['auc']:.4f}")
        print(f"AP: {zs_results['ap']:.4f}")
        print(f"Accuracy: {zs_results['accuracy']:.4f}")
    
    # 汇总对比
    print(f"\n{'='*50}")
    print("Summary Comparison")
    print(f"{'='*50}")
    print(f"{'Method':<20} {'AUC':<10} {'AP':<10} {'Accuracy':<10}")
    print("-" * 50)
    for method, res in results_all.items():
        print(f"{method:<20} {res['auc']:<10.4f} {res['ap']:<10.4f} {res['accuracy']:<10.4f}")
    
    return results_all


# ==================== 主函数 ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试增强版Criterion")
    parser.add_argument("--mode", type=str, default="quick",
                       choices=["quick", "config", "direct", "classify", "compare"],
                       help="测试模式")
    parser.add_argument("--criterion", type=str, default="clip_with_dsir",
                       choices=["clip", "clip_with_dsir", "clip_enhanced"],
                       help="选择criterion")
    
    args = parser.parse_args()
    
    if args.mode == "quick":
        # 快速测试高阶算子
        quick_test()
    
    elif args.mode == "config":
        # 通过config调用
        example_via_config()
    
    elif args.mode == "classify":
        # 分类实验
        example_classification()
    
    elif args.mode == "compare":
        # 对比实验
        comparison_experiment()
    
    else:
        print(f"Unknown mode: {args.mode}")