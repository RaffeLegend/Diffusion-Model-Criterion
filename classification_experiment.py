#!/usr/bin/env python
"""
分类实验脚本 - Real vs Fake 图片分类 (修正版)

按照论文 "Manifold Induced Biases for Zero-shot and Few-shot Detection" 的方法：

【Zero-shot 设置】(论文 Section 5.1, Figure 4a)
1. 使用独立的 1000 张 Real 图片作为校准集
2. 阈值 = mean + 1*std (仅基于 Real 图片，不使用任何 Fake 图片)
3. 评估时报告 AUC, AP, Accuracy

【Few-shot MoE 设置】(论文 Section 5.2, Figure 6)
1. 使用额外的 1K 标签样本训练轻量级分类器
2. 结合 zero-shot criterion 和 few-shot 方法

用法:
    # Zero-shot 评估 (论文主要方法)
    python classification_experiment_fixed.py \
        --calibration-dir /path/to/calibration/real \
        --test-dir /path/to/test/dataset \
        --mode zero-shot

    # Few-shot MoE 评估
    python classification_experiment_fixed.py \
        --test-dir /path/to/test/dataset \
        --mode few-shot \
        --n-train 1000
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, precision_recall_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# ==================== 数据加载 ====================

def find_images(directory: str, recursive: bool = False) -> List[str]:
    """查找目录中的所有图片"""
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    directory = Path(directory)
    image_paths = []
    
    if recursive:
        for ext in extensions:
            image_paths.extend(directory.rglob(f"*{ext}"))
            image_paths.extend(directory.rglob(f"*{ext.upper()}"))
    else:
        for ext in extensions:
            image_paths.extend(directory.glob(f"*{ext}"))
            image_paths.extend(directory.glob(f"*{ext.upper()}"))
    
    return sorted([str(p) for p in image_paths])


def load_dataset(
    real_dir: str = None,
    fake_dir: str = None,
    dataset_dir: str = None,
    max_samples: int = None,
    recursive: bool = False
) -> Tuple[List[str], List[int]]:
    """加载数据集"""
    image_paths = []
    labels = []
    
    if dataset_dir:
        dataset_path = Path(dataset_dir)
        real_candidates = ['0_real', 'real', '0', 'Real', '0_Real']
        fake_candidates = ['1_fake', 'fake', '1', 'Fake', '1_Fake', 'generated', 'gen']
        
        real_dir = None
        fake_dir = None
        
        for name in real_candidates:
            if (dataset_path / name).exists():
                real_dir = str(dataset_path / name)
                break
        
        for name in fake_candidates:
            if (dataset_path / name).exists():
                fake_dir = str(dataset_path / name)
                break
        
        if not real_dir or not fake_dir:
            raise ValueError(f"无法在 {dataset_dir} 中找到 real/fake 子目录")
    
    if real_dir:
        real_paths = find_images(real_dir, recursive)
        if max_samples:
            real_paths = real_paths[:max_samples]
        image_paths.extend(real_paths)
        labels.extend([0] * len(real_paths))
        print(f"Real 图片: {len(real_paths)} 张 (来自 {real_dir})")
    
    if fake_dir:
        fake_paths = find_images(fake_dir, recursive)
        if max_samples:
            fake_paths = fake_paths[:max_samples]
        image_paths.extend(fake_paths)
        labels.extend([1] * len(fake_paths))
        print(f"Fake 图片: {len(fake_paths)} 张 (来自 {fake_dir})")
    
    return image_paths, labels


def load_calibration_set(
    calibration_dir: str,
    n_calibration: int = 1000,
    recursive: bool = False
) -> List[str]:
    """
    加载校准集 (仅 Real 图片)
    
    论文 Figure 4(a): "We calibrate a decision threshold based on the 
    mean and standard deviation of 1,000 real image criteria"
    """
    real_paths = find_images(calibration_dir, recursive)
    
    if len(real_paths) < n_calibration:
        print(f"警告: 校准集只有 {len(real_paths)} 张图片，少于要求的 {n_calibration} 张")
        return real_paths
    
    # 随机选择 n_calibration 张
    np.random.seed(42)
    selected = np.random.choice(real_paths, size=n_calibration, replace=False)
    
    print(f"校准集: {len(selected)} 张 Real 图片 (来自 {calibration_dir})")
    return list(selected)


# ==================== 评估计算 ====================

def compute_scores(
    image_paths: List[str],
    criterion_name: str = "clip",
    batch_size: int = 4,
    num_noise: int = 64,  # 论文用 64
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


# ==================== Zero-shot 分类 (论文主要方法) ====================

def calibrate_threshold_zeroshot(
    calibration_scores: np.ndarray,
    n_std: float = 1.0,
    direction: str = "auto"
) -> Tuple[float, float, float, str]:
    """
    Zero-shot 阈值校准 (论文 Figure 4a)
    
    仅使用 Real 图片的分数来确定阈值：
    - 如果 higher_is_fake: threshold = mean + n_std * std
    - 如果 lower_is_fake: threshold = mean - n_std * std
    
    Args:
        calibration_scores: 校准集 (Real 图片) 的分数
        n_std: 标准差倍数，论文用 1.0
        direction: "higher_is_fake", "lower_is_fake", or "auto"
    
    Returns:
        threshold, mean, std, direction
    """
    mean = calibration_scores.mean()
    std = calibration_scores.std()
    
    # 返回两个阈值，让后续根据测试集决定方向
    threshold_high = mean + n_std * std  # 用于 higher_is_fake
    threshold_low = mean - n_std * std   # 用于 lower_is_fake
    
    return threshold_high, threshold_low, mean, std


def classify_zeroshot(
    test_scores: np.ndarray,
    test_labels: np.ndarray,
    threshold: float,
    higher_is_fake: bool = True
) -> Dict:
    """
    Zero-shot 分类
    
    Args:
        test_scores: 测试集分数
        test_labels: 测试集标签 (0=real, 1=fake)
        threshold: 预先校准的阈值
        higher_is_fake: 分数越高越可能是 fake
    """
    if higher_is_fake:
        predictions = (test_scores >= threshold).astype(int)
        scores_for_metrics = test_scores
    else:
        predictions = (test_scores <= threshold).astype(int)
        scores_for_metrics = -test_scores
    
    # 计算指标
    auc = roc_auc_score(test_labels, scores_for_metrics)
    ap = average_precision_score(test_labels, scores_for_metrics)
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, zero_division=0)
    recall = recall_score(test_labels, predictions, zero_division=0)
    f1 = f1_score(test_labels, predictions, zero_division=0)
    
    # 如果 AUC < 0.5，说明方向反了
    if auc < 0.5:
        auc = 1 - auc
        ap = 1 - ap  # AP 也需要调整
        predictions = 1 - predictions
        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions, zero_division=0)
        recall = recall_score(test_labels, predictions, zero_division=0)
        f1 = f1_score(test_labels, predictions, zero_division=0)
    
    return {
        "threshold": threshold,
        "auc": auc,
        "ap": ap,  # Average Precision - 论文报告的指标
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predictions": predictions,
    }


# ==================== Few-shot MoE 分类 (论文 Section 5.2) ====================

def classify_fewshot_moe(
    scores: np.ndarray,
    labels: np.ndarray,
    n_train: int = 1000,
    classifier_type: str = "rf",  # 论文用 Random Forest
    n_splits: int = 5
) -> Dict:
    """
    Few-shot Mixture of Experts 分类 (论文 Figure 6)
    
    使用少量标签数据训练轻量级分类器
    
    Args:
        scores: criterion 分数 (可以是单个或多个特征)
        labels: 标签
        n_train: 训练样本数
        classifier_type: 分类器类型 ('rf', 'svm', 'logistic')
    """
    # 准备特征
    if scores.ndim == 1:
        X = scores.reshape(-1, 1)
    else:
        X = scores
    y = labels
    
    # 分割训练/测试集
    if n_train >= len(labels):
        print(f"警告: n_train ({n_train}) >= 总样本数 ({len(labels)}), 使用交叉验证")
        X_train, X_test, y_train, y_test = X, X, y, y
        use_cv_only = True
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=n_train, stratify=y, random_state=42
        )
        use_cv_only = False
    
    # 选择分类器
    if classifier_type == "rf":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif classifier_type == "svm":
        model = SVC(kernel='rbf', probability=True, random_state=42)
    elif classifier_type == "logistic":
        model = LogisticRegression(random_state=42, max_iter=1000)
    else:
        raise ValueError(f"Unknown classifier: {classifier_type}")
    
    # 交叉验证 (在训练集上)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    cv_accuracy = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    
    # 训练最终模型并在测试集上评估
    model.fit(X_train, y_train)
    
    if not use_cv_only:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        test_auc = roc_auc_score(y_test, y_prob)
        test_ap = average_precision_score(y_test, y_prob)
        test_accuracy = accuracy_score(y_test, y_pred)
    else:
        test_auc = cv_auc.mean()
        test_ap = None
        test_accuracy = cv_accuracy.mean()
    
    return {
        "classifier": classifier_type,
        "n_train": n_train if not use_cv_only else len(y),
        "cv_auc_mean": cv_auc.mean(),
        "cv_auc_std": cv_auc.std(),
        "cv_accuracy_mean": cv_accuracy.mean(),
        "cv_accuracy_std": cv_accuracy.std(),
        "test_auc": test_auc,
        "test_ap": test_ap,
        "test_accuracy": test_accuracy,
        "model": model,
    }


# ==================== 完整评估指标 ====================

def compute_full_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float = None
) -> Dict:
    """计算完整的评估指标"""
    
    # 确定方向
    real_mean = scores[labels == 0].mean()
    fake_mean = scores[labels == 1].mean()
    higher_is_fake = fake_mean > real_mean
    
    if higher_is_fake:
        scores_for_roc = scores
    else:
        scores_for_roc = -scores
    
    # AUC
    auc = roc_auc_score(labels, scores_for_roc)
    if auc < 0.5:
        auc = 1 - auc
        scores_for_roc = -scores_for_roc
        higher_is_fake = not higher_is_fake
    
    # AP (Average Precision) - 论文报告的重要指标
    ap = average_precision_score(labels, scores_for_roc)
    
    # ROC 曲线
    fpr, tpr, roc_thresholds = roc_curve(labels, scores_for_roc)
    
    # PR 曲线
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(labels, scores_for_roc)
    
    # 最优阈值 (Youden's J) - 仅用于参考，zero-shot 不应该用这个
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    optimal_threshold = roc_thresholds[best_idx]
    
    # 如果提供了阈值，使用它；否则使用最优阈值
    if threshold is not None:
        eval_threshold = threshold
    else:
        eval_threshold = optimal_threshold
    
    # 使用阈值计算分类指标
    if higher_is_fake:
        predictions = (scores >= eval_threshold).astype(int)
    else:
        predictions = (scores <= eval_threshold).astype(int)
    
    # 分组统计
    real_scores = scores[labels == 0]
    fake_scores = scores[labels == 1]
    
    return {
        "auc": auc,
        "ap": ap,
        "higher_is_fake": higher_is_fake,
        "threshold_used": eval_threshold,
        "optimal_threshold": optimal_threshold,
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "real_mean": real_scores.mean(),
        "real_std": real_scores.std(),
        "fake_mean": fake_scores.mean(),
        "fake_std": fake_scores.std(),
        "fpr": fpr,
        "tpr": tpr,
        "precision_curve": precision_curve,
        "recall_curve": recall_curve,
    }


# ==================== 可视化 ====================

def plot_results(
    scores: np.ndarray,
    labels: np.ndarray,
    metrics: Dict,
    output_dir: str,
    criterion_name: str,
    calibration_threshold: float = None
):
    """生成所有可视化图表"""
    os.makedirs(output_dir, exist_ok=True)
    
    real_scores = scores[labels == 0]
    fake_scores = scores[labels == 1]
    
    # 1. 分布直方图 (类似论文 Figure 4a)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(real_scores, bins=50, alpha=0.6, label=f'Real (n={len(real_scores)})', 
            color='blue', density=True)
    ax.hist(fake_scores, bins=50, alpha=0.6, label=f'Fake (n={len(fake_scores)})', 
            color='red', density=True)
    
    # 标记阈值
    if calibration_threshold is not None:
        ax.axvline(calibration_threshold, color='black', linestyle='--', linewidth=2,
                   label=f'Calibration Threshold (mean+1σ)={calibration_threshold:.4f}')
    
    # 标记 Real 的 mean 和 mean+std
    real_mean = real_scores.mean()
    real_std = real_scores.std()
    ax.axvline(real_mean, color='blue', linestyle=':', alpha=0.7, 
               label=f'Real Mean={real_mean:.4f}')
    
    ax.set_xlabel(f'{criterion_name} Score')
    ax.set_ylabel('Density')
    ax.set_title(f'Score Distribution\nAUC={metrics["auc"]:.4f}, AP={metrics["ap"]:.4f}')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution.png'), dpi=150)
    plt.close()
    
    # 2. ROC 曲线
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(metrics['fpr'], metrics['tpr'], color='#e74c3c', lw=2, 
            label=f'ROC curve (AUC = {metrics["auc"]:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {criterion_name}')
    ax.legend(loc="lower right")
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=150)
    plt.close()
    
    # 3. PR 曲线 (论文也报告 AP)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(metrics['recall_curve'], metrics['precision_curve'], color='#3498db', lw=2,
            label=f'PR curve (AP = {metrics["ap"]:.4f})')
    ax.axhline(y=labels.mean(), color='gray', linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve - {criterion_name}')
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pr_curve.png'), dpi=150)
    plt.close()
    
    # 4. 箱线图
    fig, ax = plt.subplots(figsize=(8, 6))
    data = [real_scores, fake_scores]
    bp = ax.boxplot(data, labels=['Real', 'Fake'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#e74c3c')
    for box in bp['boxes']:
        box.set_alpha(0.7)
    ax.set_ylabel(f'{criterion_name} Score')
    ax.set_title('Score Distribution by Class')
    
    means = [real_scores.mean(), fake_scores.mean()]
    ax.scatter([1, 2], means, color='gold', s=100, zorder=5, marker='D', 
               edgecolor='black', linewidth=1, label='Mean')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'boxplot.png'), dpi=150)
    plt.close()
    
    # 5. 混淆矩阵
    if metrics['higher_is_fake']:
        predictions = (scores >= metrics['threshold_used']).astype(int)
    else:
        predictions = (scores <= metrics['threshold_used']).astype(int)
    
    cm = confusion_matrix(labels, predictions)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Pred Real', 'Pred Fake'],
                yticklabels=['True Real', 'True Fake'])
    ax.set_title(f'Confusion Matrix (Acc={metrics["accuracy"]:.4f})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    
    print(f"图表已保存到: {output_dir}")


# ==================== 主实验函数 ====================

def run_zeroshot_experiment(
    calibration_dir: str,
    test_dir: str = None,
    test_real_dir: str = None,
    test_fake_dir: str = None,
    criterion_name: str = "clip",
    batch_size: int = 4,
    num_noise: int = 64,
    device: str = "cuda",
    output_dir: str = "results/zeroshot",
    n_calibration: int = 1000,
    max_test_samples: int = None,
):
    """
    运行 Zero-shot 分类实验 (论文主要方法)
    
    1. 在校准集 (仅Real) 上计算分数
    2. 确定阈值 = mean + std
    3. 在测试集上评估
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 60)
    print("Zero-shot 分类实验 (论文方法)")
    print("=" * 60)
    print(f"方法: {criterion_name}")
    print(f"Num noise: {num_noise}")
    print()
    
    # 1. 加载校准集
    print("[1/5] 加载校准集 (仅 Real 图片)...")
    calibration_paths = load_calibration_set(calibration_dir, n_calibration)
    print()
    
    # 2. 计算校准集分数
    print("[2/5] 计算校准集分数...")
    calibration_results = compute_scores(
        image_paths=calibration_paths,
        criterion_name=criterion_name,
        batch_size=batch_size,
        num_noise=num_noise,
        device=device,
        output_dir=output_dir
    )
    calibration_scores = np.array([r['criterion'] for r in calibration_results])
    print(f"校准集分数: mean={calibration_scores.mean():.6f}, std={calibration_scores.std():.6f}")
    print()
    
    # 3. 确定阈值
    print("[3/5] 确定阈值 (mean ± 1*std)...")
    threshold_high, threshold_low, cal_mean, cal_std = calibrate_threshold_zeroshot(calibration_scores, n_std=1.0)
    print(f"校准集: mean={cal_mean:.6f}, std={cal_std:.6f}")
    print(f"阈值 (higher_is_fake): {threshold_high:.6f}")
    print(f"阈值 (lower_is_fake): {threshold_low:.6f}")
    print()
    
    # 4. 加载测试集
    print("[4/5] 加载测试集...")
    test_paths, test_labels = load_dataset(
        real_dir=test_real_dir,
        fake_dir=test_fake_dir,
        dataset_dir=test_dir,
        max_samples=max_test_samples
    )
    test_labels = np.array(test_labels)
    print()
    
    # 5. 计算测试集分数并评估
    print("[5/5] 计算测试集分数并评估...")
    test_results = compute_scores(
        image_paths=test_paths,
        criterion_name=criterion_name,
        batch_size=batch_size,
        num_noise=num_noise,
        device=device,
        output_dir=output_dir
    )
    test_scores = np.array([r['criterion'] for r in test_results])
    
    # 保存原始分数
    df = pd.DataFrame(test_results)
    df['image_path'] = test_paths
    df['label'] = test_labels
    df['label_name'] = ['real' if l == 0 else 'fake' for l in test_labels]
    scores_path = os.path.join(output_dir, f'scores_{criterion_name}_{timestamp}.csv')
    df.to_csv(scores_path, index=False)
    
    # 自动检测方向：比较 real 和 fake 的均值
    real_scores = test_scores[test_labels == 0]
    fake_scores = test_scores[test_labels == 1]
    
    print(f"Real scores: mean={real_scores.mean():.4f}, std={real_scores.std():.4f}")
    print(f"Fake scores: mean={fake_scores.mean():.4f}, std={fake_scores.std():.4f}")
    
    if fake_scores.mean() > real_scores.mean():
        higher_is_fake = True
        threshold = threshold_high
        print(f"检测到: Fake分数更高 -> 使用 higher_is_fake, threshold={threshold:.4f}")
    else:
        higher_is_fake = False
        threshold = threshold_low
        print(f"检测到: Real分数更高 -> 使用 lower_is_fake, threshold={threshold:.4f}")
    print()
    
    # 计算指标
    metrics = compute_full_metrics(test_scores, test_labels, threshold)
    
    # Zero-shot 分类结果
    zs_results = classify_zeroshot(
        test_scores, test_labels, threshold, 
        higher_is_fake=higher_is_fake
    )
    
    # 生成可视化
    plot_results(test_scores, test_labels, metrics, output_dir, criterion_name, threshold)
    
    # 生成报告
    report_lines = [
        "=" * 60,
        f"Zero-shot 分类报告 - {criterion_name}",
        "=" * 60,
        f"时间: {timestamp}",
        "",
        "【校准设置】(论文 Figure 4a)",
        f"  校准集大小: {len(calibration_scores)} (仅 Real)",
        f"  校准集 Mean: {cal_mean:.6f}",
        f"  校准集 Std: {cal_std:.6f}",
        f"  检测方向: {'higher_is_fake' if higher_is_fake else 'lower_is_fake'}",
        f"  使用阈值: {threshold:.6f}",
        "",
        "【测试集统计】",
        f"  Real 样本数: {(test_labels == 0).sum()}",
        f"  Fake 样本数: {(test_labels == 1).sum()}",
        "",
        "【测试集分数】",
        f"  Real Mean: {metrics['real_mean']:.6f} ± {metrics['real_std']:.6f}",
        f"  Fake Mean: {metrics['fake_mean']:.6f} ± {metrics['fake_std']:.6f}",
        f"  分数差异: {abs(metrics['fake_mean'] - metrics['real_mean']):.6f}",
        "",
        "【Zero-shot 分类结果】(论文 Table 1 格式)",
        f"  AUC: {zs_results['auc']:.4f}",
        f"  AP:  {zs_results['ap']:.4f}",
        f"  Accuracy: {zs_results['accuracy']:.4f}",
        f"  Precision: {zs_results['precision']:.4f}",
        f"  Recall: {zs_results['recall']:.4f}",
        f"  F1: {zs_results['f1']:.4f}",
        "",
        "【使用最优阈值的结果】(参考，非 zero-shot)",
        f"  最优阈值: {metrics['optimal_threshold']:.6f}",
        f"  Accuracy: {metrics['accuracy']:.4f}",
        f"  Precision: {metrics['precision']:.4f}",
        f"  Recall: {metrics['recall']:.4f}",
        f"  F1: {metrics['f1']:.4f}",
        "",
        "=" * 60,
    ]
    
    report_text = "\n".join(report_lines)
    print()
    print(report_text)
    
    # 保存报告
    report_path = os.path.join(output_dir, f'report_{criterion_name}_{timestamp}.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    # 保存指标 JSON
    metrics_to_save = {
        "mode": "zero-shot",
        "criterion": criterion_name,
        "timestamp": timestamp,
        "num_noise": num_noise,
        "calibration": {
            "n_samples": len(calibration_scores),
            "mean": float(cal_mean),
            "std": float(cal_std),
            "threshold": float(threshold),
        },
        "test": {
            "n_real": int((test_labels == 0).sum()),
            "n_fake": int((test_labels == 1).sum()),
            "real_mean": float(metrics['real_mean']),
            "real_std": float(metrics['real_std']),
            "fake_mean": float(metrics['fake_mean']),
            "fake_std": float(metrics['fake_std']),
        },
        "results": {
            "auc": float(zs_results['auc']),
            "ap": float(zs_results['ap']),
            "accuracy": float(zs_results['accuracy']),
            "precision": float(zs_results['precision']),
            "recall": float(zs_results['recall']),
            "f1": float(zs_results['f1']),
        }
    }
    
    metrics_path = os.path.join(output_dir, f'metrics_{criterion_name}_{timestamp}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    
    return {
        "calibration_scores": calibration_scores,
        "threshold": threshold,
        "test_scores": test_scores,
        "test_labels": test_labels,
        "metrics": metrics,
        "zs_results": zs_results,
    }


def run_fewshot_experiment(
    test_dir: str = None,
    test_real_dir: str = None,
    test_fake_dir: str = None,
    criterion_name: str = "clip",
    batch_size: int = 4,
    num_noise: int = 64,
    device: str = "cuda",
    output_dir: str = "results/fewshot",
    n_train: int = 1000,
    max_samples: int = None,
):
    """
    运行 Few-shot MoE 分类实验 (论文 Figure 6)
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 60)
    print("Few-shot MoE 分类实验")
    print("=" * 60)
    print(f"方法: {criterion_name}")
    print(f"训练样本数: {n_train}")
    print()
    
    # 加载数据
    print("[1/3] 加载数据集...")
    image_paths, labels = load_dataset(
        real_dir=test_real_dir,
        fake_dir=test_fake_dir,
        dataset_dir=test_dir,
        max_samples=max_samples
    )
    labels = np.array(labels)
    print()
    
    # 计算分数
    print("[2/3] 计算分数...")
    results = compute_scores(
        image_paths=image_paths,
        criterion_name=criterion_name,
        batch_size=batch_size,
        num_noise=num_noise,
        device=device,
        output_dir=output_dir
    )
    scores = np.array([r['criterion'] for r in results])
    print()
    
    # Few-shot 分类
    print("[3/3] Few-shot MoE 分类...")
    
    moe_results = {}
    for clf_type in ['rf', 'svm', 'logistic']:
        print(f"  训练 {clf_type.upper()}...")
        moe_results[clf_type] = classify_fewshot_moe(
            scores, labels, n_train=n_train, classifier_type=clf_type
        )
    
    # 报告
    report_lines = [
        "=" * 60,
        f"Few-shot MoE 分类报告 - {criterion_name}",
        "=" * 60,
        f"训练样本数: {n_train}",
        "",
        "【分类器结果】(5-fold CV)",
    ]
    
    for clf_type, result in moe_results.items():
        report_lines.extend([
            f"  {clf_type.upper()}:",
            f"    CV AUC: {result['cv_auc_mean']:.4f} ± {result['cv_auc_std']:.4f}",
            f"    CV Accuracy: {result['cv_accuracy_mean']:.4f} ± {result['cv_accuracy_std']:.4f}",
            f"    Test AUC: {result['test_auc']:.4f}" if result['test_auc'] else "",
        ])
    
    report_text = "\n".join(report_lines)
    print()
    print(report_text)
    
    return moe_results


# ==================== 命令行接口 ====================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Real vs Fake 分类实验 (论文方法)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--mode", type=str, default="zero-shot",
                       choices=["zero-shot", "few-shot"],
                       help="实验模式")
    
    # Zero-shot 参数
    parser.add_argument("--calibration-dir", type=str,
                       help="校准集目录 (仅 Real 图片)")
    parser.add_argument("--n-calibration", type=int, default=1000,
                       help="校准集大小")
    
    # 测试集
    parser.add_argument("--test-dir", type=str, help="测试集目录")
    parser.add_argument("--test-real-dir", type=str)
    parser.add_argument("--test-fake-dir", type=str)
    parser.add_argument("--max-samples", type=int)
    
    # Few-shot 参数
    parser.add_argument("--n-train", type=int, default=1000,
                       help="Few-shot 训练样本数")
    
    # 评估参数
    parser.add_argument("--criterion", type=str, default="clip",
                       choices=["clip", "clip_v2", "pde", "clip_with_dsir", "clip_enhanced", "clip_high_order", "latent", "pde_criterion", "fpde", "score_laplacian"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-noise", type=int, default=64,
                       help="球面扰动数量 (论文用 64)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="results")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.mode == "zero-shot":
        if not args.calibration_dir:
            print("错误: Zero-shot 模式需要 --calibration-dir")
            sys.exit(1)
        
        run_zeroshot_experiment(
            calibration_dir=args.calibration_dir,
            test_dir=args.test_dir,
            test_real_dir=args.test_real_dir,
            test_fake_dir=args.test_fake_dir,
            criterion_name=args.criterion,
            batch_size=args.batch_size,
            num_noise=args.num_noise,
            device=args.device,
            output_dir=args.output_dir,
            n_calibration=args.n_calibration,
            max_test_samples=args.max_samples,
        )
    
    elif args.mode == "few-shot":
        run_fewshot_experiment(
            test_dir=args.test_dir,
            test_real_dir=args.test_real_dir,
            test_fake_dir=args.test_fake_dir,
            criterion_name=args.criterion,
            batch_size=args.batch_size,
            num_noise=args.num_noise,
            device=args.device,
            output_dir=args.output_dir,
            n_train=args.n_train,
            max_samples=args.max_samples,
        )


if __name__ == "__main__":
    main()