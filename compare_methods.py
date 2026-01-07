#!/usr/bin/env python
"""
多方法对比实验 - 同时比较 CLIP 和 PDE 方法

在同一数据集上运行多种方法，生成对比报告和可视化

用法:
    python compare_methods.py --dataset-dir ./dataset --output-dir results/comparison
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from classification_experiment import (
    load_dataset, compute_scores, compute_metrics, 
    classify_by_threshold, classify_by_model
)


def run_comparison(
    dataset_dir: str = None,
    real_dir: str = None,
    fake_dir: str = None,
    methods: list = None,
    batch_size: int = 4,
    num_noise: int = 8,
    device: str = "cuda",
    output_dir: str = "results/comparison",
    max_samples: int = None,
):
    """
    运行多方法对比实验
    """
    if methods is None:
        methods = ["clip", "pde"]
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 60)
    print("多方法对比实验")
    print("=" * 60)
    print(f"方法: {methods}")
    print(f"输出目录: {output_dir}")
    print()
    
    # 1. 加载数据
    print("[1/3] 加载数据集...")
    image_paths, labels = load_dataset(
        real_dir=real_dir,
        fake_dir=fake_dir,
        dataset_dir=dataset_dir,
        max_samples=max_samples
    )
    labels = np.array(labels)
    print(f"总计: {len(image_paths)} 张图片")
    print()
    
    # 2. 对每种方法计算分数
    all_results = {}
    all_metrics = {}
    
    for method in methods:
        print(f"[2/3] 计算 {method} 分数...")
        print("-" * 40)
        
        results = compute_scores(
            image_paths=image_paths,
            criterion_name=method,
            batch_size=batch_size,
            num_noise=num_noise,
            device=device,
            output_dir=os.path.join(output_dir, method)
        )
        
        df = pd.DataFrame(results)
        df['label'] = labels
        df['method'] = method
        
        scores = df['criterion'].values
        metrics = compute_metrics(scores, labels)
        
        all_results[method] = {
            'df': df,
            'scores': scores,
        }
        all_metrics[method] = metrics
        
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  Accuracy: {metrics['accuracy_at_best']:.4f}")
        print()
    
    # 3. 生成对比报告和可视化
    print("[3/3] 生成对比报告...")
    
    # 合并所有结果
    combined_df = pd.concat([r['df'] for r in all_results.values()], ignore_index=True)
    combined_df.to_csv(os.path.join(output_dir, f'all_scores_{timestamp}.csv'), index=False)
    
    # 对比表格
    comparison_data = []
    for method, metrics in all_metrics.items():
        comparison_data.append({
            'Method': method.upper(),
            'AUC': metrics['auc'],
            'Accuracy': metrics['accuracy_at_best'],
            'Precision': metrics['precision_at_best'],
            'Recall': metrics['recall_at_best'],
            'F1': metrics['f1_at_best'],
            'Real Mean': metrics['real_mean'],
            'Real Std': metrics['real_std'],
            'Fake Mean': metrics['fake_mean'],
            'Fake Std': metrics['fake_std'],
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(os.path.join(output_dir, f'comparison_{timestamp}.csv'), index=False)
    
    # 可视化对比
    plot_comparison(all_results, all_metrics, output_dir, methods)
    
    # 打印对比结果
    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    print("=" * 60)
    
    # 保存 JSON
    metrics_json = {}
    for method, metrics in all_metrics.items():
        metrics_json[method] = {
            'auc': metrics['auc'],
            'accuracy': metrics['accuracy_at_best'],
            'precision': metrics['precision_at_best'],
            'recall': metrics['recall_at_best'],
            'f1': metrics['f1_at_best'],
        }
    
    with open(os.path.join(output_dir, f'comparison_{timestamp}.json'), 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    return {
        'all_results': all_results,
        'all_metrics': all_metrics,
        'comparison_df': comparison_df,
    }


def plot_comparison(all_results, all_metrics, output_dir, methods):
    """生成对比可视化"""
    
    # 1. AUC 对比条形图
    fig, ax = plt.subplots(figsize=(8, 6))
    aucs = [all_metrics[m]['auc'] for m in methods]
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    bars = ax.bar(methods, aucs, color=colors, edgecolor='black', alpha=0.8)
    ax.set_ylabel('AUC')
    ax.set_title('AUC Comparison')
    ax.set_ylim([0.5, 1.0])
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{auc:.4f}', ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'auc_comparison.png'), dpi=150)
    plt.close()
    
    # 2. ROC 曲线对比
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))
    for i, method in enumerate(methods):
        metrics = all_metrics[method]
        ax.plot(metrics['fpr'], metrics['tpr'], color=colors[i], lw=2,
                label=f'{method.upper()} (AUC={metrics["auc"]:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve Comparison')
    ax.legend(loc="lower right")
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_comparison.png'), dpi=150)
    plt.close()
    
    # 3. 分布对比 (violin plot)
    fig, axes = plt.subplots(1, len(methods), figsize=(5*len(methods), 6))
    if len(methods) == 1:
        axes = [axes]
    
    for i, method in enumerate(methods):
        ax = axes[i]
        df = all_results[method]['df']
        
        real_scores = df[df['label'] == 0]['criterion']
        fake_scores = df[df['label'] == 1]['criterion']
        
        parts = ax.violinplot([real_scores, fake_scores], positions=[1, 2], showmeans=True)
        parts['bodies'][0].set_facecolor('#2ecc71')
        parts['bodies'][1].set_facecolor('#e74c3c')
        for pc in parts['bodies']:
            pc.set_alpha(0.7)
        
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Real', 'Fake'])
        ax.set_title(f'{method.upper()}\nAUC={all_metrics[method]["auc"]:.4f}')
        ax.set_ylabel('Score')
    
    plt.suptitle('Score Distribution Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. 指标雷达图
    categories = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1']
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))
    
    for i, method in enumerate(methods):
        metrics = all_metrics[method]
        values = [
            metrics['auc'],
            metrics['accuracy_at_best'],
            metrics['precision_at_best'],
            metrics['recall_at_best'],
            metrics['f1_at_best'],
        ]
        values += values[:1]  # 闭合
        
        ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], label=method.upper())
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim([0, 1])
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Metrics Comparison', y=1.08)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'radar_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"对比图表已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="多方法对比实验")
    
    parser.add_argument("--dataset-dir", type=str, help="数据集目录")
    parser.add_argument("--real-dir", type=str, help="Real 图片目录")
    parser.add_argument("--fake-dir", type=str, help="Fake 图片目录")
    parser.add_argument("--methods", type=str, nargs='+', default=["clip", "pde"],
                       help="要对比的方法")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-noise", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="results/comparison")
    parser.add_argument("--max-samples", type=int, help="每类最大样本数")
    
    args = parser.parse_args()
    
    if not args.dataset_dir and not (args.real_dir and args.fake_dir):
        print("错误: 请指定 --dataset-dir 或同时指定 --real-dir 和 --fake-dir")
        sys.exit(1)
    
    run_comparison(
        dataset_dir=args.dataset_dir,
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        methods=args.methods,
        batch_size=args.batch_size,
        num_noise=args.num_noise,
        device=args.device,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()