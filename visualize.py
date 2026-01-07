#!/usr/bin/env python
"""
可视化脚本 - 对比真实图片和生成图片的评估结果

根据路径中的 "0_real" 和 "1_fake" 自动识别图片类别，
生成多种可视化图表进行对比分析。

用法:
    python visualize.py --results results/results.json
    python visualize.py --results results/results.csv --output figures/
    python visualize.py --dir ./images --criterion pde  # 先评估再可视化

    # 方式1: 从已有结果文件可视化
    python visualize.py --results results/results.json --output figures/

    # 方式2: 先评估再可视化
    python visualize.py --dir ./images --criterion pde --output figures/

    # 递归搜索子目录
    python visualize.py --dir ./dataset -r --output figures/
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适合服务器

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')


# ==================== 类别识别 ====================

def extract_category(path: str) -> str:
    """
    从路径中提取类别标签
    
    支持的模式:
    - "0_real" 或 "real" -> "Real"
    - "1_fake" 或 "fake" -> "Fake"
    - 其他 -> "Unknown"
    """
    path_lower = path.lower()
    
    # 检查各种可能的模式
    if "0_real" in path_lower or "/real/" in path_lower or "_real" in path_lower:
        return "Real"
    elif "1_fake" in path_lower or "/fake/" in path_lower or "_fake" in path_lower:
        return "Fake"
    elif "generated" in path_lower or "gen" in path_lower or "synthetic" in path_lower:
        return "Fake"
    else:
        return "Unknown"


def load_results(path: str) -> pd.DataFrame:
    """加载评估结果"""
    path = Path(path)
    
    if path.suffix == '.json':
        with open(path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    elif path.suffix == '.csv':
        df = pd.read_csv(path)
    else:
        raise ValueError(f"不支持的文件格式: {path.suffix}")
    
    # 添加类别列
    if 'category' not in df.columns:
        df['category'] = df['image_path'].apply(extract_category)
    
    return df


# ==================== 可视化函数 ====================

def plot_histogram(df: pd.DataFrame, output_dir: str, metric: str = 'criterion'):
    """绘制直方图对比"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = df['category'].unique()
    colors = {'Real': '#2ecc71', 'Fake': '#e74c3c', 'Unknown': '#95a5a6'}
    
    for cat in categories:
        data = df[df['category'] == cat][metric]
        ax.hist(data, bins=30, alpha=0.6, label=f'{cat} (n={len(data)})', 
                color=colors.get(cat, '#3498db'), edgecolor='white')
    
    ax.set_xlabel(f'{metric.capitalize()} Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Distribution of {metric.capitalize()} Scores by Category', fontsize=14)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'histogram_{metric}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: histogram_{metric}.png")


def plot_boxplot(df: pd.DataFrame, output_dir: str, metric: str = 'criterion'):
    """绘制箱线图对比"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    categories = sorted(df['category'].unique())
    colors = {'Real': '#2ecc71', 'Fake': '#e74c3c', 'Unknown': '#95a5a6'}
    
    data_to_plot = [df[df['category'] == cat][metric].values for cat in categories]
    
    bp = ax.boxplot(data_to_plot, labels=categories, patch_artist=True)
    
    for patch, cat in zip(bp['boxes'], categories):
        patch.set_facecolor(colors.get(cat, '#3498db'))
        patch.set_alpha(0.7)
    
    ax.set_ylabel(f'{metric.capitalize()} Score', fontsize=12)
    ax.set_title(f'{metric.capitalize()} Score Distribution by Category', fontsize=14)
    
    # 添加均值标记
    for i, cat in enumerate(categories):
        mean_val = df[df['category'] == cat][metric].mean()
        ax.scatter(i + 1, mean_val, color='gold', s=100, zorder=5, 
                   marker='D', edgecolor='black', linewidth=1, label='Mean' if i == 0 else '')
    
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'boxplot_{metric}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: boxplot_{metric}.png")


def plot_violin(df: pd.DataFrame, output_dir: str, metric: str = 'criterion'):
    """绘制小提琴图"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    categories = sorted(df['category'].unique())
    colors = {'Real': '#2ecc71', 'Fake': '#e74c3c', 'Unknown': '#95a5a6'}
    
    data_to_plot = [df[df['category'] == cat][metric].values for cat in categories]
    
    parts = ax.violinplot(data_to_plot, positions=range(1, len(categories) + 1), 
                          showmeans=True, showmedians=True)
    
    for i, (pc, cat) in enumerate(zip(parts['bodies'], categories)):
        pc.set_facecolor(colors.get(cat, '#3498db'))
        pc.set_alpha(0.7)
    
    ax.set_xticks(range(1, len(categories) + 1))
    ax.set_xticklabels(categories)
    ax.set_ylabel(f'{metric.capitalize()} Score', fontsize=12)
    ax.set_title(f'{metric.capitalize()} Score Distribution (Violin Plot)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'violin_{metric}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: violin_{metric}.png")


def plot_kde(df: pd.DataFrame, output_dir: str, metric: str = 'criterion'):
    """绘制核密度估计图"""
    from scipy import stats
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = df['category'].unique()
    colors = {'Real': '#2ecc71', 'Fake': '#e74c3c', 'Unknown': '#95a5a6'}
    
    for cat in categories:
        data = df[df['category'] == cat][metric].values
        if len(data) > 1:
            kde = stats.gaussian_kde(data)
            x_range = np.linspace(data.min() - data.std(), data.max() + data.std(), 200)
            ax.plot(x_range, kde(x_range), label=f'{cat} (n={len(data)})', 
                   color=colors.get(cat, '#3498db'), linewidth=2)
            ax.fill_between(x_range, kde(x_range), alpha=0.3, color=colors.get(cat, '#3498db'))
    
    ax.set_xlabel(f'{metric.capitalize()} Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Kernel Density Estimation of {metric.capitalize()} Scores', fontsize=14)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'kde_{metric}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: kde_{metric}.png")


def plot_roc_curve(df: pd.DataFrame, output_dir: str, metric: str = 'criterion'):
    """绘制 ROC 曲线（假设 Real=0, Fake=1）"""
    from sklearn.metrics import roc_curve, auc
    
    # 只处理 Real 和 Fake
    df_binary = df[df['category'].isin(['Real', 'Fake'])].copy()
    if len(df_binary) == 0:
        print("  跳过 ROC 曲线: 没有 Real/Fake 标签")
        return
    
    y_true = (df_binary['category'] == 'Fake').astype(int).values
    y_scores = df_binary[metric].values
    
    # 计算 ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # 也计算反向（如果分数越低越可能是 fake）
    fpr_inv, tpr_inv, _ = roc_curve(y_true, -y_scores)
    roc_auc_inv = auc(fpr_inv, tpr_inv)
    
    # 选择更好的方向
    if roc_auc_inv > roc_auc:
        fpr, tpr, roc_auc = fpr_inv, tpr_inv, roc_auc_inv
        direction = "(lower score = Fake)"
    else:
        direction = "(higher score = Fake)"
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(fpr, tpr, color='#e74c3c', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve for Fake Detection {direction}', fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'roc_{metric}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: roc_{metric}.png (AUC={roc_auc:.4f})")
    
    return roc_auc


def plot_scatter_comparison(df: pd.DataFrame, output_dir: str, 
                           metric1: str = 'criterion', metric2: str = None):
    """绘制散点图（如果有多个指标）"""
    if metric2 is None or metric2 not in df.columns:
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    categories = df['category'].unique()
    colors = {'Real': '#2ecc71', 'Fake': '#e74c3c', 'Unknown': '#95a5a6'}
    markers = {'Real': 'o', 'Fake': 's', 'Unknown': '^'}
    
    for cat in categories:
        mask = df['category'] == cat
        ax.scatter(df.loc[mask, metric1], df.loc[mask, metric2],
                  c=colors.get(cat, '#3498db'), marker=markers.get(cat, 'o'),
                  label=f'{cat} (n={mask.sum()})', alpha=0.6, s=50)
    
    ax.set_xlabel(f'{metric1.capitalize()}', fontsize=12)
    ax.set_ylabel(f'{metric2.capitalize()}', fontsize=12)
    ax.set_title(f'{metric1.capitalize()} vs {metric2.capitalize()}', fontsize=14)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'scatter_{metric1}_vs_{metric2}.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: scatter_{metric1}_vs_{metric2}.png")


def plot_summary_stats(df: pd.DataFrame, output_dir: str, metric: str = 'criterion'):
    """绘制统计汇总条形图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    categories = sorted(df['category'].unique())
    colors = {'Real': '#2ecc71', 'Fake': '#e74c3c', 'Unknown': '#95a5a6'}
    color_list = [colors.get(cat, '#3498db') for cat in categories]
    
    # Mean
    means = [df[df['category'] == cat][metric].mean() for cat in categories]
    axes[0].bar(categories, means, color=color_list, alpha=0.8, edgecolor='black')
    axes[0].set_title('Mean', fontsize=12)
    axes[0].set_ylabel(f'{metric.capitalize()} Score')
    for i, v in enumerate(means):
        axes[0].text(i, v + 0.01 * max(means), f'{v:.4f}', ha='center', fontsize=10)
    
    # Std
    stds = [df[df['category'] == cat][metric].std() for cat in categories]
    axes[1].bar(categories, stds, color=color_list, alpha=0.8, edgecolor='black')
    axes[1].set_title('Standard Deviation', fontsize=12)
    axes[1].set_ylabel(f'{metric.capitalize()} Score')
    for i, v in enumerate(stds):
        axes[1].text(i, v + 0.01 * max(stds), f'{v:.4f}', ha='center', fontsize=10)
    
    # Count
    counts = [len(df[df['category'] == cat]) for cat in categories]
    axes[2].bar(categories, counts, color=color_list, alpha=0.8, edgecolor='black')
    axes[2].set_title('Sample Count', fontsize=12)
    axes[2].set_ylabel('Count')
    for i, v in enumerate(counts):
        axes[2].text(i, v + 0.01 * max(counts), f'{v}', ha='center', fontsize=10)
    
    plt.suptitle(f'Summary Statistics for {metric.capitalize()}', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'summary_{metric}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: summary_{metric}.png")


def generate_report(df: pd.DataFrame, output_dir: str, metric: str = 'criterion'):
    """生成文本统计报告"""
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("评估结果统计报告")
    report_lines.append("=" * 60)
    report_lines.append("")
    
    # 总体统计
    report_lines.append("【总体统计】")
    report_lines.append(f"  总样本数: {len(df)}")
    report_lines.append(f"  指标: {metric}")
    report_lines.append(f"  均值: {df[metric].mean():.6f}")
    report_lines.append(f"  标准差: {df[metric].std():.6f}")
    report_lines.append(f"  最小值: {df[metric].min():.6f}")
    report_lines.append(f"  最大值: {df[metric].max():.6f}")
    report_lines.append("")
    
    # 分类统计
    report_lines.append("【分类统计】")
    for cat in sorted(df['category'].unique()):
        subset = df[df['category'] == cat]
        report_lines.append(f"\n  {cat}:")
        report_lines.append(f"    样本数: {len(subset)}")
        report_lines.append(f"    均值: {subset[metric].mean():.6f}")
        report_lines.append(f"    标准差: {subset[metric].std():.6f}")
        report_lines.append(f"    中位数: {subset[metric].median():.6f}")
        report_lines.append(f"    最小值: {subset[metric].min():.6f}")
        report_lines.append(f"    最大值: {subset[metric].max():.6f}")
    
    # 统计检验（如果有 Real 和 Fake）
    if 'Real' in df['category'].values and 'Fake' in df['category'].values:
        from scipy import stats
        
        real_data = df[df['category'] == 'Real'][metric]
        fake_data = df[df['category'] == 'Fake'][metric]
        
        # t-test
        t_stat, t_pval = stats.ttest_ind(real_data, fake_data)
        
        # Mann-Whitney U test
        u_stat, u_pval = stats.mannwhitneyu(real_data, fake_data, alternative='two-sided')
        
        report_lines.append("")
        report_lines.append("【统计检验】")
        report_lines.append(f"  Independent t-test:")
        report_lines.append(f"    t-statistic: {t_stat:.4f}")
        report_lines.append(f"    p-value: {t_pval:.6f}")
        report_lines.append(f"    显著性: {'显著 (p<0.05)' if t_pval < 0.05 else '不显著'}")
        report_lines.append("")
        report_lines.append(f"  Mann-Whitney U test:")
        report_lines.append(f"    U-statistic: {u_stat:.4f}")
        report_lines.append(f"    p-value: {u_pval:.6f}")
        report_lines.append(f"    显著性: {'显著 (p<0.05)' if u_pval < 0.05 else '不显著'}")
        
        # 效应量 Cohen's d
        pooled_std = np.sqrt((real_data.std()**2 + fake_data.std()**2) / 2)
        cohens_d = (real_data.mean() - fake_data.mean()) / pooled_std
        report_lines.append("")
        report_lines.append(f"  效应量 (Cohen's d): {cohens_d:.4f}")
        if abs(cohens_d) < 0.2:
            effect_size = "微小"
        elif abs(cohens_d) < 0.5:
            effect_size = "小"
        elif abs(cohens_d) < 0.8:
            effect_size = "中等"
        else:
            effect_size = "大"
        report_lines.append(f"    效应大小: {effect_size}")
    
    report_lines.append("")
    report_lines.append("=" * 60)
    
    # 保存报告
    report_text = "\n".join(report_lines)
    report_path = os.path.join(output_dir, "statistics_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"  保存: statistics_report.txt")
    print("\n" + report_text)
    
    return report_text


# ==================== 主函数 ====================

def visualize_results(df: pd.DataFrame, output_dir: str, metric: str = 'criterion'):
    """生成所有可视化"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n生成可视化图表...")
    print(f"输出目录: {output_dir}")
    print(f"评估指标: {metric}")
    print("-" * 40)
    
    # 检查类别分布
    print(f"\n类别分布:")
    for cat in df['category'].unique():
        print(f"  {cat}: {len(df[df['category'] == cat])} 张图片")
    print()
    
    # 生成各种图表
    plot_histogram(df, output_dir, metric)
    plot_boxplot(df, output_dir, metric)
    plot_violin(df, output_dir, metric)
    
    try:
        plot_kde(df, output_dir, metric)
    except Exception as e:
        print(f"  跳过 KDE 图: {e}")
    
    try:
        plot_roc_curve(df, output_dir, metric)
    except Exception as e:
        print(f"  跳过 ROC 曲线: {e}")
    
    plot_summary_stats(df, output_dir, metric)
    
    # 如果有额外的指标，生成散点图
    extra_metrics = [col for col in df.columns if col not in 
                     ['image_path', 'prompt', 'category', 'criterion']]
    for extra in extra_metrics[:3]:  # 最多3个额外指标
        try:
            plot_scatter_comparison(df, output_dir, metric, extra)
        except Exception as e:
            print(f"  跳过散点图 {extra}: {e}")
    
    # 生成统计报告
    print()
    generate_report(df, output_dir, metric)
    
    print(f"\n可视化完成！所有图表保存在: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="可视化评估结果")
    
    # 输入选项
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--results", type=str, help="结果文件路径 (JSON 或 CSV)")
    input_group.add_argument("--dir", type=str, help="图片目录（会先运行评估）")
    
    # 评估选项（当使用 --dir 时）
    parser.add_argument("--criterion", type=str, default="pde", help="评估方法")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-noise", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("-r", "--recursive", action="store_true")
    
    # 输出选项
    parser.add_argument("--output", type=str, default="figures", help="输出目录")
    parser.add_argument("--metric", type=str, default="criterion", help="要可视化的指标")
    
    args = parser.parse_args()
    
    # 加载或生成结果
    if args.results:
        print(f"加载结果: {args.results}")
        df = load_results(args.results)
    else:
        # 先运行评估
        print(f"对目录 {args.dir} 运行评估...")
        from config import EvalConfig
        from evaluator import DiffusionEvaluator
        
        config = EvalConfig(
            criterion_name=args.criterion,
            device=args.device,
            batch_size=args.batch_size,
            num_noise=args.num_noise,
            output_dir=os.path.join(args.output, "eval_results"),
            return_terms=True,
        )
        
        # 查找图片
        from main import find_images
        image_paths = find_images(args.dir, recursive=args.recursive)
        print(f"找到 {len(image_paths)} 张图片")
        
        # 运行评估
        evaluator = DiffusionEvaluator(config)
        results = evaluator.evaluate_images(image_paths)
        evaluator.save_results(results)
        evaluator.cleanup()
        
        df = pd.DataFrame(results)
        df['category'] = df['image_path'].apply(extract_category)
    
    # 生成可视化
    visualize_results(df, args.output, args.metric)


if __name__ == "__main__":
    main()
