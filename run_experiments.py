#!/usr/bin/env python
"""
实验执行脚本

提供多种运行模式：
1. 单次评估
2. 批量对比实验
3. 参数扫描实验
4. 真实 vs 生成图片对比

用法:
    python run_experiments.py single --criterion pde --dir ./images
    python run_experiments.py compare --real-dir ./real --gen-dir ./generated
    python run_experiments.py sweep --dir ./images --param num_noise --values 4 8 16 32
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd

from config import EvalConfig
from evaluator import DiffusionEvaluator
from criteria import list_criteria


def get_timestamp():
    """获取时间戳字符串"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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


# ==================== 单次评估 ====================

def run_single(args):
    """运行单次评估"""
    image_paths = find_images(args.dir, args.recursive)
    
    if not image_paths:
        print(f"错误: 在 {args.dir} 中没有找到图片")
        return
    
    print(f"找到 {len(image_paths)} 张图片")
    
    config = EvalConfig(
        criterion_name=args.criterion,
        device=args.device,
        batch_size=args.batch_size,
        num_noise=args.num_noise,
        time_frac=args.time_frac,
        image_size=args.image_size,
        output_dir=args.output_dir,
        return_terms=args.return_terms,
    )
    
    evaluator = DiffusionEvaluator(config)
    results = evaluator.evaluate_images(image_paths)
    
    evaluator.save_results(results)
    evaluator.print_summary(results)
    evaluator.cleanup()
    
    # 保存为 CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV 结果保存至: {csv_path}")


# ==================== 对比实验 ====================

def run_compare(args):
    """运行真实 vs 生成图片对比实验"""
    real_images = find_images(args.real_dir, args.recursive)
    gen_images = find_images(args.gen_dir, args.recursive)
    
    print(f"真实图片: {len(real_images)} 张")
    print(f"生成图片: {len(gen_images)} 张")
    
    if not real_images or not gen_images:
        print("错误: 图片目录为空")
        return
    
    base_output = args.output_dir or f"results/compare_{get_timestamp()}"
    os.makedirs(base_output, exist_ok=True)
    
    all_results = []
    
    for criterion in args.criteria:
        print(f"\n{'='*60}")
        print(f"使用方法: {criterion}")
        print(f"{'='*60}")
        
        config = EvalConfig(
            criterion_name=criterion,
            device=args.device,
            batch_size=args.batch_size,
            num_noise=args.num_noise,
            output_dir=os.path.join(base_output, criterion),
            return_terms=True,
        )
        
        evaluator = DiffusionEvaluator(config)
        
        # 评估真实图片
        print("\n评估真实图片...")
        real_results = evaluator.evaluate_images(real_images)
        for r in real_results:
            r["category"] = "real"
            r["criterion_name"] = criterion
        
        # 评估生成图片
        print("\n评估生成图片...")
        gen_results = evaluator.evaluate_images(gen_images)
        for r in gen_results:
            r["category"] = "generated"
            r["criterion_name"] = criterion
        
        all_results.extend(real_results)
        all_results.extend(gen_results)
        
        # 打印对比
        real_scores = [r["criterion"] for r in real_results]
        gen_scores = [r["criterion"] for r in gen_results]
        
        print(f"\n{criterion} 结果:")
        print(f"  真实图片 - 均值: {sum(real_scores)/len(real_scores):.6f}")
        print(f"  生成图片 - 均值: {sum(gen_scores)/len(gen_scores):.6f}")
        
        evaluator.cleanup()
    
    # 保存汇总结果
    df = pd.DataFrame(all_results)
    summary_path = os.path.join(base_output, "comparison_results.csv")
    df.to_csv(summary_path, index=False)
    print(f"\n汇总结果保存至: {summary_path}")
    
    # 生成统计报告
    print("\n" + "="*60)
    print("统计汇总")
    print("="*60)
    summary = df.groupby(["criterion_name", "category"])["criterion"].agg(["mean", "std", "min", "max"])
    print(summary)


# ==================== 参数扫描 ====================

def run_sweep(args):
    """运行参数扫描实验"""
    image_paths = find_images(args.dir, args.recursive)
    
    if not image_paths:
        print(f"错误: 在 {args.dir} 中没有找到图片")
        return
    
    # 限制图片数量以加速扫描
    if args.max_images and len(image_paths) > args.max_images:
        import random
        random.seed(42)
        image_paths = random.sample(image_paths, args.max_images)
        print(f"随机采样 {args.max_images} 张图片进行参数扫描")
    
    base_output = args.output_dir or f"results/sweep_{args.param}_{get_timestamp()}"
    os.makedirs(base_output, exist_ok=True)
    
    all_results = []
    
    for value in args.values:
        print(f"\n{'='*60}")
        print(f"参数 {args.param} = {value}")
        print(f"{'='*60}")
        
        # 构建配置
        config_kwargs = {
            "criterion_name": args.criterion,
            "device": args.device,
            "batch_size": args.batch_size,
            "output_dir": os.path.join(base_output, f"{args.param}_{value}"),
            "return_terms": True,
        }
        
        # 设置扫描的参数
        if args.param == "num_noise":
            config_kwargs["num_noise"] = int(value)
        elif args.param == "time_frac":
            config_kwargs["time_frac"] = float(value)
        elif args.param == "batch_size":
            config_kwargs["batch_size"] = int(value)
        elif args.param == "image_size":
            config_kwargs["image_size"] = int(value)
        else:
            print(f"警告: 未知参数 {args.param}")
        
        config = EvalConfig(**config_kwargs)
        evaluator = DiffusionEvaluator(config)
        
        results = evaluator.evaluate_images(image_paths)
        
        for r in results:
            r["param_name"] = args.param
            r["param_value"] = value
        
        all_results.extend(results)
        
        scores = [r["criterion"] for r in results]
        print(f"  均值: {sum(scores)/len(scores):.6f}")
        
        evaluator.cleanup()
    
    # 保存汇总结果
    df = pd.DataFrame(all_results)
    summary_path = os.path.join(base_output, "sweep_results.csv")
    df.to_csv(summary_path, index=False)
    print(f"\n扫描结果保存至: {summary_path}")
    
    # 打印参数对比
    print("\n" + "="*60)
    print(f"参数 {args.param} 扫描结果")
    print("="*60)
    summary = df.groupby("param_value")["criterion"].agg(["mean", "std"])
    print(summary)


# ==================== 多方法对比 ====================

def run_multi(args):
    """运行多种方法对比"""
    image_paths = find_images(args.dir, args.recursive)
    
    if not image_paths:
        print(f"错误: 在 {args.dir} 中没有找到图片")
        return
    
    base_output = args.output_dir or f"results/multi_{get_timestamp()}"
    os.makedirs(base_output, exist_ok=True)
    
    all_results = []
    
    for criterion in args.criteria:
        print(f"\n{'='*60}")
        print(f"方法: {criterion}")
        print(f"{'='*60}")
        
        config = EvalConfig(
            criterion_name=criterion,
            device=args.device,
            batch_size=args.batch_size,
            num_noise=args.num_noise,
            output_dir=os.path.join(base_output, criterion),
            return_terms=True,
        )
        
        evaluator = DiffusionEvaluator(config)
        results = evaluator.evaluate_images(image_paths)
        
        for r in results:
            r["criterion_name"] = criterion
        
        all_results.extend(results)
        evaluator.print_summary(results)
        evaluator.cleanup()
    
    # 保存汇总
    df = pd.DataFrame(all_results)
    summary_path = os.path.join(base_output, "multi_results.csv")
    df.to_csv(summary_path, index=False)
    print(f"\n汇总结果保存至: {summary_path}")


# ==================== 主入口 ====================

def main():
    parser = argparse.ArgumentParser(
        description="实验执行脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="运行模式")
    
    # 通用参数
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--device", default="cuda", help="设备 (cuda/cpu)")
    common_parser.add_argument("--batch-size", type=int, default=4, help="批处理大小")
    common_parser.add_argument("--num-noise", type=int, default=8, help="噪声样本数")
    common_parser.add_argument("--time-frac", type=float, default=0.01, help="时间步比例")
    common_parser.add_argument("--image-size", type=int, default=512, help="图片尺寸")
    common_parser.add_argument("-r", "--recursive", action="store_true", help="递归搜索")
    common_parser.add_argument("--return-terms", action="store_true", help="返回详细指标")
    
    # single 命令
    single_parser = subparsers.add_parser("single", parents=[common_parser], help="单次评估")
    single_parser.add_argument("--criterion", default="pde", help="评估方法")
    single_parser.add_argument("--dir", required=True, help="图片目录")
    single_parser.add_argument("--output-dir", default="results", help="输出目录")
    
    # compare 命令
    compare_parser = subparsers.add_parser("compare", parents=[common_parser], help="真实 vs 生成对比")
    compare_parser.add_argument("--real-dir", required=True, help="真实图片目录")
    compare_parser.add_argument("--gen-dir", required=True, help="生成图片目录")
    compare_parser.add_argument("--criteria", nargs="+", default=["pde", "clip"], help="评估方法列表")
    compare_parser.add_argument("--output-dir", help="输出目录")
    
    # sweep 命令
    sweep_parser = subparsers.add_parser("sweep", parents=[common_parser], help="参数扫描")
    sweep_parser.add_argument("--criterion", default="pde", help="评估方法")
    sweep_parser.add_argument("--dir", required=True, help="图片目录")
    sweep_parser.add_argument("--param", required=True, help="要扫描的参数名")
    sweep_parser.add_argument("--values", nargs="+", required=True, help="参数值列表")
    sweep_parser.add_argument("--max-images", type=int, help="最大图片数量")
    sweep_parser.add_argument("--output-dir", help="输出目录")
    
    # multi 命令
    multi_parser = subparsers.add_parser("multi", parents=[common_parser], help="多方法对比")
    multi_parser.add_argument("--dir", required=True, help="图片目录")
    multi_parser.add_argument("--criteria", nargs="+", default=["pde", "clip"], help="评估方法列表")
    multi_parser.add_argument("--output-dir", help="输出目录")
    
    # list 命令
    subparsers.add_parser("list", help="列出可用的评估方法")
    
    args = parser.parse_args()
    
    if args.command == "list":
        print("可用的评估方法:")
        for name in list_criteria():
            print(f"  - {name}")
    elif args.command == "single":
        run_single(args)
    elif args.command == "compare":
        run_compare(args)
    elif args.command == "sweep":
        run_sweep(args)
    elif args.command == "multi":
        run_multi(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

# # 单次评估
# python run_experiments.py single --criterion pde --dir ./images

# # 真实 vs 生成图片对比
# python run_experiments.py compare --real-dir ./real --gen-dir ./generated

# # 参数扫描（如扫描 num_noise 的影响）
# python run_experiments.py sweep --dir ./images --param num_noise --values 4 8 16 32

# # 多方法对比
# python run_experiments.py multi --dir ./images --criteria pde clip

# # 查看可用方法
# python run_experiments.py list