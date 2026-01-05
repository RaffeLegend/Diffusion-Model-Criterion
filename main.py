#!/usr/bin/env python
"""
Main entry point for diffusion model evaluation.

Usage:
    # Basic evaluation with PDE criterion
    python main.py --dir /path/to/images --criterion pde
    
    # Evaluation with CLIP criterion
    python main.py --dir /path/to/images --criterion clip
    
    # With custom parameters
    python main.py --dir /path/to/images --criterion pde --batch-size 8 --num-noise 16
    
    # List available criteria
    python main.py --list-criteria
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional

from config import EvalConfig
from evaluator import DiffusionEvaluator
from criteria import list_criteria


def find_images(
    directory: str, 
    extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp'),
    recursive: bool = False
) -> List[str]:
    """Find all images in a directory"""
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


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Diffusion Model Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dir ./images --criterion pde
  %(prog)s --dir ./images --criterion clip --batch-size 8
  %(prog)s --list-criteria
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--dir", type=str,
        help="Directory containing images"
    )
    input_group.add_argument(
        "--images", type=str, nargs='+',
        help="List of image paths"
    )
    input_group.add_argument(
        "--list-criteria", action="store_true",
        help="List available criteria and exit"
    )
    
    # Criterion selection
    parser.add_argument(
        "--criterion", type=str, default="pde",
        help="Criterion to use (default: pde)"
    )
    
    # Processing options
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size for processing (default: 4)"
    )
    parser.add_argument(
        "--num-noise", type=int, default=8,
        help="Number of noise samples per image (default: 8)"
    )
    parser.add_argument(
        "--time-frac", type=float, default=0.01,
        help="Time fraction for diffusion (default: 0.01)"
    )
    parser.add_argument(
        "--image-size", type=int, default=512,
        help="Image size (default: 512)"
    )
    
    # Device options
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Output directory (default: results)"
    )
    parser.add_argument(
        "--return-terms", action="store_true",
        help="Return detailed intermediate terms"
    )
    
    # Directory traversal
    parser.add_argument(
        "-r", "--recursive", action="store_true",
        help="Recursively search subdirectories"
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # List criteria mode
    if args.list_criteria:
        print("Available criteria:")
        for name in list_criteria():
            print(f"  - {name}")
        return
    
    # Get image paths
    if args.dir:
        image_paths = find_images(args.dir, recursive=args.recursive)
        print(f"Found {len(image_paths)} images in {args.dir}")
    elif args.images:
        image_paths = args.images
        print(f"Processing {len(image_paths)} specified images")
    else:
        print("Error: Must specify --dir, --images, or --list-criteria")
        return
    
    if not image_paths:
        print("Error: No images found!")
        return
    
    # Create configuration
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
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run evaluation
    print(f"\nUsing criterion: {args.criterion}")
    print(f"Output directory: {args.output_dir}")
    print("-" * 50)
    
    evaluator = DiffusionEvaluator(config)
    results = evaluator.evaluate_images(image_paths)
    
    # Save and display results
    evaluator.save_results(results)
    evaluator.print_summary(results)
    
    # Print individual results
    print("\nIndividual Results:")
    print("-" * 50)
    for result in results:
        print(f"  {Path(result['image_path']).name}: {result['criterion']:.6f}")
    
    # Cleanup
    evaluator.cleanup()


if __name__ == "__main__":
    main()
