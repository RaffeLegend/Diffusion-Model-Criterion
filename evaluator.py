"""
Diffusion Model Evaluator

Main evaluator class that orchestrates the evaluation pipeline.
"""

import os
import json
import logging
from typing import List, Dict, Optional
from tqdm import tqdm
import numpy as np

from config import EvalConfig
from models import ModelManager
from image_utils import ImageProcessor
from caption import CaptionGenerator
from criteria import create_criterion, BaseCriterion


def setup_logging(output_dir: str, level=logging.INFO) -> logging.Logger:
    """Setup logging configuration"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'evaluation.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class DiffusionEvaluator:
    """
    Main evaluator class for diffusion model evaluation.
    
    Coordinates image loading, preprocessing, caption generation,
    and criterion evaluation.
    
    Example:
        config = EvalConfig(
            criterion_name="pde",
            batch_size=4,
            num_noise=8,
        )
        evaluator = DiffusionEvaluator(config)
        results = evaluator.evaluate_images(image_paths)
        evaluator.print_summary(results)
    """
    
    def __init__(self, config: EvalConfig):
        """
        Initialize the evaluator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.logger = setup_logging(config.output_dir)
        self.logger.info(f"Initialized evaluator with config: {config}")
        
        # Initialize components
        self.model_manager = ModelManager(config)
        self.image_processor = ImageProcessor(config)
        self.caption_generator = CaptionGenerator(self.model_manager)
        
        # Create criterion based on config
        self.criterion: BaseCriterion = create_criterion(config, self.model_manager)
        self.logger.info(f"Using criterion: {config.criterion_name}")
        
        # Save config
        config.save(os.path.join(config.output_dir, "config.json"))
    
    def evaluate_images(
        self,
        image_paths: List[str],
        prompts: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Evaluate a list of images.
        
        Args:
            image_paths: List of paths to images
            prompts: Optional list of prompts (one per image).
                    If None, captions will be generated automatically.
        
        Returns:
            List of result dictionaries, one per image
        """
        self.logger.info(f"Evaluating {len(image_paths)} images")
        
        all_results = []
        
        # Process in batches
        for i in tqdm(range(0, len(image_paths), self.config.batch_size), 
                     desc="Processing batches"):
            batch_paths = image_paths[i:i + self.config.batch_size]
            
            # Load and preprocess images
            raw_images = self.image_processor.load_image_batch(batch_paths)
            processed_images = self.image_processor.preprocess(raw_images)
            
            # Get or generate prompts
            if prompts is not None:
                batch_prompts = prompts[i:i + self.config.batch_size]
            else:
                batch_prompts = self.caption_generator.generate_captions_batch(raw_images)
            
            # Evaluate batch
            batch_results = self.criterion.evaluate_batch(
                processed_images,
                batch_prompts,
                images_raw=raw_images  # Pass raw images for criteria that need them
            )
            
            # Add metadata
            for j, result in enumerate(batch_results):
                result["image_path"] = batch_paths[j]
                result["prompt"] = batch_prompts[j]
            
            all_results.extend(batch_results)
            
            # Clear cache periodically
            if (i // self.config.batch_size) % 10 == 0:
                self.model_manager.clear_cache()
        
        return all_results
    
    def evaluate_single(
        self,
        image_path: str,
        prompt: Optional[str] = None
    ) -> Dict:
        """
        Evaluate a single image.
        
        Args:
            image_path: Path to the image
            prompt: Optional prompt. If None, caption will be generated.
            
        Returns:
            Result dictionary
        """
        results = self.evaluate_images(
            [image_path], 
            [prompt] if prompt else None
        )
        return results[0]
    
    def save_results(self, results: List[Dict], filename: str = "results.json"):
        """Save results to JSON file"""
        output_path = os.path.join(self.config.output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Results saved to {output_path}")
    
    def print_summary(self, results: List[Dict]):
        """Print summary statistics"""
        criteria = [r["criterion"] for r in results]
        
        self.logger.info("=" * 50)
        self.logger.info("EVALUATION SUMMARY")
        self.logger.info("=" * 50)
        self.logger.info(f"Criterion: {self.config.criterion_name}")
        self.logger.info(f"Total images: {len(results)}")
        self.logger.info(f"Mean criterion: {np.mean(criteria):.6f}")
        self.logger.info(f"Std criterion: {np.std(criteria):.6f}")
        self.logger.info(f"Min criterion: {np.min(criteria):.6f}")
        self.logger.info(f"Max criterion: {np.max(criteria):.6f}")
        self.logger.info(f"Median criterion: {np.median(criteria):.6f}")
        self.logger.info("=" * 50)
    
    def cleanup(self):
        """Release all resources"""
        self.model_manager.unload_all()
        self.logger.info("Evaluator cleaned up")


# ==================== Convenience Functions ====================

def quick_evaluate(
    image_paths: List[str],
    criterion_name: str = "pde",
    prompts: Optional[List[str]] = None,
    **config_kwargs
) -> List[Dict]:
    """
    Quick evaluation function for simple use cases.
    
    Args:
        image_paths: List of image paths
        criterion_name: Name of criterion to use
        prompts: Optional prompts
        **config_kwargs: Additional config parameters
        
    Returns:
        List of result dictionaries
    """
    config = EvalConfig(
        criterion_name=criterion_name,
        **config_kwargs
    )
    evaluator = DiffusionEvaluator(config)
    results = evaluator.evaluate_images(image_paths, prompts)
    evaluator.cleanup()
    return results
