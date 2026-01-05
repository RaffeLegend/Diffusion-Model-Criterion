"""
Template for Custom Criterion

Copy this file and modify to create your own criterion.
The key method to implement is evaluate_batch().

Usage:
    1. Copy this file to criteria/my_criterion.py
    2. Rename the class and change the register name
    3. Implement your evaluate_batch logic
    4. Import in criteria/__init__.py
    5. Use with config.criterion_name = "my_criterion"
"""

import torch
from typing import Dict, List, Optional

from criteria.base import BaseCriterion, register_criterion


# Change "template" to your criterion name
@register_criterion("template")
class TemplateCriterion(BaseCriterion):
    """
    Template criterion - copy and modify for your experiments.
    
    Document your criterion's purpose and methodology here.
    """
    
    def __init__(self, config, model_manager):
        super().__init__(config, model_manager)
        
        # Initialize any additional components your criterion needs
        # Example: self.some_layer = torch.nn.Linear(...)
        
        # You can access criterion-specific params from config
        self.custom_param = config.criterion_params.get("custom_param", 1.0)
    
    def evaluate_batch(
        self,
        images: torch.Tensor,
        prompts: List[str],
        images_raw: Optional[List[torch.Tensor]] = None
    ) -> List[Dict]:
        """
        Evaluate a batch of images with your custom criterion.
        
        Args:
            images: Preprocessed images (B, C, H, W) in [-1, 1], dtype=float16
            prompts: Text prompts for each image, length B
            images_raw: Optional raw images (H, W, C) in [0, 255]
            
        Returns:
            List of dicts, one per image, each containing at least:
                {"criterion": float_value}
            
            If config.return_terms is True, include additional metrics.
        """
        num_images = images.shape[0]
        num_noise = self.config.num_noise
        
        # ========== YOUR ALGORITHM HERE ==========
        
        # Example: Access models
        unet = self.model_manager.unet
        scheduler = self.model_manager.scheduler
        
        # Example: Get text embeddings (inherited helper)
        text_emb = self.compute_text_embeddings(prompts, num_noise)
        
        # Example: Encode to latent space (inherited helper)
        z0 = self.encode_images_to_latents(images)
        
        # Example: Your custom computation
        # ...
        
        # Placeholder: random scores for template
        scores = torch.rand(num_images)
        
        # ========== END OF YOUR ALGORITHM ==========
        
        # Build results
        results = []
        for i in range(num_images):
            result = {"criterion": float(scores[i].item())}
            
            # Add extra terms if requested
            if self.config.return_terms:
                result.update({
                    "extra_metric_1": 0.0,
                    "extra_metric_2": 0.0,
                })
            
            results.append(result)
        
        return results


# ==================== Quick Test ====================

if __name__ == "__main__":
    # Quick test of your criterion
    print(f"Criterion 'template' registered successfully")
    print(f"Available criteria: {list_criteria()}")
