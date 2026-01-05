"""
Caption Generator - generates text descriptions for images
"""

import torch
import torchvision.transforms.functional as TF
from typing import List

from models import ModelManager


class CaptionGenerator:
    """Generates captions for images using vision-language models"""
    
    DEFAULT_PROMPT = "Generate a caption for the image that contains only facts and detailed."
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
    
    def generate_caption(self, image: torch.Tensor) -> str:
        """
        Generate caption for a single image.
        
        Args:
            image: Tensor of shape (H, W, C) with values in [0, 255]
            
        Returns:
            Generated caption string
        """
        image_to_text = self.model_manager.caption_model
        
        # Convert tensor to PIL Image
        pil_image = TF.to_pil_image(image.permute(2, 0, 1))
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": self.DEFAULT_PROMPT},
            ],
        }]
        
        output = image_to_text(
            text=messages, 
            generate_kwargs={"max_new_tokens": 76}
        )
        
        caption = output[0]["generated_text"][-1]["content"].strip()
        return caption
    
    def generate_captions_batch(self, images: List[torch.Tensor]) -> List[str]:
        """
        Generate captions for a batch of images.
        
        Args:
            images: List of tensors, each (H, W, C) with values in [0, 255]
            
        Returns:
            List of caption strings
        """
        return [self.generate_caption(img) for img in images]
