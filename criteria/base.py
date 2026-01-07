"""
Base Criterion Class and Registry

This module provides the abstract base class for all evaluation criteria
and a registry system for easy registration and lookup of criteria.

To create a new criterion:
1. Create a new file in the criteria/ directory
2. Inherit from BaseCriterion
3. Implement the evaluate_batch method
4. Register with @register_criterion("name")

Example:
    @register_criterion("my_criterion")
    class MyCriterion(BaseCriterion):
        def evaluate_batch(self, images, prompts, images_raw=None):
            # Your algorithm here
            return [{"criterion": score} for score in scores]
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type
import logging

import torch

from config import EvalConfig
from models import ModelManager


# Global registry for criteria
_CRITERION_REGISTRY: Dict[str, Type['BaseCriterion']] = {}


def register_criterion(name: str):
    """
    Decorator to register a criterion class.
    
    Args:
        name: Unique name for the criterion
        
    Example:
        @register_criterion("pde")
        class PDECriterion(BaseCriterion):
            ...
    """
    def decorator(cls: Type['BaseCriterion']) -> Type['BaseCriterion']:
        if name in _CRITERION_REGISTRY:
            logging.warning(f"Criterion '{name}' already registered. Overwriting.")
        _CRITERION_REGISTRY[name] = cls
        cls.criterion_name = name
        return cls
    return decorator


def get_criterion(name: str) -> Type['BaseCriterion']:
    """
    Get a criterion class by name.
    
    Args:
        name: Name of the registered criterion
        
    Returns:
        The criterion class
        
    Raises:
        ValueError: If criterion not found
    """
    if name not in _CRITERION_REGISTRY:
        available = list(_CRITERION_REGISTRY.keys())
        raise ValueError(
            f"Unknown criterion '{name}'. Available criteria: {available}"
        )
    return _CRITERION_REGISTRY[name]


def list_criteria() -> List[str]:
    """List all registered criteria names"""
    return list(_CRITERION_REGISTRY.keys())


def create_criterion(
    config: EvalConfig, 
    model_manager: ModelManager
) -> 'BaseCriterion':
    """
    Factory function to create a criterion instance.
    
    Args:
        config: Evaluation configuration
        model_manager: Model manager instance
        
    Returns:
        Instantiated criterion
    """
    criterion_cls = get_criterion(config.criterion_name)
    return criterion_cls(config, model_manager)


class BaseCriterion(ABC):
    """
    Abstract base class for all evaluation criteria.
    
    Subclasses must implement the evaluate_batch method.
    """
    
    criterion_name: str = "base"  # Will be set by register_criterion
    
    def __init__(self, config: EvalConfig, model_manager: ModelManager):
        """
        Initialize the criterion.
        
        Args:
            config: Evaluation configuration
            model_manager: Model manager for accessing models
        """
        self.config = config
        self.model_manager = model_manager
        self.device = torch.device(config.device)
        self.dtype = torch.float16 if config.dtype == "float16" else torch.float32
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def evaluate_batch(
        self,
        images: torch.Tensor,
        prompts: List[str],
        images_raw: Optional[List[torch.Tensor]] = None
    ) -> List[Dict]:
        """
        Evaluate a batch of images.
        
        This is the core method that each criterion must implement.
        
        Args:
            images: Preprocessed images tensor of shape (B, C, H, W) 
                   with values in [-1, 1]
            prompts: List of text prompts for each image
            images_raw: Optional list of raw image tensors (H, W, C) 
                       with values in [0, 255]. Some criteria may need this.
        
        Returns:
            List of dictionaries, one per image, containing at minimum:
                - "criterion": float - the main criterion score
            May contain additional keys if config.return_terms is True.
        """
        pass
    
    # ==================== Common Utility Methods ====================
    
    def compute_text_embeddings(
        self, 
        prompts: List[str], 
        num_repeats: int = 1
    ) -> torch.Tensor:
        """
        Compute text embeddings for prompts using SD's text encoder.
        
        注意: 此方法返回的tensor需要调用者在用完后手动del以释放显存
        
        Args:
            prompts: List of text prompts
            num_repeats: Number of times to repeat each prompt's embedding
            
        Returns:
            Text embeddings tensor of shape (len(prompts) * num_repeats, 77, dim)
        """
        tokenizer = self.model_manager.tokenizer
        text_encoder = self.model_manager.text_encoder
        
        # Expand prompts if needed
        if num_repeats > 1:
            expanded_prompts = []
            for p in prompts:
                expanded_prompts.extend([p] * num_repeats)
        else:
            expanded_prompts = prompts
        
        text_tokens = tokenizer(
            expanded_prompts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        
        input_ids = text_tokens.input_ids.to(self.device)
        
        with torch.no_grad():
            text_emb = text_encoder(input_ids).last_hidden_state
        
        del input_ids  # 释放
        return text_emb
    
    def encode_images_to_latents(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent space using VAE.
        
        Args:
            images: Tensor of shape (B, C, H, W) with values in [-1, 1]
            
        Returns:
            Latent tensor of shape (B, 4, H/8, W/8)
        """
        vae = self.model_manager.vae
        
        with torch.no_grad():
            z0 = vae.encode(images).latent_dist.sample()
            z0 = z0 * vae.config.scaling_factor
        
        return z0
    
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to image space using VAE.
        
        Args:
            latents: Tensor of shape (B, 4, H/8, W/8)
            
        Returns:
            Image tensor of shape (B, C, H, W) with values in [-1, 1]
        """
        vae = self.model_manager.vae
        
        latents_scaled = latents / vae.config.scaling_factor
        
        with torch.no_grad():
            decoded = vae.decode(latents_scaled.half()).sample
        
        return decoded
    
    @staticmethod
    def normalize_batch(batch: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
        """Normalize each element in batch to unit norm"""
        dims_to_normalize = tuple(range(1, batch.dim()))
        norms = torch.norm(batch, p=2, dim=dims_to_normalize, keepdim=True)
        return batch / (norms + epsilon)