"""
Criteria Package

This package contains all evaluation criteria for diffusion models.
Each criterion implements the BaseCriterion interface.

Available criteria:
    - pde: PDE/Stein residual criterion
    - clip: CLIP-based manifold criterion
    - template: Template for creating new criteria

To add a new criterion:
    1. Create a new file: criteria/my_criterion.py
    2. Implement a class inheriting from BaseCriterion
    3. Decorate with @register_criterion("my_name")
    4. Import here to register automatically

Usage:
    from criteria import create_criterion, list_criteria
    
    # List available criteria
    print(list_criteria())
    
    # Create a criterion instance
    criterion = create_criterion(config, model_manager)
    results = criterion.evaluate_batch(images, prompts)
"""

# Import base class and registry functions
from criteria.base import (
    BaseCriterion,
    register_criterion,
    get_criterion,
    list_criteria,
    create_criterion,
)

# Import all criterion implementations to register them
from criteria.pde_criterion import PDECriterion
from criteria.clip_criterion import CLIPCriterion
from criteria.clip_criterion_enhanced import EnhancedCLIPCriterion, CLIPWithDSIRCriterion

# Optional: Import template for reference (comment out in production)
# from criteria.template_criterion import TemplateCriterion


__all__ = [
    # Base class
    "BaseCriterion",
    # Registry functions
    "register_criterion",
    "get_criterion", 
    "list_criteria",
    "create_criterion",
    # Concrete implementations
    "PDECriterion",
    "CLIPCriterion",
    "EnhancedCLIPCriterion"
]
