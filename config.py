"""
Configuration for Diffusion Model Evaluation Framework
"""
import json
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any


@dataclass
class EvalConfig:
    """Configuration for evaluation"""
    
    # Model settings
    model_name: str = "CompVis/stable-diffusion-v1-4"
    device: str = "cuda"
    dtype: str = "float16"
    
    # Criterion settings
    criterion_name: str = "pde"  # 'pde', 'clip', 'clip_with_dsir', 'clip_enhanced'
    num_noise: int = 8
    time_frac: float = 0.01
    epsilon_reg: float = 1e-8
    image_size: int = 512
    
    # Batch processing
    batch_size: int = 4
    num_workers: int = 4
    
    # Output settings
    output_dir: str = "results"
    save_intermediates: bool = False
    return_terms: bool = False
    
    # Caption settings
    use_provided_prompts: bool = False
    caption_model: str = "llava-hf/llava-1.5-7b-hf"
    
    # Extra criterion-specific parameters
    criterion_params: Dict[str, Any] = field(default_factory=dict)
    
    # ===== 新增: 增强版参数 =====
    dsir_weight: float = 0.01              # clip_with_dsir 用
    lambda_high_order: float = 0.1         # clip_enhanced 用
    high_order_type: str = "laplacian"     # laplacian/biharmonic/dsir
    
    def save(self, path: str):
        """Save configuration to JSON"""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'EvalConfig':
        """Load configuration from JSON"""
        with open(path, 'r') as f:
            return cls(**json.load(f))
    
    def __post_init__(self):
        """Validate configuration"""
        if self.dtype not in ("float16", "float32"):
            raise ValueError(f"dtype must be 'float16' or 'float32', got {self.dtype}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.num_noise < 1:
            raise ValueError(f"num_noise must be >= 1, got {self.num_noise}")