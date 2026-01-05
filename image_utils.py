"""
Image Processing Utilities
"""

import torch
import torchvision
import numpy as np
import imageio
from typing import List
from PIL import Image

from config import EvalConfig


class ImageProcessor:
    """Handles image loading and preprocessing"""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = torch.float16 if config.dtype == "float16" else torch.float32
    
    @staticmethod
    def resize_and_crop(img_t: torch.Tensor, size: int) -> torch.Tensor:
        """
        Resize and center crop image tensor.
        
        Args:
            img_t: Input tensor of shape (C, H, W) or (B, C, H, W)
            size: Target size for the cropped square
            
        Returns:
            Resized and cropped tensor
        """
        # Resize slightly larger
        img_t = torchvision.transforms.Resize(size + 3)(img_t)
        
        # Center crop
        start_x = (img_t.size(-1) - size) // 2
        start_y = (img_t.size(-2) - size) // 2
        
        if img_t.dim() == 3:  # CHW
            return img_t[:, start_y:start_y + size, start_x:start_x + size]
        elif img_t.dim() == 4:  # BCHW
            return img_t[:, :, start_y:start_y + size, start_x:start_x + size]
        else:
            raise ValueError(f"Unsupported tensor shape: {img_t.shape}")
    
    def load_image(self, path: str) -> torch.Tensor:
        """
        Load single image from path.
        
        Returns:
            Tensor of shape (H, W, C) with values in [0, 255]
        """
        img = imageio.imread(path, pilmode='RGB')
        return torch.tensor(img, device=self.device)
    
    def load_image_batch(self, paths: List[str]) -> List[torch.Tensor]:
        """Load batch of images from paths"""
        return [self.load_image(p) for p in paths]
    
    def preprocess(self, images: List[torch.Tensor]) -> torch.Tensor:
        """
        Preprocess images to [-1, 1] range BCHW format.
        
        Args:
            images: List of tensors, each (H, W, C) with values in [0, 255]
            
        Returns:
            Tensor of shape (B, C, H, W) with values in [-1, 1]
        """
        processed = []
        for img in images:
            # Convert to float and permute: HWC -> CHW
            img = img.float().permute(2, 0, 1)
            img = self.resize_and_crop(img, self.config.image_size)
            processed.append(img)
        
        # Stack and normalize to [-1, 1]
        batch = torch.stack(processed)
        batch = 2.0 * (batch / 255.0) - 1.0
        
        # Convert to target dtype
        if self.dtype == torch.float16:
            batch = batch.half()
        
        return batch
    
    def postprocess(self, img_t: torch.Tensor, do_resize: bool = True) -> np.ndarray:
        """
        Convert from [-1, 1] BCHW to [0, 255] BHWC uint8.
        
        Args:
            img_t: Tensor of shape (B, C, H, W) with values in [-1, 1]
            do_resize: Whether to resize to config.image_size
            
        Returns:
            NumPy array of shape (B, H, W, C) with uint8 values
        """
        if do_resize:
            img_t = self.resize_and_crop(img_t, self.config.image_size)
        
        img_t = (img_t / 2.0 + 0.5).clamp(0, 1) * 255
        img_t = img_t.detach().cpu()
        img_t = img_t.permute(0, 2, 3, 1).float().numpy()
        img_t = img_t.round().astype("uint8")
        return img_t
    
    @staticmethod
    def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        """Convert tensor (H, W, C) or (C, H, W) to PIL Image"""
        if tensor.dim() == 3 and tensor.shape[0] in (1, 3, 4):
            # CHW format
            tensor = tensor.permute(1, 2, 0)
        
        arr = tensor.cpu().numpy().astype(np.uint8)
        return Image.fromarray(arr)


def normalize_batch(batch: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Normalize each element in a pytorch batch to unit norm.
    
    Args:
        batch: Tensor of shape (B, ...) where dim 0 is batch dimension
        epsilon: Small value for numerical stability
        
    Returns:
        Normalized tensor with same shape
    """
    dims_to_normalize = tuple(range(1, batch.dim()))
    norms = torch.norm(batch, p=2, dim=dims_to_normalize, keepdim=True)
    return batch / (norms + epsilon)


def numpy_chunk(arr: np.ndarray, num_chunks: int, axis: int = 0) -> List[np.ndarray]:
    """
    Split a NumPy array into approximately equal chunks.
    
    Args:
        arr: NumPy array to split
        num_chunks: Number of chunks to create
        axis: Axis along which to split
        
    Returns:
        List of NumPy arrays
    """
    chunk_size = arr.shape[axis] // num_chunks
    remainder = arr.shape[axis] % num_chunks
    
    indices = np.cumsum([0] + [
        chunk_size + 1 if i < remainder else chunk_size 
        for i in range(num_chunks)
    ])
    
    return np.split(arr, indices[1:-1], axis=axis)
