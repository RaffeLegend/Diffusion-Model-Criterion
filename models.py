"""
Model Manager - handles loading and memory management of all models
"""

import logging
import torch
from typing import Tuple, Optional

from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPModel, AutoImageProcessor, pipeline as pipeline_caption

from config import EvalConfig


class ModelManager:
    """
    Manages model loading and memory.
    Uses lazy loading to only load models when needed.
    """
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = torch.float16 if config.dtype == "float16" else torch.float32
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Models (lazy loading)
        self._pipeline: Optional[StableDiffusionPipeline] = None
        self._unet = None
        self._vae = None
        self._text_encoder = None
        self._tokenizer = None
        self._scheduler = None
        self._clip = None
        self._processor = None
        self._image_to_text = None
    
    @property
    def pipeline(self) -> StableDiffusionPipeline:
        """Get SD pipeline (loads if not loaded)"""
        self.load_sd_pipeline()
        return self._pipeline
    
    @property
    def unet(self):
        self.load_sd_pipeline()
        return self._unet
    
    @property
    def vae(self):
        self.load_sd_pipeline()
        return self._vae
    
    @property
    def text_encoder(self):
        self.load_sd_pipeline()
        return self._text_encoder
    
    @property
    def tokenizer(self):
        self.load_sd_pipeline()
        return self._tokenizer
    
    @property
    def scheduler(self):
        self.load_sd_pipeline()
        return self._scheduler
    
    @property
    def clip(self):
        self.load_clip()
        return self._clip
    
    @property
    def clip_processor(self):
        self.load_clip()
        return self._processor
    
    @property
    def caption_model(self):
        self.load_caption_model()
        return self._image_to_text
        
    def load_sd_pipeline(self):
        """Load Stable Diffusion pipeline"""
        if self._pipeline is None:
            self.logger.info(f"Loading SD pipeline: {self.config.model_name}")
            self._pipeline = StableDiffusionPipeline.from_pretrained(
                self.config.model_name,
                torch_dtype=self.dtype
            )
            self._pipeline.to(self.device)
            
            # Extract components
            self._unet = self._pipeline.unet.eval()
            self._vae = self._pipeline.vae.eval()
            self._text_encoder = self._pipeline.text_encoder.eval()
            self._tokenizer = self._pipeline.tokenizer
            self._scheduler = DDPMScheduler.from_pretrained(
                self.config.model_name,
                subfolder="scheduler"
            )
            
            self.logger.info("SD pipeline loaded successfully")
    
    def load_clip(self):
        """Load CLIP model"""
        if self._clip is None:
            self.logger.info("Loading CLIP model")
            self._clip = CLIPModel.from_pretrained(
                "openai/clip-vit-large-patch14"
            ).to(self.device)
            self._processor = AutoImageProcessor.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
            self.logger.info("CLIP loaded successfully")
    
    def load_caption_model(self):
        """Load image captioning model"""
        if self._image_to_text is None:
            self.logger.info(f"Loading caption model: {self.config.caption_model}")
            self._image_to_text = pipeline_caption(
                "image-text-to-text",
                model=self.config.caption_model,
                device=self.device
            )
            self.logger.info("Caption model loaded successfully")
    
    def clear_cache(self):
        """Clear GPU cache"""
        torch.cuda.empty_cache()
        self.logger.debug("GPU cache cleared")
    
    def unload_all(self):
        """Unload all models to free memory"""
        self._pipeline = None
        self._unet = None
        self._vae = None
        self._text_encoder = None
        self._tokenizer = None
        self._scheduler = None
        self._clip = None
        self._processor = None
        self._image_to_text = None
        self.clear_cache()
        self.logger.info("All models unloaded")
    
    def get_sd_components(self) -> Tuple:
        """Get all SD components as a tuple"""
        self.load_sd_pipeline()
        return (self._unet, self._tokenizer, self._text_encoder, 
                self._vae, self._scheduler)
