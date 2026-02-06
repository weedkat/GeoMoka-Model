"""
Model and Trainer wrappers for clean separation of concerns.
- SegmentationModel: Handles loading, inference, and metadata management
- SegmentationTrainer: Handles training logic (separate from model)
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import yaml
import numpy as np

from geomoka.dataloader.transform import TransformsCompose
from geomoka.model.build_model import build_segmentation_model
from geomoka.dataloader.mask_converter import MaskConverter


class SegmentationModel(nn.Module):
    """
    Unified model wrapper for loading, inference, and metadata management.
    
    Single responsibility: Model loading and inference only.
    Training is handled by SegmentationTrainer.
    
    Attributes:
        model: PyTorch segmentation model
        transform: Albumentations transform pipeline
        metadata: Dict containing model configuration and info
        device: torch device
    """
    def __init__(
        self,
        model_cfg: dict,
        transform_cfg: dict,
        metadata: Dict[str, Any],
        device: str = 'auto',
    ):
        """
        Initialize model wrapper.
        
        Args:
            model_cfg: Configuration for building the model
            transform_cfg: Configuration for input transformations
            metadata: Metadata class dictionary (MaskConverter compatible)
            device: 'auto', 'cuda', or 'cpu'
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.transform_cfg = transform_cfg
        self.model = build_segmentation_model(**model_cfg)
        self.metadata = metadata
        self.device = self._setup_device(device)
        
        # Class interpreter
        self.mc = MaskConverter(metadata)
        
        # Initialize transforms
        self.transform = TransformsCompose(transform_cfg) if transform_cfg else None
        
        # Move model to device
        self.to(self.device)
        self.eval()
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)
    
    @classmethod
    def from_pth(cls, pth_path: Union[str, Path]) -> 'SegmentationModel':
        """
        Load model from checkpoint.
        
        Args:
            pth_path: Path to model checkpoint (.pth)
        """
        pth_path = Path(pth_path)
        state = torch.load(pth_path, map_location='cpu')
        
        model_cfg = state['model_cfg']
        transform_cfg = state['transform_cfg']
        metadata = state['metadata']
        
        model_wrapper = cls(
            model_cfg=model_cfg,
            transform_cfg=transform_cfg,
            metadata=metadata,
        )
        model_wrapper.load_state_dict(state['model_state_dict'])
        
        return model_wrapper
    
    def load_state_dict(self, state_dict, strict: bool = True):
        return self.model.load_state_dict(state_dict, strict=strict)
    
    def save(self, save_path) -> None:
        """
        Save model checkpoint and metadata.
        
        Args:
            save_path: Path to save model checkpoint
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'model_state_dict': self.model.state_dict(),
            'model_cfg': self.model_cfg,
            'transform_cfg': self.transform_cfg,
            'metadata': self.metadata,
        }

        torch.save(state, save_path)
    
    def get_class_dict(self) -> Dict[int, str]:
        """ Get class dictionary from metadata. """
        return self.mc.get_class_dict()
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup torch device."""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.device(device)


