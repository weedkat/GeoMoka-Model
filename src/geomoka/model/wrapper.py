"""
Model and Trainer wrappers for clean separation of concerns.
- SegmentationModel: Handles loading, inference, and metadata management
- SegmentationTrainer: Handles training logic (separate from model)
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple

from geomoka.model.build_model import build_segmentation_model
from geomoka.dataloader.mask_converter import MaskConverter
from geomoka.inference.engine import SegmentationInference


class SegmentationModel:
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
        self.model = build_segmentation_model(model=model_cfg['model'],
                                              in_channels=model_cfg['in_channels'],
                                              nclass=len(metadata['class_dict']),
                                              lock_backbone=model_cfg.get('lock_backbone', False),
                                              model_kwargs=model_cfg.get('model_kwargs', {}))
        self.transform_cfg = transform_cfg
        self.metadata = metadata
        
        # Class interpreter
        self.mc = MaskConverter(metadata)
        self.nclass = len(self.mc.get_class_dict())

        # Device
        self.device = self._setup_device(device)
        self.model.to(self.device)

        # Inferencer
        self.inferencer = SegmentationInference(
            model=self.model,
            patch_size=model_cfg['crop_size'],
            overlap_ratio=0.5,
            device=self.device,
            transform_cfg=transform_cfg.get('inference', []),
            reject_class=metadata.get('ignore_index', 255),
        )
    
    def __call__(self, x):
        return self.model(x)

    def predict(self, images, **kwargs) -> torch.Tensor:
        return self.inferencer(images, **kwargs)
    
    @classmethod
    def load(cls, pth_path: Union[str, Path]) -> 'SegmentationModel':
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
    
    def get_encoder_decoder_params(self):
        model = self.model
        if hasattr(model, 'encoder'):  # smp models
            encoder_params = model.encoder.parameters()
            decoder_params = [p for n, p in model.named_parameters() if 'encoder' not in n]
        elif hasattr(model, 'backbone'):  # DPT models
            encoder_params = model.backbone.parameters()
            decoder_params = [p for n, p in model.named_parameters() if 'backbone' not in n]
        else:
            raise ValueError('Model does not have encoder/backbone attribute for optimizer setup.')
        
        return encoder_params, decoder_params
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup torch device."""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.device(device)


