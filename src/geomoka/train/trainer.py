import torch
from geomoka.train.supervised import train_supervised
from geomoka.model.wrapper import SegmentationModel
from typing import Dict, Any, Optional, Union
from pathlib import Path
import yaml

class SegmentationTrainer:
    """
    Trainer wrapper for model training.
    """
    
    def __init__(self):
        """
        Initialize trainer.
        
        Args:
            model_wrapper: SegmentationModel instance
            optimizer: PyTorch optimizer
            criterion: Loss function
            scheduler: Learning rate scheduler (optional)
        """

    @classmethod
    def from_config(cls, config_path):
        cfg = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
        model_cfg = cfg['model_cfg']
        transform_cfg = cfg['transform_cfg']
        metadata = yaml.load(open(cfg['metadata'], 'r'), Loader=yaml.Loader)

    def run(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        num_epochs: int = 100,
        save_path: Optional[Union[str, Path]] = None,
    ):
        """
        Run training loop.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            num_epochs: Number of training epochs
            save_path: Path to save model checkpoints (optional)
        """
        train_supervised(
            model=self.model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=self.optimizer,
            criterion=self.criterion,
            scheduler=self.scheduler,
            device=self.device,
            num_epochs=num_epochs,
            save_path=save_path,
            model_wrapper=self.model_wrapper,
        )