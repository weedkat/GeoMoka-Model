import torch
import pprint

from torch import nn
from geomoka.losses.ohem import ProbOhemCrossEntropy2d
from geomoka.model.wrapper import SegmentationModel
from geomoka.dataloader.build_dataset import get_dataset
from geomoka.dataloader.wrapper import SupervisedDataset
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Union
from pathlib import Path
import yaml
from geomoka.util.utils import count_params, init_log, generate_model_name
import logging
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from geomoka.train.supervised import run_supervised

class SegmentationTrainer:
    """
    Trainer wrapper for model training.
    """
    
    def __init__(
        self,
        model: SegmentationModel,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        train_cfg: Dict[str, Any],
        save_path: Union[str, Path],
        writer: SummaryWriter,
        logger: logging.Logger,
    ):
        """
        Initialize trainer.
        
        Args:
            model: SegmentationModel instance
            optimizer: PyTorch optimizer
            criterion: Loss function
            trainloader: Training DataLoader
            valloader: Validation DataLoader
            train_cfg: Training configuration dict
            save_path: Path to save checkpoints and logs
            writer: Tensorboard writer
            logger: Logger instance
        """

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.trainloader = trainloader
        self.valloader = valloader
        self.train_cfg = train_cfg
        self.save_path = Path(save_path)
        self.writer = writer
        self.logger = logger
        self.device = model.device
        self.class_dict = model.mc.get_class_dict(exclude_ignore=True)

    @classmethod
    def from_config(cls, config_path: Union[str, Path], save_dir: str = 'outputs'):
        """
        Load trainer from config YAML file.
        
        Args:
            config_path: Path to config YAML
            save_dir: Base directory for saving outputs
            
        Returns:
            SegmentationTrainer instance
        """
        # Load configuration from YAML file
        cfg = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
        train_cfg = cfg['train_cfg']
        model_cfg = cfg['model_cfg']
        transform_cfg = cfg['transform_cfg']
        
        # Load metadata
        with open(cfg['metadata'], 'r') as f:
            metadata = yaml.safe_load(f)

        # Generate unique model name and save path
        model_name = generate_model_name(cfg)
        save_path = Path(save_dir) / model_name
        save_path.mkdir(parents=True, exist_ok=True)

        # Save full config
        with open(save_path / 'config.yaml', 'w') as f:
            yaml.safe_dump(cfg, f)

        # ====================== Logger & Tensorboard =========================
        logger = init_log(
            name='global',
            level=logging.INFO,
            log_file=str(save_path / 'training.log'),
            add_console=False,
            rank_filter=True,
        )
        
        print('\n' + '='*80)
        print('Training Configuration:')
        print('='*80)
        print(pprint.pformat(cfg))
        print('='*80 + '\n')
        
        logger.info('{}\n'.format(pprint.pformat(cfg)))
        writer = SummaryWriter(str(save_path))

        # ==================== Model ====================

        model = SegmentationModel(model_cfg, transform_cfg, metadata)

        # ==================== Optimizer ====================
        encoder_params, decoder_params = model.get_encoder_decoder_params()
        
        optimizer = AdamW(
            [
                {'params': [p for p in encoder_params if p.requires_grad], 'lr': train_cfg['lr']},
                {'params': [p for p in decoder_params if p.requires_grad], 'lr': train_cfg['lr'] * train_cfg['lr_multi']}
            ], 
            lr=train_cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01
        )

        # ==================== Criterion ====================
        if train_cfg['criterion']['name'] == 'CELoss':
            criterion = nn.CrossEntropyLoss(**train_cfg['criterion']['kwargs']).cuda()
        elif train_cfg['criterion']['name'] == 'OHEM':
            criterion = ProbOhemCrossEntropy2d(**train_cfg['criterion']['kwargs']).cuda()
        else:
            raise NotImplementedError(f"{train_cfg['criterion']['name']} criterion is not implemented")
        
        print(f'Total params: {count_params(model.model):.1f}M\n')
        logger.info(f'Total params: {count_params(model.model):.1f}M\n')

        # ==================== DataLoader ========================
        trainloader, valloader = cls.build_supervised_dataloader(cfg)

        return cls(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            trainloader=trainloader,
            valloader=valloader,
            train_cfg=train_cfg,
            save_path=save_path,
            writer=writer,
            logger=logger
        )
    
    @staticmethod
    def build_supervised_dataloader(cfg):
        """
        Build supervised training and validation dataloaders.
        
        Args:
            cfg: Full configuration dict
            
        Returns:
            (trainloader, valloader) tuple
        """
        transform_cfg = cfg['transform_cfg']
        loader_cfg = cfg['dataloader_cfg']

        train_ds = get_dataset(cfg['dataset'], cfg['train_split'], root_dir=cfg.get('root_dir'), metadata=cfg['metadata'])
        val_ds = get_dataset(cfg['dataset'], cfg['val_split'], root_dir=cfg.get('root_dir'), metadata=cfg['metadata'])
        
        # Wrapper datasets
        trainset = SupervisedDataset(train_ds, transform_cfg['train'])
        # Validation transforms handled by SegmentationInference
        valset = SupervisedDataset(val_ds, [])

        trainloader = DataLoader(
            trainset, 
            batch_size=loader_cfg['batch_size'], 
            num_workers=loader_cfg.get('num_workers_train', 4), 
            pin_memory=loader_cfg.get('pin_memory', True), 
            prefetch_factor=loader_cfg.get('prefetch_factor', 2), 
            persistent_workers=loader_cfg.get('persistent_workers', True), 
            drop_last=True,
            shuffle=True,
        )
        valloader = DataLoader(
            valset, 
            batch_size=loader_cfg['batch_size'], 
            num_workers=loader_cfg.get('num_workers_val', 2), 
            pin_memory=loader_cfg.get('pin_memory', True),
            prefetch_factor=loader_cfg.get('prefetch_factor', 2), 
            persistent_workers=loader_cfg.get('persistent_workers', True), 
            drop_last=False,
        )

        return trainloader, valloader

    def run(self):
        """
        Run training loop.
        
        Dispatches to the appropriate training method (supervised, semi-supervised, etc.)
        based on train_cfg['method'].
        """
        method = self.train_cfg.get('method', 'supervised')
        
        if method == 'supervised':
            run_supervised(self)
        elif method == 'unimatch_v2':
            self._run_unimatch_v2()
        else:
            raise ValueError(f"Unknown training method: {method}")
    
    def _run_unimatch_v2(self):
        """Run UniMatch v2 semi-supervised training loop."""
        raise NotImplementedError("UniMatch v2 training not yet implemented")