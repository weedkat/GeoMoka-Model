import argparse
import logging
import os
import pprint

import torch
import numpy as np
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm

from geomoka.eval.evaluate import inference_evaluate
from geomoka.util.utils import count_params, AverageMeter, init_log, generate_model_name
from geomoka.eval.evaluate import evaluate
from geomoka.dataloader.mask_converter import MaskConverter
from geomoka.eval.evaluate import inference_evaluate
from geomoka.segmentation.semantic.method.validators import validate_supervised_inputs


import torch
import pprint

from torch import nn
from geomoka.losses.ohem import ProbOhemCrossEntropy2d
from geomoka.model.segmentation.base import SegmentationModel
from geomoka.dataloader.build import get_dataset
from geomoka.dataloader.wrapper import SupervisedDataset
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Any, Optional, Union
from pathlib import Path
import yaml
from geomoka.util.utils import count_params, init_log, generate_model_name
import logging
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from geomoka.train.supervised import run_supervised
from torch.utils.data import Dataset

from geomoka._core.base import BaseTrainer
from geomoka._core.mixins import LoggingMixin
from geomoka._core.registry import Registry

@Registry.register_method
class Supervised(BaseTrainer, LoggingMixin):
    """
    Supervised training method for semantic segmentation.
    """
    def __init__(self, model:SegmentationModel):
        self.model = model
        self.validate_inputs()

    def validate_init(self):
        # Validate that required attributes are set for supervised training
        attrs = ['model', 'optimizer', 'loss', 'transform_cfg']
        for attr in attrs:
            if not hasattr(self, attr):
                raise ValueError(f"Missing required attribute '{attr}' in Supervised trainer initialization.")
            if getattr(self, attr) is None:
                raise ValueError(f"Attribute '{attr}' cannot be None in Supervised trainer initialization.")

        # Optimizer and loss
        optimizer = self.optimizer
        if isinstance(optimizer, dict):
            if 'name' not in optimizer:
                raise ValueError("Optimizer config dict must contain 'name' key.")
        
        loss = self.loss
        if isinstance(loss, dict):
            if 'name' not in loss:
                raise ValueError("Loss config dict must contain 'name' key.")

    def validate_run(self):
        attrs = ['train_ds', 'val_ds', 'epochs']
        for attr in attrs:
            if not hasattr(self, attr):
                raise ValueError(f"Missing required attribute '{attr}' in Supervised trainer before run.")
            if getattr(self, attr) is None:
                raise ValueError(f"Attribute '{attr}' cannot be None in Supervised trainer before run.")

    def save_checkpoint(self, save_dir: str | Path):
        pass

    def load_checkpoint(self, checkpoint_path: str | Path):
        pass

    def run(self):
        """Run the supervised training loop."""
        self.validate_run()
        run_supervised(self.trainer)


class SegmentationTrainer:
    """
    Trainer wrapper for model training.
    """
    
    def __init__(
        self,
        model:SegmentationModel=None,
        train_ds:Dataset=None,
        train_ul:Dataset=None,
        val_ds:Dataset=None,
        epochs:int=None,
        resume:bool=True,
        method :str=None,
        eval_mode:str=None,
    ):
        self.model = model
        self.train_ds = train_ds
        self.train_ul = train_ul
        self.val_ds = val_ds
        self.epochs = epochs
        self.method = method
        self.eval_mode = eval_mode

        self.curr_epoch = 1
        self.prev_best = 0.0

        if resume:
            self.load_checkpoint_if_exists()

    def load_checkpoint_if_exists(self):
        save_path = Path('.checkpoints') / self.model.model_id
        if save_path.exists():
            self.load_checkpoint(save_path)
        else:
            logging.info(f"No checkpoint found at {save_path}")

    def save_checkpoint(self, state):
        save_path = Path('.checkpoints') / self.model.model_id / 'latest.pth'
        torch.save(state, save_path)
        logging.info(f"Checkpoint saved to {save_path}")

    def load_checkpoint(self, save_dir='.checkpoints'):
        """Load model checkpoint."""
        load_path = Path(save_dir) / self.model.model_id / 'latest.pth'
        state = torch.load(load_path, map_location=self.model.device)
        self.logger = init_log(
            name='global',
            level=logging.INFO,
            log_file=str(load_path / 'training.log'),
            add_console=False,
            rank_filter=True,
        )
        self.writer = SummaryWriter(str(load_path.parent))

        return state

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
    


def run_supervised(trainer) -> None:
    """
    Supervised training loop driven by SegmentationTrainer.
    """
    validate_supervised_inputs(trainer)

    cudnn.enabled = True
    cudnn.benchmark = True

    epochs = trainer.train_cfg['epochs']
    previous_best = 0.0
    start_epoch = 1

    # Resume from checkpoint if exists
    latest_checkpoint = trainer.save_path / 'latest.pth'
    if latest_checkpoint.exists():
        checkpoint = torch.load(latest_checkpoint, map_location='cpu')
        trainer.model.load_state_dict(checkpoint['model'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        previous_best = checkpoint.get('previous_best', 0.0)
        trainer.logger.info(f'Resumed from checkpoint at epoch {start_epoch}\n')

    # Training loop
    for epoch in range(start_epoch, epochs + 1):
        train_fn(
            model=trainer.model.model,
            loader=trainer.trainloader,
            optimizer=trainer.optimizer,
            criterion=trainer.criterion,
            epoch=epoch,
            cfg=trainer.train_cfg,
            class_dict=trainer.class_dict,
            writer=trainer.writer,
            logger=trainer.logger,
        )

        # Validation
        trainer.model.model.eval()
        torch.cuda.empty_cache()

        eval_results = inference_evaluate(
            model=trainer.model.model,
            dataloader=trainer.valloader,
            ignore_index=trainer.train_cfg['ignore_index'],
            mode=trainer.train_cfg.get('eval_mode', 'resize'),
            patch_size=trainer.model.model_cfg['crop_size'],
            class_dict=trainer.class_dict,
            device=str(trainer.device),
            verbose=True,
            logger=trainer.logger,
            transform_cfg=trainer.model.transform_cfg.get('inference', []),
        )

        # Log metrics
        miou = eval_results['miou']
        trainer.writer.add_scalar('eval/mIoU', miou, epoch)
        trainer.writer.add_scalar('eval/mean_dice', eval_results['mean_dice'], epoch)
        trainer.writer.add_scalar('eval/micro_accuracy', eval_results['micro_accuracy'], epoch)
        trainer.writer.add_scalar('eval/macro_accuracy', eval_results['macro_accuracy'], epoch)

        # Log per-class metrics to tensorboard
        for class_name, iou_val in eval_results['per_class_iou'].items():
            if not np.isnan(iou_val):
                trainer.writer.add_scalar(f'eval/per_class_iou/{class_name}', iou_val, epoch)
        
        for class_name, dice_val in eval_results['per_class_dice'].items():
            if not np.isnan(dice_val):
                trainer.writer.add_scalar(f'eval/per_class_dice/{class_name}', dice_val, epoch)
        
        # Save checkpoints
        is_best = miou > previous_best
        previous_best = max(miou, previous_best)

        checkpoint = {
            'model': trainer.model.model.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
            'epoch': epoch,
            'previous_best': previous_best,
        }

        torch.save(checkpoint, trainer.save_path / 'latest.pth')

        if is_best:
            torch.save(checkpoint, trainer.save_path / 'best.pth')
            trainer.model.save(trainer.save_path / 'best_model.pth')
            print(f'New best model saved with mIoU: {miou:.2f}%\n')
            trainer.logger.info(f'New best model saved with mIoU: {miou:.2f}%\n')

    # Finalize
    trainer.writer.close()
    trainer.logger.info('Training completed.')

def train_fn(model, loader, optimizer, criterion, epoch, cfg, class_dict, writer=None, logger=None):
    model.train()
    total_loss = AverageMeter()
    device = next(model.parameters()).device
    
    # Metrics accumulators
    all_preds = []
    all_targets = []

    total_iters = len(loader) * cfg['epochs']

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch}')
    for i, (img, mask) in pbar:
        img, mask = img.to(device), mask.to(device)
        pred = model(img)
        loss = criterion(pred, mask)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss.update(loss.item())
        
        # Accumulate predictions and targets for metrics calculation
        with torch.no_grad():
            pred_labels = torch.argmax(pred, dim=1)
            all_preds.append(pred_labels.cpu().numpy())
            all_targets.append(mask.cpu().numpy())

        # Decay learning rate using zero-based global step and clamped base
        global_step = (epoch - 1) * len(loader) + i
        decay_base = max(0.0, 1.0 - (global_step / float(total_iters)))
        lr = float(cfg['lr']) * (decay_base ** 0.9)
        optimizer.param_groups[0]["lr"] = lr
        optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{total_loss.avg:.3f}', 'lr': f'{lr:.6f}'})
        
        if writer:
            writer.add_scalar('train/loss_all', loss.item(), global_step)
            writer.add_scalar('train/loss_x', loss.item(), global_step)
            # Log learning rates for encoder and decoder
            writer.add_scalar('train/lr_encoder', optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar('train/lr_decoder', optimizer.param_groups[1]["lr"], global_step)
            
        if logger and (i % max(1, len(loader) // 8) == 0):
            logger.info(f'Iters: {global_step}, Total loss: {total_loss.avg:.3f}, LR_enc: {optimizer.param_groups[0]["lr"]:.6f}, LR_dec: {optimizer.param_groups[1]["lr"]:.6f}')
    
    # Calculate and log training metrics
    if writer and all_preds and all_targets:
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Compute all metrics using evaluate function
        eval_results = evaluate(
            all_preds,
            all_targets,
            class_dict=class_dict,
            ignore_index=cfg['ignore_index']
        )
        
        # Log aggregate metrics to tensorboard
        writer.add_scalar('train/mIoU', eval_results['miou'], epoch)
        writer.add_scalar('train/mean_dice', eval_results['mean_dice'], epoch)
        writer.add_scalar('train/micro_accuracy', eval_results['micro_accuracy'], epoch)
        writer.add_scalar('train/macro_accuracy', eval_results['macro_accuracy'], epoch)
        
        # Log per-class metrics to tensorboard
        for class_name, iou_val in eval_results['per_class_iou'].items():
            if not np.isnan(iou_val):
                writer.add_scalar(f'train/per_class_iou/{class_name}', iou_val, epoch)
        
        for class_name, dice_val in eval_results['per_class_dice'].items():
            if not np.isnan(dice_val):
                writer.add_scalar(f'train/per_class_dice/{class_name}', dice_val, epoch)
        
        if logger:
            logger.info(
                f'Epoch [{epoch}] Train Metrics - '
                f'mIoU: {eval_results["miou"]:.2f}% | '
                f'Dice: {eval_results["mean_dice"]:.2f}% | '
                f'Acc: {eval_results["micro_accuracy"]:.2f}% | '
                f'Mean Class Acc: {eval_results["macro_accuracy"]:.2f}%'
            )
        
        # Clear memory
        del all_preds, all_targets, eval_results
    
    # Ensure all GPU operations are complete
    torch.cuda.synchronize()
    torch.cuda.empty_cache()