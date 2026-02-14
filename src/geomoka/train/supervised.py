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

from geomoka.dataloader.wrapper import SemiDataset, SupervisedDataset
from geomoka.eval.evaluate import inference_evaluate
from geomoka.losses.ohem import ProbOhemCrossEntropy2d
from geomoka.util.utils import count_params, AverageMeter, init_log, generate_model_name
from geomoka.util.dist_helper import setup_distributed
from geomoka.model.build_model import build_segmentation_model
from geomoka.dataloader.build_dataset import get_dataset
from geomoka.eval.evaluate import evaluate
from geomoka.dataloader.mask_converter import MaskConverter
from geomoka.eval.evaluate import inference_evaluate


def run_supervised(trainer) -> None:
    """
    Supervised training loop driven by SegmentationTrainer.
    """
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
    effective_cfg = {
        **trainer.train_cfg,
    }
    for epoch in range(start_epoch, epochs + 1):
        train_fn(
            model=trainer.model.model,
            loader=trainer.trainloader,
            optimizer=trainer.optimizer,
            criterion=trainer.criterion,
            epoch=epoch,
            cfg=effective_cfg,
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
            ignore_index=trainer.train_cfg.get('ignore_index', 255),
            mode=trainer.train_cfg.get('eval_mode', 'resize'),
            patch_size=trainer.model.model_cfg.get('crop_size'),
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
            ignore_index=cfg.get('ignore_index', 255)
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