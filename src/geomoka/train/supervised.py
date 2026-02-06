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

from dataloader.wrapper import SemiDataset, SupervisedDataset
from geomoka.eval.evaluate import inference_evaluate
from geomoka.losses.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, init_log, generate_model_name
from util.dist_helper import setup_distributed
from geomoka.model.build_model import build_segmentation_model
from geomoka.dataloader.build_dataset import get_dataset
from geomoka.eval.evaluate import evaluate
from dataloader.mask_converter import MaskConverter

def train_fn(model, loader, optimizer, criterion, epoch, cfg, writer=None, logger=None):
    model.train()
    total_loss = AverageMeter()
    
    # Metrics accumulators
    all_preds = []
    all_targets = []

    total_iters = len(loader) * cfg['epochs']

    pbar = tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch}')
    for i, (img, mask) in pbar:
        img, mask = img.cuda(), mask.cuda()
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
        eval_results = evaluate(all_preds, all_targets, cfg['nclass'], ignore_index=cfg.get('ignore_index', 255))
        
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

def train_supervised(args):
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    transform_cfg = yaml.load(open(cfg['transform'], 'r'), Loader=yaml.Loader)
    cfg = {**cfg, **transform_cfg}

    train_ds = get_dataset(cfg['dataset'], cfg['train_split'], root_dir=cfg.get('root_dir'), metadata=cfg.get('metadata'))
    val_ds = get_dataset(cfg['dataset'], cfg['val_split'], root_dir=cfg.get('root_dir'), metadata=cfg.get('metadata'))

    model_name = generate_model_name(cfg)
    save_path = os.path.join(args.save_path, model_name)
    os.makedirs(save_path, exist_ok=True)

    # ====================== logger =========================
    logger = init_log(
        name='global',
        level=logging.INFO,
        log_file=os.path.join(save_path, 'training.log'),
        add_console=False,
        rank_filter=True,
    )

    all_args = {**cfg, 'ngpus': 1, 'model_name': model_name, 'transform': transform_cfg}

    # Save config file
    yaml.dump(all_args, open(os.path.join(save_path, 'config.yaml'), 'w'))
    
    # Print config to console
    print('\n' + '='*80)
    print('Training Configuration:')
    print('='*80)
    print(pprint.pformat(all_args))
    print('='*80 + '\n')
    
    logger.info('{}\n'.format(pprint.pformat(all_args)))
    writer = SummaryWriter(save_path)

    # ====================== load model, optimizer, criterion, loader =========================
    cudnn.enabled = True
    cudnn.benchmark = True

    # Build model - use smp if specified, otherwise DPT
    model = build_segmentation_model(
        model_name=cfg['model'],
        backbone=cfg['backbone'],
        in_channels=cfg['in_channels'],
        nclass=cfg['nclass'],
        pretrained=cfg.get('pretrained', True),
        lock_backbone=cfg.get('lock_backbone', False)
    ).cuda()
    
    # Setup optimizer with different learning rates for encoder and decoder
    if hasattr(model, 'encoder'):  # smp models
        encoder_params = model.encoder.parameters()
        decoder_params = [p for n, p in model.named_parameters() if 'encoder' not in n]
    elif hasattr(model, 'backbone'):  # DPT models
        encoder_params = model.backbone.parameters()
        decoder_params = [p for n, p in model.named_parameters() if 'backbone' not in n]
    else:
        raise ValueError('Model does not have encoder/backbone attribute for optimizer setup.')
    
    optimizer = AdamW(
        [
            {'params': [p for p in encoder_params if p.requires_grad], 'lr': cfg['lr']},
            {'params': [p for p in decoder_params if p.requires_grad], 'lr': cfg['lr'] * cfg['lr_multi']}
        ], 
        lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01
    )

    print(f'Total params: {count_params(model):.1f}M\n')
    logger.info(f'Total params: {count_params(model):.1f}M\n')

    # Define loss criterion
    if cfg['criterion']['name'] == 'CELoss':
        criterion = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda()
    elif cfg['criterion']['name'] == 'OHEM':
        criterion = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda()
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])
    
    # Data loaders
    trainset = SupervisedDataset(train_ds, cfg['train'])
    valset = SupervisedDataset(val_ds, [])  # No transforms - inference_evaluate applies them
    print('Using SupervisedDataset with albumentations')

    # trainset = SemiDataset(train_ds, mode='train_l', size=cfg['crop_size'], nsample=cfg.get('nsample'))
    # valset = SemiDataset(val_ds, mode='val', size=cfg['crop_size'], nsample=None)
    # print('✓ Using SemiDataset')

    trainloader = DataLoader(
        trainset, batch_size=cfg['batch_size'], num_workers=cfg.get('num_workers_train', 4), pin_memory=cfg.get('pin_memory', True), 
        prefetch_factor=cfg.get('prefetch_factor', 2), persistent_workers=cfg.get('persistent_workers', True), drop_last=True
    )
    valloader = DataLoader(
        valset, batch_size=cfg['batch_size'], num_workers=cfg.get('num_workers_val', 2), pin_memory=cfg.get('pin_memory', True),
        prefetch_factor=cfg.get('prefetch_factor', 2), persistent_workers=cfg.get('persistent_workers', True), drop_last=False
    )

    class_dict = train_ds.mc.class_dict if hasattr(train_ds, 'mc') else None

    # ====================== Train =====================================================

    previous_best = 0.0
    start_epoch = 1
    best_metrics = {}
    
    if os.path.exists(os.path.join(save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(save_path, 'latest.pth'), map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        previous_best = checkpoint['previous_best']
        
        logger.info('************ Load from checkpoint at epoch %i\n' % start_epoch)

        
    for epoch in range(start_epoch, cfg['epochs'] + 1):
        train_fn(model, trainloader, optimizer, criterion, epoch, cfg, writer=writer, logger=logger)
        
        # Free memory and sync before validation
        model.eval()  # Explicitly set to eval mode
        torch.cuda.empty_cache()
        
        # ========================= Evaluate ======================
        eval_results = inference_evaluate(
            model=model,
            dataloader=valloader,
            num_classes=cfg['nclass'],
            ignore_index=cfg.get('ignore_index', 255),
            mode=cfg.get('eval_mode', 'resize'),
            patch_size=cfg.get('crop_size', None),
            class_dict=class_dict,
            device='cuda',
            verbose=True,
            logger=logger,
            transform_cfg=cfg.get('inference', [])
        )
        
        # Log to tensorboard
        miou = eval_results['miou']
        mean_dice = eval_results['mean_dice']
        micro_acc = eval_results['micro_accuracy']
        macro_acc = eval_results['macro_accuracy']
        
        writer.add_scalar('eval/mIoU', miou, epoch)
        writer.add_scalar('eval/mean_dice', mean_dice, epoch)
        writer.add_scalar('eval/micro_accuracy', micro_acc, epoch)
        writer.add_scalar('eval/macro_accuracy', macro_acc, epoch)
        
        # Log per-class metrics to tensorboard
        for class_name, iou_val in eval_results['per_class_iou'].items():
            if not np.isnan(iou_val):
                writer.add_scalar(f'eval/per_class_iou/{class_name}', iou_val, epoch)
        
        for class_name, dice_val in eval_results['per_class_dice'].items():
            if not np.isnan(dice_val):
                writer.add_scalar(f'eval/per_class_dice/{class_name}', dice_val, epoch)
        
        # ========================= Save Model ============================
        is_best = miou > previous_best
        previous_best = max(miou, previous_best)
        
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'previous_best': previous_best,
            'model_name': model_name,
        }

        torch.save(checkpoint, os.path.join(save_path, 'latest.pth'))

        if is_best:
            torch.save(checkpoint, os.path.join(save_path, 'best.pth'))
            print(f'New best model saved with mIoU: {miou:.2f}%\n')
            logger.info(f'New best model saved with mIoU: {miou:.2f}%\n')
        
        # Always track last validation results for final hparams logging
        best_metrics = eval_results.copy()
    
    # ========================= Finalize Tensorboard ============================
    hparams = {
        'lr': cfg['lr'],
        'batch_size': cfg['batch_size'],
        'optimizer': cfg.get('optimizer', 'AdamW'),
    }

    writer.add_hparams(hparam_dict=hparams, metric_dict={
        'best_mIoU': best_metrics.get('miou', 0.0),
        'best_mean_dice': best_metrics.get('mean_dice', 0.0),
        'best_micro_accuracy': best_metrics.get('micro_accuracy', 0.0),
        'best_macro_accuracy': best_metrics.get('macro_accuracy', 0.0),
    })

    writer.close()

    return model

def test(model, testloader, cfg, logger=None):
    model.eval()
    eval_results = inference_evaluate(
        model=model,
        dataloader=testloader,
        num_classes=cfg['nclass'],
        ignore_index=cfg.get('ignore_index', 255),
        mode=cfg.get('eval_mode', 'resize'),
        class_names=None,
        device='cuda',
        verbose=True,
        logger=logger,
        transform_config=cfg.get('inference', [])
    )

    return eval_results
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config file path')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--save_path', type=str, default='checkpoints', help='path to save checkpoints and logs')
    args = parser.parse_args()
    
    model = train_supervised(args)

    print('Training completed.')