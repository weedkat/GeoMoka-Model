import argparse
import logging
import glob
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

from dataloader.semi import SemiDataset
from model.semseg.dpt import DPT
from evaluate import inference_evaluate
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
from util.dist_helper import setup_distributed

def build_vit_model(backbone, nclass, lock_backbone=True, pretrain_dir='../pretrain'):
    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    config = backbone.split('_')[-1]
    assert config in model_configs, f'Unknown backbone config {config}'

    model = DPT(**{**model_configs[config], 'nclass': nclass})

    path = os.path.join(pretrain_dir, f'{backbone}.pth')

    if os.path.exists(path):
        state_dict = torch.load(path)
        model.backbone.load_state_dict(state_dict)
        print('Pretrained weight loaded')

    if lock_backbone:
        model.lock_backbone()
    
    return model


def train_fn(model, loader, optimizer, criterion, epoch, cfg, writer=None, logger=None):
    model.train()
    total_loss = AverageMeter()

    iters = 0
    total_iters = len(loader) * cfg['epochs']

    for i, (img, mask) in enumerate(loader):
        img, mask = img.cuda(), mask.cuda()
        pred = model(img)
        loss = criterion(pred, mask)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss.update(loss.item())

        # Decay learning rate
        iters = epoch * len(loader) + i
        lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
        optimizer.param_groups[0]["lr"] = lr
        optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
        
        if writer:
            writer.add_scalar('train/loss_all', loss.item(), iters)
            writer.add_scalar('train/loss_x', loss.item(), iters)
            
        if logger and (i % (len(loader) // 8) == 0):
            logger.info(f'Iters: {i}, Total loss: {total_loss.avg:.3f}')


def train(args, trainset, valset, eval_mode='original'):
    
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    cfg['batch_size'] *= 2

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    all_args = {**cfg, **vars(args), 'ngpus': 1}
    logger.info('{}\n'.format(pprint.pformat(all_args)))
    writer = SummaryWriter(args.save_path)
    os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    # ====================== load model, optimizer, criterion, loader =========================

    model = build_vit_model(cfg['backbone'], cfg['nclass'])
    
    optimizer = AdamW(
            [
                {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': cfg['lr']},
                {'params': [param for name, param in model.named_parameters() if 'backbone' not in name], 'lr': cfg['lr'] * cfg['lr_multi']}
            ], 
            lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01
        )

    logger.info(f'Total params: {count_params(model):.1f}M\n')

    if cfg['criterion']['name'] == 'CELoss':
        criterion = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda()
    elif cfg['criterion']['name'] == 'OHEM':
        criterion = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda()
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    trainloader = DataLoader(
        trainset, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True
    )
    valloader = DataLoader(
        valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False
    )

    # ====================== Train =====================================================

    previous_best = 0.0
    start_epoch = 0
    
    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'), map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        previous_best = checkpoint['previous_best']
        
        logger.info('************ Load from checkpoint at epoch %i\n' % start_epoch)
    
    for epoch in range(start_epoch, cfg['epochs']):
        train_fn(model, trainloader, optimizer, criterion, epoch, cfg, writer=writer, logger=logger)
        
        # ========================= Evaluate ======================
        eval_results = inference_evaluate(
            model=model,
            dataloader=valloader,
            num_classes=cfg['nclass'],
            ignore_index=255,
            mode='resize',
            class_names=None,
            device='cuda',
            verbose=False
        )
        
        miou = eval_results['miou']
        mean_dice = eval_results['mean_dice']
        overall_acc = eval_results['overall_accuracy']
        
        logger.info(
            f'Epoch [{epoch}/{cfg["epochs"]}] - '
            f'mIoU: {miou:.2f}% | '
            f'Dice: {mean_dice:.2f}% | '
            f'Acc: {overall_acc:.2f}%'
        )
        
        # Log to tensorboard
        writer.add_scalar('eval/mIoU', miou, epoch)
        writer.add_scalar('eval/mean_dice', mean_dice, epoch)
        writer.add_scalar('eval/overall_accuracy', overall_acc, epoch)
        
        # ========================= Save Model ============================
        is_best = miou > previous_best
        previous_best = max(miou, previous_best)
        
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'previous_best': previous_best,
        }

        torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))

        if is_best:
            torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))
            logger.info(f'New best model saved with mIoU: {miou:.2f}%\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config file path')
    parser.add_argument('--save_path', type=str, required=True, help='path to save logs and models')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    args = parser.parse_args()

    # trainset = SemiDataset(split='train')
    # valset = SemiDataset(split='val')

    train(args, trainset, valset, eval_mode='original')