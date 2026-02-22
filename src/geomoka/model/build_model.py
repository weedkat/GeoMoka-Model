import os
import torch
from pathlib import Path
from geomoka.model.semseg.dpt import DPT

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    print("Warning: segmentation_models_pytorch not installed")

# Use cache directory for pretrained weights
# Works for both dev and pip-installed versions
PRETRAINED_DIR = Path.home() / '.cache' / 'geomoka' / 'pretrained'

# Map model names to smp classes
dpt_map = {
    'dpt_dinov2_small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'dpt_dinov2_base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'dpt_dinov2_large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'dpt_dinov2_giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}

smp_map = {
    'unet': smp.Unet,
    'unet++': smp.UnetPlusPlus,
    'unetplusplus': smp.UnetPlusPlus,
    'deeplabv3': smp.DeepLabV3,
    'deeplabv3+': smp.DeepLabV3Plus,
    'deeplabv3plus': smp.DeepLabV3Plus,
    'fpn': smp.FPN,
    'pspnet': smp.PSPNet,
    'pan': smp.PAN,
    'linknet': smp.Linknet,
    'manet': smp.MAnet,
}

def download_pretrained_dinov2(model, pretrain_dir=PRETRAINED_DIR):
    urls ={
        'dinov2_small': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth',
        'dinov2_base': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth',
        'dinov2_large': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth',
        'dinov2_giant': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth'
    }

    extract_dir = pretrain_dir
    os.makedirs(extract_dir, exist_ok=True)

    if model not in urls:
        raise ValueError(f"{model} not recognized. Valid options are: {list(urls.keys())}")

    url = urls[model]
    pth_path = os.path.join(extract_dir, f"{model}.pth")

    if not os.path.exists(pth_path):
        print(f"Downloading pretrained DINOv2 {model} weights...")
        torch.hub.download_url_to_file(url, pth_path)
        print("Download complete.")
    else:
        print(f"Pretrained DINOv2 {model} weights already exists at {pth_path}. Skipping download.")

def build_dpt_model(model, in_channels, nclass, pretrain, lock_encoder=True, pretrain_dir=PRETRAINED_DIR, **kwargs):
    model_cfg = dpt_map[model]
    model = DPT(**{**model_cfg, 'nclass': nclass, 'in_chans': in_channels})
    
    if pretrain:
        backbone = model_cfg['encoder_size']
        backbone_name = f'dinov2_{backbone}'
        path = os.path.join(pretrain_dir, f'{backbone_name}.pth')
        download_pretrained_dinov2(backbone_name, pretrain_dir)
        state_dict = torch.load(path, map_location='cpu')
        model.backbone.load_state_dict(state_dict)
        print(f'Pretrained DINOv2 weights loaded from {path}')

    else:
        print(f'No pretrained weights, training from scratch')
    
    if lock_encoder:
        model.lock_backbone()
        print('Encoder frozen')
    
    return model


def build_smp_model(model, in_channels, nclass, pretrain, lock_encoder, **kwargs):
    if not SMP_AVAILABLE:
        raise ImportError("segmentation_models_pytorch is required for non-DPT models. Install with: pip install segmentation-models-pytorch")
    
    # Build model
    if not pretrain:
        kwargs['encoder_weights'] = None  # Disable pretrained weights if pretrain=False

    model = smp_map[model](
        in_channels=in_channels,
        classes=nclass,
        **kwargs
    )
    
    if lock_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        print('Encoder frozen')

    return model


def build_segmentation_model(model, in_channels, nclass, pretrain, lock_encoder, **kwargs):
    model_name = model.lower()

    if model_name in dpt_map:
        model = build_dpt_model(model_name, in_channels, nclass, pretrain, lock_encoder, **kwargs)
    elif model_name in smp_map:
        model = build_smp_model(model_name, in_channels, nclass, pretrain, lock_encoder, **kwargs)    
    else:
        raise ValueError(f"Model {model_name} is not recognized as either DPT or SMP model.")

    print(f'Built {model_name} model with {in_channels} input channels and {nclass} classes with lock_encoder={lock_encoder}')
    
    return model
