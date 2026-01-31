import os
import torch
from pathlib import Path

from model.semseg.dpt import DPT

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    print("Warning: segmentation_models_pytorch not installed. Only DPT models available.")

PROJECT_ROOT = Path(__file__).parent.parent.parent
PRETRAINED_DIR = PROJECT_ROOT / 'pretrained'

def build_segmentation_model(model_name, 
                             backbone, 
                             in_channels, 
                             nclass, 
                             pretrained=True, 
                             lock_backbone=True, 
                             pretrain_dir=PRETRAINED_DIR):
    """
    Build segmentation model from either smp library or custom DPT.
    
    Args:
        model_name: Model architecture ('unet', 'unet++', 'deeplabv3', 'deeplabv3+', 'fpn', 'pspnet', 'pan', 'linknet', 'manet', 'dpt')
        backbone: Encoder backbone (e.g., 'resnet50', 'efficientnet-b4', 'dinov2_base', etc.)
        nclass: Number of classes
        pretrained: Use ImageNet pretrained weights (for smp models)
        lock_backbone: Freeze encoder weights
        pretrain_dir: Directory for custom pretrained weights (for DPT)
        in_channels: Number of input channels (passed to SMP encoders and DPT backbone)
    
    Returns:
        PyTorch model
    """
    model_name = model_name.lower()
    
    # DPT with DINOv2 backbone
    if model_name == 'dpt' or backbone.startswith('dinov2'):
        model_configs = {
            'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        config = backbone.split('_')[-1] if '_' in backbone else 'base'
        assert config in model_configs, f'Unknown DINOv2 config {config}'
        
        model = DPT(**{**model_configs[config], 'nclass': nclass, 'in_chans': in_channels})
        
        # Load pretrained DINOv2 weights
        path = os.path.join(pretrain_dir, f'{backbone}.pth')
        if os.path.exists(path):
            state_dict = torch.load(path, map_location='cpu')
            model.backbone.load_state_dict(state_dict)
            print(f'Pretrained DINOv2 weights loaded from {path}')
        else:
            print(f'No pretrained weights found at {path}, training from scratch')
        
        if lock_backbone:
            model.lock_backbone()
            print('Backbone frozen')
        
        return model
    
    # SMP models
    if not SMP_AVAILABLE:
        raise ImportError("segmentation_models_pytorch is required for non-DPT models. Install with: pip install segmentation-models-pytorch")
    
    # Map model names to smp classes
    model_map = {
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
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_map.keys())}")
    
    # Build model
    model = model_map[model_name](
        encoder_name=backbone,
        encoder_weights='imagenet' if pretrained else None,
        in_channels=in_channels,
        classes=nclass,
    )
    
    if lock_backbone:
        for param in model.encoder.parameters():
            param.requires_grad = False
        print('Encoder frozen')
    
    print(f'Built {model_name} with {backbone} backbone (pretrained={pretrained})')
    
    return model
