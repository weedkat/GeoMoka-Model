# GeoMoka Model

Geospatial semantic segmentation framework with support for multiple architectures and advanced remote sensing workflows.

## Overview

**GeoMoka Model** is a PyTorch-based framework for semantic segmentation of geospatial and remote sensing imagery. It provides:

- **Multiple segmentation models** via Segmentation Models PyTorch (SMP)
- **DINOv2-based DPT** architecture for vision foundation models
- **Flexible training** with config-driven workflows
- **Dataset tools** for splitting and preprocessing remote sensing data
- **Calibration utilities** for confidence thresholding

📖 **[Full Documentation](https://weedkat.github.io/GeoMoka-Model/)**

## Quickstart

```bash
pip install -r requirements.txt
pip install -e .

# Scaffold a new project
geomoka-scaffold --root .
geomoka-template --root . --config generic

# Train a model
geomoka-train --config config/generic/train_dpt.yaml --save_dir output
```

## Features

### 1. Segmentation Models PyTorch (SMP) Integration

The framework now supports multiple segmentation architectures through the `segmentation-models-pytorch` library:

**Available Models:**
- UNet
- UNet++
- DeepLabV3
- DeepLabV3+
- FPN (Feature Pyramid Network)
- PSPNet
- PAN (Pyramid Attention Network)
- LinkNet
- MAnet
- DPT (original DINOv2-based model)

**Supported Backbones:**
- ResNet (resnet18, resnet34, resnet50, resnet101, resnet152)
- EfficientNet (efficientnet-b0 to efficientnet-b7)
- MobileNet (mobilenet_v2)
- DenseNet (densenet121, densenet169, densenet201)
- VGG (vgg11, vgg13, vgg16, vgg19)
- And many more! See [SMP documentation](https://github.com/qubvel/segmentation_models.pytorch)

### 2. SupervisedDataset with Albumentations

New `SupervisedDataset` class provides rich data augmentation using albumentations:

**Training Augmentations:**
- Random resize crop (scale 0.5-2.0x)
- Horizontal flip (50%)
- Vertical flip (30%)
- Random 90° rotation (50%)
- Color jittering (brightness, contrast, saturation, hue)
- Gaussian/Median blur (30%)
- Gaussian noise (30%)
- ImageNet normalization

**Validation:**
- Resize to crop_size
- ImageNet normalization

## Installation
```bash
pip install -r requirements.txt
```

## Usage

### Basic Configuration

```yaml
# config/ISPRS-Postdam/train.yaml

dataset: ISPRSPostdam
nclass: 6
crop_size: 256
use_albumentations: true  # Enable SupervisedDataset

epochs: 60
batch_size: 8
lr: 0.0001
lr_multi: 10.0

model: unet++  # Choose your model
backbone: efficientnet-b4  # Choose your encoder
pretrained: true  # Use ImageNet pretrained weights
lock_backbone: false  # Train encoder
```

### Training Examples
