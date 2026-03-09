from geomoka._core.transform import *
from geomoka._core.transform import TransformsCompose

from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2

class SupervisedDataset(Dataset):
    """
    Dataset wrapper for supervised training with augmentations using albumentations module.
    Args:
        dataset: Base dataset (e.g., GenericDataset) with __getitem__ returning (image, mask) in numpy form.
        transform_cfg: List of transform configs. If empty/None, no transforms applied.
    """
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __getitem__(self, idx):
        img, mask = self.dataset[idx]
        
        # Convert to numpy if needed
        if isinstance(img, Image.Image):
            img = np.array(img)
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
        
        # Ensure uint8 type for albumentations
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        
        # Apply transformations if configured
        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask'].long()
            return img, mask
        else:
            # No transforms - return numpy arrays (inference_evaluate handles conversion)
            return img, mask
    
    def __len__(self):
        return len(self.dataset)
