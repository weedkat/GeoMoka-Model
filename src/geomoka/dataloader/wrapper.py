from geomoka.dataloader.transform import *
from geomoka.dataloader.transform import TransformsCompose

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

# The wrapper dataset for semi-supervised learning
# The dataset supports three modes: 'train_l', 'train_u', and 'val'
# nsample specifies the number of samples per epoch, semi-supervised learning requries balanced sample between labeled and unlabeled data
class SemiDataset(Dataset):
    def __init__(self, dataset, mode, size, nsample=None):

        self.dataset = dataset
        self.mode = mode
        self.size = size
        n = len(dataset)

        if nsample is None or nsample <= n:
            self.indices = list(range(n))
        else:
            reps = math.ceil(nsample / n)
            self.indices = list(range(n)) * reps
            self.indices = self.indices[:nsample]

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        img, mask = self.dataset[base_idx]
        
        if self.mode == 'val':
            return img, mask
        
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask)

        if self.mode == 'train_u': # for unlabeled data, we do not have mask
            mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8))

        # The augmentation pipeline is resize + crop + hflip
        img, mask = resize(img, mask, (0.5, 2.0))
        ignore_value = 254 if self.mode == 'train_u' else 255
        # ignore_value is used to denote padded region 
        img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)

        if self.mode == 'train_l':
            return normalize(img, mask)

        # For unlabeled data, we generate two strong augmented views and one weak augmented view
        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
            
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        img_s2 = normalize(img_s2)

        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255

        return normalize(img_w), img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2

    
    def __len__(self):
        return len(self.indices)
    

# Dataset wrapper for supervised training with augmentations.
class SupervisedDataset(Dataset):
    def __init__(self, dataset, transform_cfg = None):
        """
        Args:
            dataset: Base dataset (e.g., ISPRSPostdam) with __getitem__ returning (image, mask) in numpy form.
            transform_cfg: List of transform configs. If empty/None, no transforms applied.
        """
        self.dataset = dataset
        self.transform = TransformsCompose(transform_cfg) if transform_cfg else None
    
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
