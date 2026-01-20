import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SupervisedDataset(Dataset):
    """
    Dataset wrapper for supervised training with augmentations.
    Uses albumentations for efficient augmentation pipeline.
    """
    def __init__(self, dataset, mode='train', size=256, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """
        Args:
            dataset: Base dataset (e.g., ISPRSPostdam)
            mode: 'train' or 'val'
            size: Crop size for training
            mean: ImageNet mean for normalization
            std: ImageNet std for normalization
        """
        self.dataset = dataset
        self.mode = mode
        self.size = size
        
        if mode == 'train':
            self.transform = A.Compose([
                A.RandomCrop(height=size, width=size, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.5),
                # A.OneOf([
                #     A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
                #     A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                # ], p=0.5),
                # A.OneOf([
                #     A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                #     A.MedianBlur(blur_limit=5, p=1.0),
                # ], p=0.3),
                # A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])

        else:  # val mode
            self.transform = A.Compose([
                # A.Resize(height=size, width=size),
                # A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ])
    
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
        
        # Apply transformations
        transformed = self.transform(image=img, mask=mask)
        img = transformed['image']
        mask = transformed['mask'].long()
        
        return img, mask
    
    def __len__(self):
        return len(self.dataset)
