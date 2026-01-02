from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import os
import glob
import rasterio
import pandas as pd
import dataloader.utils as utils
import yaml

IMG_DIR  = 'dataset/ISPRS-Postdam/patches/Images'
MASK_DIR = 'dataset/ISPRS-Postdam/patches/Labels'
CLASS_DICT = 'dataset/ISPRS-Postdam/color.yaml'

class ISPRSPostdam(Dataset):
    def __init__(self, data_csv, mode=None, transform=None):
        self.df = pd.read_csv(data_csv)
        self.transform = transform
        
        class_dict = yaml.safe_load(open(CLASS_DICT))
        self.mc = utils.MaskConverter(class_dict)
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        assert isinstance(idx, int)

        row = self.df.iloc[idx]

        image = self._load_image(row['Source'])
        mask = self._load_mask(row['Target'])

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

        return image, mask

    def _load_image(self, path):
        with rasterio.open(path) as src:
            img = src.read()
        img = np.transpose(img, (1, 2, 0))
        return img

    def _load_mask(self, path):
        with rasterio.open(path) as src:
            m = src.read()
        m = np.transpose(m, (1, 2, 0))
        return self.mc.rgb_to_class(m)

class SemiDataset(Dataset):
    def __init__(self, dataset, mode, ):
        pass

    def __getitem__(self, idx):
        pass