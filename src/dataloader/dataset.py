from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import os
import glob
import rasterio
from rasterio.errors import NotGeoreferencedWarning
import warnings
import pandas as pd
import dataloader.utils as utils
import yaml

# Suppress NotGeoreferencedWarning from rasterio
warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)

# Get the project root (two levels up from this file: src/dataloader/ISPRSPostdam.py)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# The mask output for standardized dataset is in range [0, num_classes-1], compatible with CrossEntropyLoss
class ISPRSPostdam(Dataset):
    def __init__(self, data_csv, 
                 root_dir=f"{PROJECT_ROOT}/dataset/ISPRS-Postdam", 
                 class_dict=f"{PROJECT_ROOT}/config/ISPRS-Postdam/color.yaml"):
        self.df = pd.read_csv(data_csv)
        self.img_dir = Path(root_dir) / 'patches' / 'Images'
        self.mask_dir = Path(root_dir) / 'patches' / 'Labels'
        
        class_dict = yaml.safe_load(open(str(class_dict), 'r'))
        self.mc = utils.MaskConverter(class_dict)

    def __getitem__(self, idx): 
        assert isinstance(idx, int)

        row = self.df.iloc[idx]
        
        img_path = self.img_dir / row['Source']
        mask_path = self.mask_dir / row['Target']
        image = self._load_image(str(img_path))
        mask = self._load_mask(str(mask_path))

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
    
    def __len__(self):
        return len(self.df)