from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import os
import glob
import rasterio
import warnings
import pandas as pd
import dataloader.mask_converter as mask_converter
import yaml
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)

PROJECT_ROOT = Path(__file__).parent.parent.parent

# The mask output for standardized dataset is in range [0, num_classes-1], compatible with torch criterion module
class ISPRSPostdam(Dataset):
    def __init__(self, data_csv, root_dir, metadata):
        """
        Args:
            data_csv (str): Path to the CSV file containing image and mask paths.
            root_dir (str): Root directory of the dataset.
            metadata (str): Path to the YAML file defining dataset metadata.
        """
        # Resolve data_csv relative to PROJECT_ROOT if it's a relative path
        data_csv = Path(data_csv)
        if not data_csv.is_absolute():
            data_csv = PROJECT_ROOT / data_csv
        if not Path(metadata).is_absolute():
            metadata = PROJECT_ROOT / metadata
        if not Path(root_dir).is_absolute():
            root_dir = PROJECT_ROOT / root_dir
    
        self.df = pd.read_csv(data_csv)
        self.img_dir = Path(f"{root_dir}/patches/Images")
        self.mask_dir = Path(f"{root_dir}/patches/Labels")

        self.mc = mask_converter.MaskConverter(metadata)

    def __getitem__(self, idx): 
        assert isinstance(idx, int)

        row = self.df.iloc[idx]
        img_path = Path(f"{self.img_dir}/{row['Source']}")
        mask_path = Path(f"{self.mask_dir}/{row['Target']}")
        image = self._load_image(img_path)
        mask = self._load_mask(mask_path)

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