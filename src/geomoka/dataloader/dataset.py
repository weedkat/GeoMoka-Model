from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import os
import glob
import rasterio
import warnings
import pandas as pd
from geomoka.dataloader.mask_converter import MaskConverter
import yaml
from rasterio.errors import NotGeoreferencedWarning
from PIL import Image

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

        self.mc = MaskConverter(metadata)

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
            img = src.read() # in (C, H, W) format
        img = np.transpose(img, (1, 2, 0)) # to (H, W, C) format
        return img

    def _load_mask(self, path):
        with rasterio.open(path) as src:
            m = src.read()
        m = np.transpose(m, (1, 2, 0))
        return self.mc.rgb_to_class(m)
    
    def __len__(self):
        return len(self.df)

class GenericDataset(Dataset):
    """
    Generic dataset handler for various image formats (TIF, JPG, PNG, etc.)
    Supports masks in both RGB format and index format.
    """
    def __init__(self, data_csv, root_dir, metadata):
        """
        Args:
            data_csv (str): Path to the CSV file containing image and mask paths.
            root_dir (str): Root directory of the dataset.
            metadata (str): Path to the YAML file defining dataset metadata.
        """
        # Resolve paths relative to PROJECT_ROOT if needed
        data_csv = Path(data_csv)
        if not data_csv.is_absolute():
            data_csv = PROJECT_ROOT / data_csv
        if not Path(metadata).is_absolute():
            metadata = PROJECT_ROOT / metadata
        if not Path(root_dir).is_absolute():
            root_dir = PROJECT_ROOT / root_dir
    
        self.df = pd.read_csv(data_csv)
        self.img_dir = Path(f"{root_dir}/Source")
        self.mask_dir = Path(f"{root_dir}/Target")
        self.mc = MaskConverter(metadata)

    def __getitem__(self, idx):
        assert isinstance(idx, int)

        row = self.df.iloc[idx]
        
        img_path = self.img_dir / row['Source']
        image = self._load_image(img_path)
        mask_path = self.mask_dir / row['Target']
        mask = self._load_mask(mask_path)

        return image, mask
    
    def _load_image(self, path):
        """Load image from various formats (TIF, JPG, PNG, etc.)"""
        path = Path(path)
        ext = path.suffix.lower()
        
        if ext in ['.tif', '.tiff']:
            # Use rasterio for TIFF files
            with rasterio.open(path) as src:
                img = src.read()  # (C, H, W)
                img = np.transpose(img, (1, 2, 0))  # (H, W, C)
        else:
            # Use PIL for common formats (JPG, PNG, etc.)
            img = Image.open(path)
            img = np.array(img)  # Already in (H, W, C) or (H, W)
        
        return img

    def _load_mask(self, path):
        """Load mask in either RGB format or index format"""
        path = Path(path)
        ext = path.suffix.lower()
        
        if ext in ['.tif', '.tiff']:
            # Use rasterio for TIFF files
            with rasterio.open(path) as src:
                m = src.read()  # (C, H, W)
                m = np.transpose(m, (1, 2, 0))  # (H, W, C) or (H, W, 1)
        else:
            # Use PIL for common formats
            m = Image.open(path)
            m = np.array(m)  # (H, W, C) or (H, W)
        
        # Detect if mask is RGB or index format
        if m.ndim == 3 and m.shape[2] == 3:
            # RGB format - convert to class indices
            mask = self.mc.rgb_to_class(m)
        elif m.ndim == 3 and m.shape[2] == 1:
            # Single channel with extra dimension - squeeze it
            mask = m.squeeze(-1).astype(np.int64)
        elif m.ndim == 2:
            # Already in index format
            mask = m.astype(np.int64)
        else:
            raise ValueError(f"Unexpected mask shape: {m.shape}")
        
        return mask
    
    def __len__(self):
        return len(self.df)