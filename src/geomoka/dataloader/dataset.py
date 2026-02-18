from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
import rasterio
import warnings
import pandas as pd
from geomoka.dataloader.mask_converter import MaskConverter
from rasterio.errors import NotGeoreferencedWarning
from PIL import Image

warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)


class GenericDataset(Dataset):
    """
    Generic dataset handler for various image formats (TIF, JPG, PNG, etc.)
    Supports masks in both RGB format and index format.

    Args:
        data_csv: Path to CSV file (relative to cwd or absolute)
        root_dir: Root directory of dataset (relative to cwd or absolute)
        metadata: Path to YAML metadata (relative to cwd or absolute)
    """
    def __init__(self, data_csv, root_dir, metadata):
        """
        Args:
            data_csv (str): Path to the CSV file containing image and mask paths.
            root_dir (str): Root directory of the dataset.
            metadata (str): Path to the YAML file defining dataset metadata.
        """
        # Resolve relative paths from current working directory
        data_csv = Path(data_csv).resolve()
        root_dir = Path(root_dir).resolve()
        metadata = Path(metadata).resolve()
    
        self.df = pd.read_csv(data_csv)
        self.img_dir = root_dir / "images"
        self.mask_dir = root_dir / "labels"
        self.mc = MaskConverter(metadata)

    def __getitem__(self, idx):
        assert isinstance(idx, int)

        row = self.df.iloc[idx]
        
        img_path = self.img_dir / row['Image']
        image = self._load_image(img_path)
        mask_path = self.mask_dir / row['Label']
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