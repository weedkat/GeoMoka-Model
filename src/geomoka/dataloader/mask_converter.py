import numpy as np
import yaml
from pathlib import Path    


class MaskConverter:
    """ 
    Handles conversion between RGB masks and class index masks based on provided metadata and vice versa.

    Args:
        metadata: Dict or path to YAML file containing 'class_dict'
    """
    def __init__(self, metadata):
        if isinstance(metadata, (str, Path)):
            metadata = yaml.load(open(metadata, 'r'), Loader=yaml.Loader)
        
        self.class_dict = metadata["class_dict"]
    
    def rgb_to_class(self, mask_rgb):
        """ 
        Input is in (H, W, 3) format and returns (H, W) format.
        """
        if not isinstance(mask_rgb, np.ndarray):
            mask_rgb = np.array(mask_rgb)

        h, w, _ = mask_rgb.shape
        target = np.zeros((h, w), dtype=np.int64)

        for class_idx, item in self.class_dict.items():
            match = np.all(mask_rgb == item['rgb'], axis=-1)
            target[match] = class_idx

        return target
    
    def class_to_rgb(self, mask_class):
        """ 
        Input is in (H, W) format and output is in (H, W, 3) format
        """
        if not isinstance(mask_class, np.ndarray):
            mask_class = np.array(mask_class)

        h, w = mask_class.shape
        mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)

        for class_idx, item in self.class_dict.items():
            rgb = np.array(item["rgb"], dtype=np.uint8)
            mask_rgb[mask_class == class_idx] = rgb

        return mask_rgb
    
    def get_class_dict(self):
        return self.class_dict
