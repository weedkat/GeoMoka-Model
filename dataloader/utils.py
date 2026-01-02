import numpy as np

class MaskConverter:
    def __init__(self, class_dict):
        self.class_dict = class_dict

    def rgb_to_class(self, mask_rgb):
        """ Input is in (H, W, C) format
        """
        if not isinstance(mask_rgb, np.ndarray):
            mask_rgb = np.array(mask_rgb)

        h, w, _ = mask_rgb.shape
        target = np.zeros((h, w))

        for class_idx, item in self.class_dict.items():
            match = np.all(mask_rgb == item['rgb'], axis=-1)
            target[match] = class_idx

        return target
    
    def class_to_rgb(self, mask_class):
        if not isinstance(mask_class, np.ndarray):
            mask_class = np.array(mask_class)

        h, w = mask_class.shape
        mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)

        for class_idx, item in self.class_dict.items():
            rgb = np.array(item["rgb"], dtype=np.uint8)
            mask_rgb[mask_class == class_idx] = rgb

        return mask_rgb
