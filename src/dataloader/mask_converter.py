import numpy as np
import yaml

class MaskConverter:
    def __init__(self, metadata):
        if isinstance(metadata, str):
            metadata = yaml.load(open(metadata, 'r'), Loader=yaml.Loader)
        
        self.class_dict = metadata["class_dict"]
        self.class_map = metadata.get("class_map", {})
        self.reverse_map = {v: k for k, v in self.class_map.items()}
        self.class_ids = sorted(self.class_dict.keys())

    def rgb_to_class(self, mask_rgb):
        """ Input is in (H, W, C) format
        """
        if not isinstance(mask_rgb, np.ndarray):
            mask_rgb = np.array(mask_rgb)

        h, w, _ = mask_rgb.shape
        target = np.zeros((h, w), dtype=np.int64)

        for class_idx, item in self.class_dict.items():
            match = np.all(mask_rgb == item['rgb'], axis=-1)
            target[match] = class_idx

        if self.class_map:
            mapped_target = np.full((h, w), 255, dtype=np.int64)  # default ignore value
            for index, class_id in self.class_map.items():
                mapped_target[target == class_id] = index
            return mapped_target

        return target
    
    def class_to_rgb(self, mask_class):
        if not isinstance(mask_class, np.ndarray):
            mask_class = np.array(mask_class)

        h, w = mask_class.shape
        mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)

        if self.class_map:
            remapped_mask = np.full((h, w), 255, dtype=np.int64)  # default ignore value
            for index, class_id in self.reverse_map.items():
                remapped_mask[mask_class == index] = class_id
            mask_class = remapped_mask

        for class_idx, item in self.class_dict.items():
            rgb = np.array(item["rgb"], dtype=np.uint8)
            mask_rgb[mask_class == class_idx] = rgb

        return mask_rgb
    
    def get_class_dict(self):
        if class_map := self.class_map:
            mapped_class_dict = {}
            for index, class_id in class_map.items():
                mapped_class_dict[index] = self.class_dict[class_id]
            return mapped_class_dict
        
        return self.class_dict
