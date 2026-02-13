from typing import Union
import torch
from torch import nn

from geomoka.inference.engine import SegmentationInference

class RasterioSegmentationInference(SegmentationInference):
    """
    Inference engine for rasterio datasets.
    """
    def __init__(
        self,
        model: nn.Module,
        patch_size: int,
        overlap_ratio: float = 0.5,
        device: Union[str, torch.device] = 'cuda',
        transform_cfg: dict = None,
        reject_class: int = 255,
        load_messages: bool = True,
    ):
        super().__init__(
            model=model,
            patch_size=patch_size,
            overlap_ratio=overlap_ratio,
            transform_cfg=transform_cfg,
            reject_class=reject_class,
            load_messages=load_messages,
            device=device,
        )