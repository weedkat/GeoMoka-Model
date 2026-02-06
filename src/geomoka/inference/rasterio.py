from geomoka.inference.engine import SegmentationInference

class RasterioSegmentationInference(SegmentationInference):
    """
    Inference engine for rasterio datasets.
    """
    def __init__(
        self,
        model: nn.Module,
        patch_size: int,
        overlap: int = 0,
        batch_size: int = 4,
        num_workers: int = 2,
        device: Union[str, torch.device] = 'cuda',
        num_classes: int = None,
    ):
        super().__init__(
            model=model,
            patch_size=patch_size,
            overlap=overlap,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            num_classes=num_classes,
        )