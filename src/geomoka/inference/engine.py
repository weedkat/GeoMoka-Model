from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import time
from typing import Union, Tuple, Dict, List
from geomoka.dataloader.transform import TransformsCompose
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SegmentationInference:
    """
    Inference wrapper for semantic segmentation models.
    
    Supports two inference modes:
    1. 'resize': Direct inference with upscaling to original size
    2. 'sliding_window': Patch-based inference with overlap and stitching
    
    Args:
        model: Trained segmentation model
        device: 'auto' (GPU if available), 'cuda', or 'cpu'
        patch_size: Patch size of the backbone (default 14 for DINOv2)
        overlap_ratio: Overlap ratio for sliding window (0.0-1.0, default 0.5)
    """
    def __init__(
        self,
        model: torch.nn.Module,
        patch_size: int,
        overlap_ratio: float = 0.5,
        transform_cfg: dict = None,
        reject_class: int = 255,
        load_messages: bool = True,
        confidence_threshold: float = 0.0,
        device: str = 'auto',
    ):
        assert 0.0 <= confidence_threshold <= 1.0, "Threshold must be in [0, 1]"
        assert 0.0 <= overlap_ratio < 1.0, "Overlap ratio must be in [0, 1)"
        assert device in ['auto', 'cuda', 'cpu'], f"Invalid device: {device}"

        self.model = model
        self.patch_size = patch_size
        self.overlap_ratio = overlap_ratio
        self.reject_class = reject_class
        self.confidence_threshold = confidence_threshold

        # Transform configuration
        if transform_cfg:
            self.transform = TransformsCompose(transform_cfg)
        else:
            print("Warning: No transform_cfg provided, using default ToTensor transform.")
            self.transform = A.Compose([ToTensorV2()])
        
        self.device = self._detect_device(device)
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Extract model info - try multiple approaches for different model types
        self.num_classes = self._detect_num_classes()
        
        if load_messages:
            print(f'[Inference] Model loaded on {self.device}')
            print(f'[Inference] Num classes: {self.num_classes}')

    def __call__(
        self,
        images: Union[Image.Image, np.ndarray, torch.Tensor],
        mode: str = 'sliding_window',
        verbose: bool = True,
    ) -> Union[Tuple[np.ndarray, Dict], Tuple[np.ndarray, np.ndarray, Dict]]:
        """
        Perform inference on input image(s).
        
        Args:
            image: Single image (PIL/numpy/tensor) or batch tensor/ndarray
            mode: 'sliding_window' or 'resize'
            verbose: Print processing info
            
        Returns:
            If single image:
                pred: (H, W) predicted labels
                conf: (H, W, C) confidence scores per class
                metadata: dict with processing info
            If batch:
                preds: (N, H, W) predicted labels
                confs: (N, H, W, C) confidence scores per class
                metadata: dict with processing info of the first image
        """
        assert mode in ['resize', 'sliding_window'], f"Invalid mode: {mode}"
        
        start_time = time.time()
        
        # Normalize input to tensor batch and get original shape
        img_tensor, orig_shape, input_type = self._normalize_input(images)
        
        with torch.no_grad():
            if mode == 'resize':
                output = self._infer_resize(img_tensor)
            elif mode == 'sliding_window':
                output = self._infer_sliding_window(img_tensor)
            else:
                raise ValueError(f"Unsupported mode: {mode}")
        
        # Extract predictions and confidence
        pred = output.argmax(dim=1).cpu().numpy()  # (B, C, H, W) -> (B, H, W)
        conf = output.softmax(dim=1).cpu().numpy().transpose(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)

        # Apply confidence threshold to reject uncertain predictions
        max_conf = np.max(conf, axis=-1)  # (B, H, W, C) -> (B, H, W)
        pred[max_conf < self.confidence_threshold] = self.reject_class
        
        elapsed = time.time() - start_time
        
        metadata = {
            'input_type': input_type,
            'original_shape': orig_shape,
            'patch_size': self.patch_size,
            'overlap_ratio': self.overlap_ratio,
            'confidence_threshold': self.confidence_threshold,
            'mode': mode,
            'device': self.device,
            'processing_time': elapsed,
            'num_classes': self.num_classes ,
        }
        
        if verbose:
            print(f'[Inference] Processed {orig_shape} in {elapsed:.2f}s using {mode}')
        
        return pred, conf, metadata
    
    def _detect_device(self, device: str) -> None:
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
        
    def _detect_num_classes(self) -> int:
        """
        Detect number of classes from model architecture.
        Supports DPT, DeepLabV3, and other segmentation models.
        """
        # Try DPT architecture
        if hasattr(self.model, 'head') and hasattr(self.model.head, 'scratch'):
            try:
                return self.model.head.scratch.output_conv[-1].out_channels
            except:
                pass
        
        # Try segmentation_models_pytorch (DeepLabV3, FPN, etc.)
        if hasattr(self.model, 'segmentation_head'):
            try:
                # segmentation_head is usually a Sequential with final layer
                for layer in reversed(self.model.segmentation_head.modules()):
                    if hasattr(layer, 'out_channels'):
                        return layer.out_channels
                    if hasattr(layer, 'out_features'):
                        return layer.out_features
            except:
                pass
        
        # Try decoder
        if hasattr(self.model, 'decoder'):
            try:
                if hasattr(self.model.decoder, 'out_channels'):
                    return self.model.decoder.out_channels
            except:
                pass
        
        # Fallback: try to infer from forward pass with dummy input
        try:
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 256, 256).to(self.device)
                output = self.model(dummy_input)
                if isinstance(output, dict):
                    output = output.get('out', list(output.values())[0])
                return output.shape[1]
        except:
            pass
        
        raise RuntimeError(
            "Could not detect num_classes. Even with a dummy forward pass."
        )
    
    def _normalize_input(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[int, int], str]:
        """
        Convert input to normalized tensor batch.

        Returns:
            img_tensor: (B, C, H, W) normalized tensor
            orig_shape: (H, W) tuple
            input_type: str indicating original type
        """
        # Handle tensor inputs that are already properly formatted
        if isinstance(image, torch.Tensor):
            if image.ndim == 4:  # Could be (B, C, H, W) or (B, H, W, C)
                # Check if it's already in (B, C, H, W) format
                # Heuristic: if dim 1 is small (1-4), assume it's channels
                if image.shape[1] <= 4:  # (B, C, H, W) - already transformed
                    orig_shape = (image.shape[2], image.shape[3])
                    return image.to(self.device), orig_shape, 'tensor'
                else:  # (B, H, W, C) - needs transpose
                    img_np = image.cpu().numpy()
                    input_type = 'tensor'
            elif image.ndim == 3 and image.shape[0] in [1, 3]:  # (C, H, W) - already transformed
                orig_shape = (image.shape[1], image.shape[2])
                return image.unsqueeze(0).to(self.device), orig_shape, 'tensor'
            else:
                # Other tensor formats - convert to numpy for transform
                img_np = image.cpu().numpy()
                input_type = 'tensor'
        
        # For numpy arrays and PIL images, apply transforms
        if isinstance(image, Image.Image):
            img_np = np.array(image)
            input_type = 'PIL'

        elif isinstance(image, np.ndarray):
            img_np = image
            input_type = 'numpy'
            
        elif not isinstance(image, torch.Tensor):
            raise TypeError(f"Unsupported input type: {type(image)}")

        # Handle batched input (B, H, W, C) or (B, H, W)
        if img_np.ndim == 4:
            # Already batched (B, H, W, C)
            pass
        # Ensure (H, W, C) format for single images
        elif img_np.ndim == 2:
            img_np = np.expand_dims(img_np, axis=-1)  # (H, W, 1)
            img_np = np.expand_dims(img_np, axis=0)  # (1, H, W, 1)
        # Add batch dimension if single image (H, W, C)
        elif img_np.ndim == 3:
            img_np = np.expand_dims(img_np, axis=0)  # (1, H, W, C)
        
        # Now img_np is (B, H, W, C)
        orig_shape = (img_np.shape[1], img_np.shape[2])  # (H, W)
        
        # Ensure uint8
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)
        
        # Process each image in batch through transforms
        img_tensors = []
        for i in range(img_np.shape[0]):
            img_pil = Image.fromarray(img_np[i])
            img_tensor = self.transform(image=np.array(img_pil))['image']
            img_tensors.append(img_tensor)
        
        img_tensor = torch.stack(img_tensors, dim=0)  # (B, C, H, W)
        
        return img_tensor.to(self.device), orig_shape, input_type
    
    def _infer_resize(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Direct inference with resizing to patch_size and upscaling back.
        
        Args:
            img_tensor: (B, C, H, W)
            
        Returns:
            output: (B, C, H, W) logits
        """
        b, c, h, w = img_tensor.shape
        
        target_size = self.patch_size
        
        # Resize input
        img_resized = F.interpolate(
            img_tensor,
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False
        )
        
        # Inference on resized image
        output = self.model(img_resized)
        
        # Resize output back to original size
        output = F.interpolate(
            output,
            size=(h, w),
            mode='bilinear',
            align_corners=False
        )
        
        return output
    
    def _infer_sliding_window(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Sliding window inference matching the original evaluation method.
        
        Args:
            img_tensor: (B, C, H, W)
            
        Returns:
            output: (B, C, H, W) logits
        """
        grid = self.patch_size
        b, _, h, w = img_tensor.shape
        final = torch.zeros(b, self.num_classes, h, w).to(self.device)

        # Semantic segmentation tend to be more accurate on the center of patches
        # hann_window create a bell curve to weight the center more
        weight = torch.zeros(b, 1, h, w).to(self.device)
        window_weight = torch.hann_window(grid, periodic=False).to(self.device)
        window_2d = window_weight.unsqueeze(0) * window_weight.unsqueeze(1)
        
        row = 0
        while row < h:
            col = 0
            while col < w:
                # Extract window, handling edge cases
                row_end = min(row + grid, h)
                col_end = min(col + grid, w)
                
                window = img_tensor[:, :, row:row_end, col:col_end]
                
                # Run inference
                pred = self.model(window)
                
                # Accumulate logits
                final[:, :, row:row_end, col:col_end] += pred * window_2d
                weight[:, :, row:row_end, col:col_end] += window_2d
                
                # Move to next column
                if col >= w - grid:
                    break

                # Make sure to start at w - grid when near edge
                col = min(col + int(grid * self.overlap_ratio), w - grid)
            
            # Move to next row
            if row >= h - grid:
                break
            row = min(row + int(grid * self.overlap_ratio), h - grid)
        
        final /= weight

        return final