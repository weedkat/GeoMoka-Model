from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import time
from typing import Tuple, Dict, List
from geomoka.dataloader.transform import TransformsCompose
import albumentations as A
from albumentations.pytorch import ToTensorV2
from geomoka.model.segmentation.base import SegmentationModel


class Inference:
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
        model: SegmentationModel,
        load_messages: bool = True,
    ):
        self.model = model.model
        self.patch_size = model.model_cfg['crop_size']
        self.reject_class = model.metadata['ignore_index']
        self.confidence_threshold = model.model_cfg.get('confidence_threshold', 0.0)
        self.num_classes = model.model_cfg['nclass']

        # Transform configuration
        self.transform = model.transform_inference
        
        self.model.eval()
        
        if load_messages:
            print(f'[Inference] Model loaded on {self.device}')
            print(f'[Inference] Classes: {self.num_classes}')
    
    def __call__(
        self,
        images: np.ndarray,
        mode: str = 'sliding_window',
        overlap_ratio: float = None,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Perform inference on input image(s).
        
        Args:
            images: Numpy array (H, W, C) or (B, H, W, C)
            mode: 'sliding_window' or 'resize'
            overlap_ratio: Overlap ratio for sliding window (default: from init)
            verbose: Print processing info
            
        Returns:
            pred: (H, W) or (B, H, W) predicted labels
            conf: (H, W, C) or (B, H, W, C) confidence scores per class
            metadata: dict with processing info
        """
        assert 0.0 <= overlap_ratio < 1.0, "Overlap ratio must be in [0, 1)"
        assert mode in ['resize', 'sliding_window'], f"Invalid mode: {mode}"
        
        start_time = time.time()
        
        # Normalize input to tensor batch and get original shape
        img_tensor, orig_shape = self._normalize_input(images)
        
        with torch.no_grad():
            if mode == 'resize':
                output = self._infer_resize(img_tensor)
            elif mode == 'sliding_window':
                output = self._infer_sliding_window(img_tensor, overlap_ratio)
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
            'original_shape': orig_shape,
            'patch_size': self.patch_size,
            'overlap_ratio': overlap_ratio,
            'confidence_threshold': self.confidence_threshold,
            'mode': mode,
            'device': self.device,
            'processing_time': elapsed,
            'num_classes': self.num_classes,
        }
        
        if verbose:
            print(f'[Inference] Processed {orig_shape} in {elapsed:.2f}s using {mode}')
        
        return pred, conf, metadata

    def preprocess(
        self,
        img_np: np.ndarray
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Convert numpy array to normalized tensor batch.
        
        Args:
            image: Numpy array, either (H, W, C) or (B, H, W, C)

        Returns:
            img_tensor: (B, C, H, W) normalized tensor
            orig_shape: (H, W) tuple
        """
        # Handle batched input (B, H, W, C) or (B, H, W)
        # Ensure (1, H, W, C) format for single images
        if img_np.ndim == 2:
            img_np = np.expand_dims(img_np, axis=-1)  # (H, W, 1)
        # Add batch dimension if single image (H, W, C)
        if img_np.ndim == 3:
            img_np = np.expand_dims(img_np, axis=0)  # (1, H, W, C)
        
        # Now img_np is (B, H, W, C)
        orig_shape = (img_np.shape[1], img_np.shape[2])  # (H, W)
        
        # Select bands if specified
        img_np = self._select_bands(img_np)
        
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
        
        return img_tensor.to(self.device), orig_shape
    
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
    
    def _infer_sliding_window(self, img_tensor: torch.Tensor, overlap_ratio: float = 0.5) -> torch.Tensor:
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