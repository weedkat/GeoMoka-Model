import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import time
from typing import Union, Tuple, Dict, List
from pathlib import Path


class SegmentationInference:
    """
    Inference wrapper for semantic segmentation models.
    
    Supports two inference modes:
    1. 'resize': Direct inference with upscaling to original size
    2. 'sliding_window': Patch-based inference with overlap and stitching
    
    Args:
        model: Trained segmentation model (DPT-based)
        device: 'auto' (GPU if available), 'cuda', or 'cpu'
        patch_size: Patch size of the backbone (default 14 for DINOv2)
        overlap_ratio: Overlap ratio for sliding window (0.0-1.0, default 0.5)
    """
    
    def __init__(
        self,
        model,
        device: str = 'auto',
        patch_size: int = 14,
        overlap_ratio: float = 0.5,
    ):
        self.model = model
        self.patch_size = 768  # 14 * 56
        self.overlap_ratio = overlap_ratio
        
        # Device setup
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Extract model info
        self.num_classes = model.head.scratch.output_conv[-1].out_channels
        
        # ImageNet normalization
        self.img_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.img_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        print(f'[Inference] Model loaded on {self.device}')
        print(f'[Inference] Num classes: {self.num_classes}')
    
    def __call__(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor, List, DataLoader],
        mode: str = 'resize',
        return_confidence: bool = False,
        verbose: bool = True,
    ) -> Union[Tuple[np.ndarray, Dict], Tuple[np.ndarray, np.ndarray, Dict]]:
        """
        Perform inference on input image(s).
        
        Args:
            image: Single image (PIL/numpy/tensor) or batch (list/DataLoader)
            mode: 'resize' or 'sliding_window'
            return_confidence: Return confidence scores alongside predictions
            verbose: Print processing info
            
        Returns:
            If single image:
                - return_confidence=False: (pred, metadata)
                - return_confidence=True: (pred, conf, metadata)
            If batch: List of above
        """
        assert mode in ['resize', 'sliding_window'], f"Invalid mode: {mode}"
        
        start_time = time.time()
        
        # Handle different input types
        is_batch = isinstance(image, (list, DataLoader))
        
        if is_batch:
            results = []
            for img in image:
                result = self(img, mode, return_confidence, verbose=False)
                results.append(result)
            
            if verbose:
                elapsed = time.time() - start_time
                print(f'[Inference] Processed {len(results)} images in {elapsed:.2f}s')
            return results
        
        # Single image inference
        img_tensor, orig_shape, input_type = self._normalize_input(image)
        
        with torch.no_grad():
            if mode == 'resize':
                output = self._infer_resize(img_tensor)
            else:  # sliding_window
                output = self._infer_sliding_window(img_tensor)
        
        # Extract predictions and confidence
        pred = output.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)
        conf = output.softmax(dim=1).squeeze(0).cpu().numpy()  # (C, H, W)
        
        # Resize to original size
        pred = self._resize_output(pred, orig_shape)
        conf = self._resize_output(conf, orig_shape)
        
        elapsed = time.time() - start_time
        
        metadata = {
            'original_shape': orig_shape,
            'input_type': input_type,
            'mode': mode,
            'device': self.device,
            'processing_time': elapsed,
            'num_classes': self.num_classes,
        }
        
        if verbose:
            print(f'[Inference] Processed {orig_shape} in {elapsed:.2f}s using {mode}')
        
        if return_confidence:
            return pred, conf, metadata
        else:
            return pred, metadata
    
    def _normalize_input(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[int, int], str]:
        """
        Convert input to normalized tensor batch.
        
        Returns:
            img_tensor: (1, 3, H, W) normalized tensor
            orig_shape: (H, W) tuple
            input_type: str indicating original type
        """
        # Convert to numpy
        if isinstance(image, Image.Image):
            img_np = np.array(image)
            input_type = 'PIL'
        elif isinstance(image, np.ndarray):
            img_np = image
            input_type = 'numpy'
        elif isinstance(image, torch.Tensor):
            if image.ndim == 4:  # Already (B, C, H, W)
                img_np = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            elif image.ndim == 3:
                if image.shape[0] == 3:  # (C, H, W)
                    img_np = image.cpu().numpy().transpose(1, 2, 0)
                else:  # (H, W, C)
                    img_np = image.cpu().numpy()
            input_type = 'tensor'
        else:
            raise TypeError(f"Unsupported input type: {type(image)}")
        
        # Handle grayscale -> RGB
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        
        orig_shape = img_np.shape[:2]  # (H, W)
        
        # Normalize to [0, 1]
        if img_np.dtype != np.float32:
            img_np = img_np.astype(np.float32) / 255.0
        
        # Convert to tensor (H, W, 3) -> (1, 3, H, W)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()
        
        # Apply ImageNet normalization
        img_tensor = (img_tensor - self.img_mean.to(img_tensor.device)) / self.img_std.to(img_tensor.device)
        
        return img_tensor.to(self.device), orig_shape, input_type
    
    def _infer_resize(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Direct inference with upscaling.
        
        Args:
            img_tensor: (1, 3, H, W)
            
        Returns:
            output: (1, C, H, W) logits
        """
        output = self.model(img_tensor)
        return output
    
    def _infer_sliding_window(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Sliding window inference with overlap and soft blending.
        
        Args:
            img_tensor: (1, 3, H, W)
            
        Returns:
            output: (1, C, H, W) logits
        """
        B, C, H, W = img_tensor.shape
        
        # Patch size (model expects divisible by 14)
        patch_h = patch_w = self.patch_size
        
        # Calculate stride based on overlap
        stride_h = int(patch_h * (1 - self.overlap_ratio))
        stride_w = int(patch_w * (1 - self.overlap_ratio))
        
        # Initialize output and weight accumulation
        output = torch.zeros(B, self.num_classes, H, W).to(self.device)
        weights = torch.zeros(1, 1, H, W).to(self.device)
        
        # Gaussian weights for smooth blending in overlap regions
        gaussian_h = torch.exp(-torch.linspace(-1, 1, patch_h).pow(2) / 0.2).unsqueeze(0)
        gaussian_w = torch.exp(-torch.linspace(-1, 1, patch_w).pow(2) / 0.2).unsqueeze(0)
        gaussian_weight = (gaussian_h.T @ gaussian_w).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Sliding window extraction
        row = 0
        while row < H:
            col = 0
            while col < W:
                # Anchor patch at image edge if near the end
                row_end = min(row + patch_h, H)
                col_end = min(col + patch_w, W)
                row_start = max(0, row_end - patch_h)
                col_start = max(0, col_end - patch_w)
                
                # Extract patch
                patch = img_tensor[:, :, row_start:row_end, col_start:col_end]
                
                # Pad if necessary (shouldn't happen but safety check)
                if patch.shape[2] < patch_h or patch.shape[3] < patch_w:
                    pad_h = patch_h - patch.shape[2]
                    pad_w = patch_w - patch.shape[3]
                    patch = F.pad(patch, (0, pad_w, 0, pad_h), mode='reflect')
                
                # Inference
                patch_output = self.model(patch)
                
                # Resize to patch size if needed
                if patch_output.shape[2:] != (patch_h, patch_w):
                    patch_output = F.interpolate(
                        patch_output,
                        size=(patch_h, patch_w),
                        mode='bilinear',
                        align_corners=True
                    )
                
                # Accumulate with Gaussian weighting
                output[:, :, row_start:row_end, col_start:col_end] += \
                    patch_output[:, :, :row_end-row_start, :col_end-col_start] * \
                    gaussian_weight[:, :, :row_end-row_start, :col_end-col_start]
                
                weights[:, :, row_start:row_end, col_start:col_end] += \
                    gaussian_weight[:, :, :row_end-row_start, :col_end-col_start]
                
                # Move to next column
                if col_end >= W:
                    break
                col = min(col + stride_w, W - patch_w)
            
            # Move to next row
            if row_end >= H:
                break
            row = min(row + stride_h, H - patch_h)
        
        # Normalize by accumulated weights
        output = output / (weights + 1e-8)
        
        return output
    
    def _resize_output(
        self,
        output: np.ndarray,
        target_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Resize output to target shape using nearest neighbor.
        
        Args:
            output: (H, W) or (C, H, W)
            target_shape: (target_H, target_W)
            
        Returns:
            resized output matching target shape
        """
        if output.shape[-2:] == target_shape:
            return output
        
        if output.ndim == 2:  # (H, W)
            output_tensor = torch.from_numpy(output[np.newaxis, np.newaxis, :, :]).float()
            resized = F.interpolate(
                output_tensor,
                size=target_shape,
                mode='nearest'
            )
            return resized.squeeze(0).squeeze(0).numpy()
        else:  # (C, H, W)
            output_tensor = torch.from_numpy(output[np.newaxis, :, :, :]).float()
            resized = F.interpolate(
                output_tensor,
                size=target_shape,
                mode='bilinear',
                align_corners=False
            )
            return resized.squeeze(0).numpy()


def infer_single_image(
    model,
    image_path: Union[str, Path],
    mode: str = 'resize',
    return_confidence: bool = False,
) -> Union[Tuple[np.ndarray, Dict], Tuple[np.ndarray, np.ndarray, Dict]]:
    """
    Convenience function for single image inference from file.
    
    Args:
        model: Trained segmentation model
        image_path: Path to input image
        mode: 'resize' or 'sliding_window'
        return_confidence: Whether to return confidence scores
        
    Returns:
        Prediction and metadata (optionally with confidence scores)
    """
    image = Image.open(image_path)
    inferencer = SegmentationInference(model)
    return inferencer(image, mode=mode, return_confidence=return_confidence)
