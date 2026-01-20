import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import time
from typing import Union, Tuple, Dict, List
from pathlib import Path
from torchvision import transforms


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
        self.patch_size = patch_size
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
        self.img_mean = [0.485, 0.456, 0.406]
        self.img_std = [0.229, 0.224, 0.225]
        
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
        
        # Ensure uint8 for PIL Image
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)
        
        # Convert to PIL Image for transforms
        img_pil = Image.fromarray(img_np)
        
        img_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.img_mean, self.img_std),
        ])(img_pil)
        
        # Add batch dimension: (C, H, W) -> (1, C, H, W)
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor.to(self.device), orig_shape, input_type
    
    def _infer_resize(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Direct inference with resizing to patch_size and upscaling back.
        
        Args:
            img_tensor: (1, 3, H, W)
            
        Returns:
            output: (1, C, H, W) logits
        """
        B, C, H, W = img_tensor.shape
        
        # Resize to patch_size (must be divisible by 14)
        target_size = self.patch_size
        # Make sure target_size is divisible by 14
        target_size = (target_size // 14) * 14
        
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
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )
        
        return output
    
    def _infer_sliding_window(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Sliding window inference matching the original evaluation method.
        
        Args:
            img_tensor: (1, 3, H, W)
            
        Returns:
            output: (1, C, H, W) logits
        """
        grid = self.patch_size
        b, _, h, w = img_tensor.shape
        final = torch.zeros(b, self.num_classes, h, w).to(self.device)
        
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
                final[:, :, row:row_end, col:col_end] += pred
                
                # Move to next column
                if col >= w - grid:
                    break
                col = min(col + int(grid * 2 / 3), w - grid)
            
            # Move to next row
            if row >= h - grid:
                break
            row = min(row + int(grid * 2 / 3), h - grid)
        
        return final

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
