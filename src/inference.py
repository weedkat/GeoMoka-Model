import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import time
from typing import Union, Tuple, Dict, List
from pathlib import Path
from torchvision import transforms
from dataloader.transform import TransformsCompose
import albumentations as A
from albumentations.pytorch import ToTensorV2

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
        num_classes: int = None,
        device: str = 'auto',
        patch_size: int = 14,
        overlap_ratio: float = 0.5,
        load_messages: bool = True,
        transform_cfg: dict = None,
    ):
        self.model = model
        self.patch_size = patch_size
        self.overlap_ratio = overlap_ratio

        # Transform configuration
        if transform_cfg:
            self.transform = TransformsCompose(transform_cfg)
        else:
            print("Warning: No transform_cfg provided, using default ToTensor transform.")
            self.transform = A.Compose([ToTensorV2()])
        
        # Device setup
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Extract model info - try multiple approaches for different model types
        if num_classes is not None:
            self.num_classes = num_classes
        else:
            self.num_classes = self._detect_num_classes()
        
        if load_messages:
            print(f'[Inference] Model loaded on {self.device}')
            print(f'[Inference] Num classes: {self.num_classes}')
    
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
            "Could not automatically detect num_classes. "
            "Please pass num_classes parameter explicitly to SegmentationInference."
        )
    
    def _apply_confidence_rejection(
        self,
        pred: np.ndarray,
        max_conf: np.ndarray
    ) -> np.ndarray:
        """
        Apply confidence-based rejection to predictions.
        
        Args:
            pred: Predicted labels (H, W)
            max_conf: Max confidence per pixel (H, W)
            
        Returns:
            pred_rejected: Predictions with low-confidence pixels set to reject_class
        """
        pred_rejected = pred.copy()
        pred_rejected[max_conf < self.confidence_threshold] = self.reject_class
        return pred_rejected
    
    def set_confidence_threshold(self, threshold: float, reject_class: int = 255) -> None:
        """
        Set confidence threshold for rejection.
        
        Args:
            threshold: Confidence threshold [0, 1]. None to disable.
            reject_class: Class ID for rejected pixels
        """
        self.confidence_threshold = threshold
        self.reject_class = reject_class
    
    def get_confidence_stats(
        self,
        images: Union[List, DataLoader],
        mode: str = 'sliding_window'
    ) -> Dict:
        """
        Compute confidence statistics over multiple images to guide threshold selection.
        
        Args:
            images: List of images or DataLoader
            mode: 'resize' or 'sliding_window'
            
        Returns:
            Dictionary with confidence statistics
        """
        all_conf = []
        
        image_list = images if isinstance(images, list) else list(images)
        
        # compute max confidence across classes for each pixel
        for img in image_list:
            _, conf, _ = self(img, mode=mode, return_confidence=True, verbose=False)
            max_conf = np.max(conf, axis=0).flatten()
            all_conf.append(max_conf)
        
        all_conf = np.concatenate(all_conf)
        
        return {
            'mean': float(np.mean(all_conf)),
            'median': float(np.median(all_conf)),
            'std': float(np.std(all_conf)),
            'min': float(np.min(all_conf)),
            'max': float(np.max(all_conf)),
            'percentile_10': float(np.percentile(all_conf, 10)),
            'percentile_25': float(np.percentile(all_conf, 25)),
            'percentile_75': float(np.percentile(all_conf, 75)),
            'percentile_90': float(np.percentile(all_conf, 90)),
        }
    
    def __call__(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor, List, DataLoader],
        mode: str = 'sliding_window',
        return_confidence: bool = False,
        verbose: bool = True,
    ) -> Union[Tuple[np.ndarray, Dict], Tuple[np.ndarray, np.ndarray, Dict]]:
        """
        Perform inference on input image(s).
        
        Args:
            image: Single image (PIL/numpy/tensor) or batch (list/DataLoader)
            mode: 'sliding_window' or 'resize'  
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

        max_conf = np.max(conf, axis=0)  # (H, W)
        pred = self._apply_confidence_rejection(pred, max_conf) if hasattr(self, 'confidence_threshold') else pred
        
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
        
        img_tensor = self.transform_config(image=np.array(img_pil))['image'] if self.transform_config else transforms.ToTensor()(img_pil)
        
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
    inferencer = SegmentationInference(model, load_messages=False)
    return inferencer(image, mode=mode, return_confidence=return_confidence)
