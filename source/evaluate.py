import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm

from inference import SegmentationInference


# ====================== Core Metric Functions ======================

def compute_confusion_matrix(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = 255
) -> np.ndarray:
    """
    Compute confusion matrix for semantic segmentation.
    
    Args:
        pred: Predicted labels (H, W) or flattened (N,)
        target: Ground truth labels (H, W) or flattened (N,)
        num_classes: Number of classes
        ignore_index: Label to ignore in evaluation
        
    Returns:
        confusion_matrix: (num_classes, num_classes) matrix where
                         matrix[i, j] = number of pixels with true class i predicted as j
    """
    pred = pred.flatten()
    target = target.flatten()
    
    # Mask out ignore index
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]
    
    # Compute confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    for true_class in range(num_classes):
        for pred_class in range(num_classes):
            confusion_matrix[true_class, pred_class] = np.sum(
                (target == true_class) & (pred == pred_class)
            )
    
    return confusion_matrix


def compute_iou_from_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Compute IoU metrics from confusion matrix.
    
    Args:
        confusion_matrix: (num_classes, num_classes) confusion matrix
        class_names: Optional list of class names for better readability
        
    Returns:
        Dictionary containing:
            - 'per_class_iou': dict mapping class index/name to IoU
            - 'miou': mean IoU across all classes
            - 'miou_valid': mean IoU excluding classes with no ground truth
    """
    num_classes = confusion_matrix.shape[0]
    
    # Compute IoU for each class
    iou_per_class = []
    per_class_iou_dict = {}
    
    for i in range(num_classes):
        true_positive = confusion_matrix[i, i]
        false_positive = confusion_matrix[:, i].sum() - true_positive
        false_negative = confusion_matrix[i, :].sum() - true_positive
        
        denominator = true_positive + false_positive + false_negative
        
        if denominator == 0:
            iou = np.nan  # Class not present in ground truth or predictions
        else:
            iou = true_positive / denominator
        
        iou_per_class.append(iou)
        
        # Create dict key
        if class_names and i < len(class_names):
            key = f"class_{i}_{class_names[i]}"
        else:
            key = f"class_{i}"
        
        per_class_iou_dict[key] = iou * 100.0 if not np.isnan(iou) else np.nan
    
    iou_array = np.array(iou_per_class)
    
    # mIoU (all classes)
    miou = np.nanmean(iou_array) * 100.0
    
    # mIoU (only valid classes with ground truth)
    valid_classes = ~np.isnan(iou_array)
    miou_valid = np.mean(iou_array[valid_classes]) * 100.0 if valid_classes.any() else 0.0
    
    return {
        'per_class_iou': per_class_iou_dict,
        'miou': miou,
        'miou_valid': miou_valid,
        'iou_array': iou_array * 100.0  # For backward compatibility
    }


def compute_dice_coefficient(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Compute Dice coefficient (F1 score) for each class.
    
    Args:
        pred: Predicted labels
        target: Ground truth labels
        num_classes: Number of classes
        ignore_index: Label to ignore
        class_names: Optional class names
        
    Returns:
        Dictionary containing per-class Dice and mean Dice
    """
    pred = pred.flatten()
    target = target.flatten()
    
    # Mask out ignore index
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]
    
    dice_per_class = {}
    dice_values = []
    
    for i in range(num_classes):
        pred_mask = pred == i
        target_mask = target == i
        
        intersection = np.sum(pred_mask & target_mask)
        pred_sum = np.sum(pred_mask)
        target_sum = np.sum(target_mask)
        
        denominator = pred_sum + target_sum
        
        if denominator == 0:
            dice = np.nan
        else:
            dice = 2.0 * intersection / denominator
        
        dice_values.append(dice)
        
        if class_names and i < len(class_names):
            key = f"class_{i}_{class_names[i]}"
        else:
            key = f"class_{i}"
        
        dice_per_class[key] = dice * 100.0 if not np.isnan(dice) else np.nan
    
    dice_array = np.array(dice_values)
    mean_dice = np.nanmean(dice_array) * 100.0
    
    return {
        'per_class_dice': dice_per_class,
        'mean_dice': mean_dice
    }


def compute_pixel_accuracy(
    pred: np.ndarray,
    target: np.ndarray,
    ignore_index: int = 255
) -> Dict:
    """
    Compute pixel-wise accuracy metrics.
    
    Args:
        pred: Predicted labels
        target: Ground truth labels
        ignore_index: Label to ignore
        
    Returns:
        Dictionary containing overall and mean class accuracy
    """
    pred = pred.flatten()
    target = target.flatten()
    
    # Mask out ignore index
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]
    
    # Overall pixel accuracy
    correct = np.sum(pred == target)
    total = len(target)
    overall_acc = (correct / total * 100.0) if total > 0 else 0.0
    
    # Mean class accuracy
    unique_classes = np.unique(target)
    class_accs = []
    
    for cls in unique_classes:
        cls_mask = target == cls
        cls_correct = np.sum((pred == target) & cls_mask)
        cls_total = np.sum(cls_mask)
        cls_acc = cls_correct / cls_total if cls_total > 0 else 0.0
        class_accs.append(cls_acc)
    
    mean_class_acc = np.mean(class_accs) * 100.0 if class_accs else 0.0
    
    return {
        'overall_accuracy': overall_acc,
        'mean_class_accuracy': mean_class_acc,
        'correct_pixels': int(correct),
        'total_pixels': int(total)
    }


# ====================== Main Evaluation Function ======================

def evaluate(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
    class_names: Optional[List[str]] = None,
    verbose: bool = False
) -> Dict:
    """
    Compute all evaluation metrics from predictions and targets.
    
    Args:
        pred: Predicted labels (H, W) or (N, H, W) for multiple images
        target: Ground truth labels, same shape as pred
        num_classes: Number of classes
        ignore_index: Label to ignore in evaluation
        class_names: Optional list of class names
        verbose: Print detailed results
        
    Returns:
        Dictionary containing all metrics:
            - confusion_matrix
            - per_class_iou, miou, miou_valid
            - per_class_dice, mean_dice
            - overall_accuracy, mean_class_accuracy
    """
    # Ensure inputs are numpy arrays
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    # Compute confusion matrix
    confusion_matrix = compute_confusion_matrix(pred, target, num_classes, ignore_index)
    
    # Compute all metrics
    iou_metrics = compute_iou_from_confusion_matrix(confusion_matrix, class_names)
    dice_metrics = compute_dice_coefficient(pred, target, num_classes, ignore_index, class_names)
    accuracy_metrics = compute_pixel_accuracy(pred, target, ignore_index)
    
    # Combine all metrics
    results = {
        'confusion_matrix': confusion_matrix,
        **iou_metrics,
        **dice_metrics,
        **accuracy_metrics
    }
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"Evaluation Results:")
        print(f"{'='*50}")
        print(f"mIoU: {results['miou']:.2f}%")
        print(f"Mean Dice: {results['mean_dice']:.2f}%")
        print(f"Overall Accuracy: {results['overall_accuracy']:.2f}%")
        print(f"Mean Class Accuracy: {results['mean_class_accuracy']:.2f}%")
        print(f"\nPer-class IoU:")
        for cls_name, iou in results['per_class_iou'].items():
            if not np.isnan(iou):
                print(f"  {cls_name}: {iou:.2f}%")
        print(f"{'='*50}\n")
    
    return results


# ====================== Inference-based Evaluation ======================

def inference_evaluate(
    model,
    dataloader: DataLoader,
    num_classes: int,
    ignore_index: int = 255,
    mode: str = 'resize',
    class_names: Optional[List[str]] = None,
    device: str = 'auto',
    verbose: bool = True
) -> Dict:
    """
    Evaluate model on a dataset using the inference module.
    Wrapper around evaluate() that performs inference first.
    
    Args:
        model: Trained segmentation model
        dataloader: DataLoader for evaluation dataset
        num_classes: Number of classes
        ignore_index: Label to ignore in evaluation
        mode: Inference mode ('resize' or 'sliding_window')
        class_names: Optional list of class names
        device: Device for inference ('auto', 'cuda', or 'cpu')
        verbose: Print progress and results
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    inferencer = SegmentationInference(model, device=device)
    
    all_preds = []
    all_targets = []
    
    iterator = tqdm(dataloader, desc="Evaluating") if verbose else dataloader
    
    for images, targets in iterator:
        # Handle batched inputs
        if images.dim() == 4:  # (B, C, H, W)
            batch_size = images.size(0)
            for i in range(batch_size):
                img = images[i]
                target = targets[i].numpy()
                
                # Inference
                pred, _ = inferencer(img, mode=mode, return_confidence=False, verbose=False)
                
                all_preds.append(pred)
                all_targets.append(target)
        else:
            target = targets.numpy()
            pred, _ = inferencer(images, mode=mode, return_confidence=False, verbose=False)
            all_preds.append(pred)
            all_targets.append(target)
    
    # Stack all predictions and targets
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Call the main evaluate function
    results = evaluate(
        pred=all_preds,
        target=all_targets,
        num_classes=num_classes,
        ignore_index=ignore_index,
        class_names=class_names,
        verbose=verbose
    )
    
    return results
