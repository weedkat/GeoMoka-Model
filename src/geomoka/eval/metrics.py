import numpy as np
from typing import Optional, List, Dict

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


def compute_iou(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
    class_dict: Optional[List[str]] = None
) -> Dict:
    """
    Compute IoU metrics directly from predictions and targets.
    
    Args:
        pred: Predicted labels (H, W) or flattened (N,)
        target: Ground truth labels (H, W) or flattened (N,)
        num_classes: Number of classes
        ignore_index: Label to ignore in evaluation
        class_dict: Optional list of class dictionaries for better readability
        
    Returns:
        Dictionary containing:
            - 'per_class_iou': dict mapping class index/name to IoU
            - 'miou': mean IoU across all classes
            - 'miou_valid': mean IoU excluding classes with no ground truth
    """
    pred = pred.flatten()
    target = target.flatten()
    
    # Mask out ignore index
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]
    
    # Compute IoU for each class
    iou_per_class = []
    per_class_iou_dict = {}
    
    for i in range(num_classes):
        pred_mask = pred == i
        target_mask = target == i
        
        true_positive = np.sum(pred_mask & target_mask)
        false_positive = np.sum(pred_mask & ~target_mask)
        false_negative = np.sum(~pred_mask & target_mask)
        
        denominator = true_positive + false_positive + false_negative
        
        if denominator == 0:
            iou = np.nan  # Class not present in ground truth or predictions
        else:
            iou = true_positive / denominator
        
        iou_per_class.append(iou)
        
        # Create dict key
        if class_dict and i < len(class_dict):
            key = f"class_{i}_{class_dict[i]['name']}"
        else:
            key = f"class_{i}"
        
        per_class_iou_dict[key] = iou * 100.0 if not np.isnan(iou) else np.nan
    
    # mIoU (excluding classes not present in ground truth)
    iou_array = np.array(iou_per_class)
    miou = np.nanmean(iou_array) * 100.0
    
    return {
        'per_class_iou': per_class_iou_dict,
        'miou': miou,
    }


def compute_dice_coefficient(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
    class_dict: Optional[List[Dict]] = None
) -> Dict:
    """
    Compute Dice coefficient (F1 score) for each class.
    
    Args:
        pred: Predicted labels
        target: Ground truth labels
        num_classes: Number of classes
        ignore_index: Label to ignore
        class_dict: Optional list of class dictionaries for better readability
        
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
        
        if class_dict and i < len(class_dict):
            key = f"class_{i}_{class_dict[i]['name']}"
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
        'micro_accuracy': overall_acc,
        'macro_accuracy': mean_class_acc,
        'correct_pixels': int(correct),
        'total_pixels': int(total)
    }
