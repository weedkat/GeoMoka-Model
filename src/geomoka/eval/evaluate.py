import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, List
import logging
from tqdm import tqdm

from geomoka.inference.engine import SegmentationInference
from geomoka.eval.metrics import compute_confusion_matrix, compute_iou, compute_dice_coefficient, compute_pixel_accuracy

# ====================== Verbose Printing Helper ======================

def format_evaluation_results(
    results: Dict,
) -> List[str]:
    """
    Format evaluation results into readable strings.
    """
    messages = []
    
    messages.append(f"{'='*50}")
    messages.append("Evaluation Results:")
    messages.append(f"{'='*50}")
    messages.append(f"mIoU: {results['miou']:.2f}%")
    messages.append(f"Mean Dice: {results['mean_dice']:.2f}%")
    messages.append(f"Micro Accuracy: {results['micro_accuracy']:.2f}%")
    messages.append(f"Macro Accuracy: {results['macro_accuracy']:.2f}%")
    
    messages.append("Per-class IoU:")
    for cls_name, iou in results['per_class_iou'].items():
        if not np.isnan(iou):
            messages.append(f"  {cls_name}: {iou:.2f}%")
    
    messages.append(f"{'='*50}\n")
    
    return messages

# ====================== Main Evaluation Function ======================

def evaluate(
    pred: np.ndarray,
    target: np.ndarray,
    class_dict: Dict[int, Dict],
    ignore_index: int = 255,
    verbose: bool = False,
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    Compute all evaluation metrics from predictions and targets.
    
    Args:
        pred: Predicted labels (H, W) or (N, H, W) for multiple images
        target: Ground truth labels, same shape as pred
        class_dict: Class dictionary mapping class_id -> metadata (mandatory)
        ignore_index: Label to ignore in evaluation
        verbose: Print detailed results to console
        logger: Optional logger instance for logging results
        
    Returns:
        Dictionary containing all metrics:
            - confusion_matrix
            - per_class_iou, miou, miou_valid
            - per_class_dice, mean_dice
            - micro_accuracy, macro_accuracy
    """
    # Ensure inputs are numpy arrays
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    num_classes = len(class_dict)

    # Compute confusion matrix
    confusion_matrix = compute_confusion_matrix(pred, target, num_classes, ignore_index)
    
    # Compute all metrics
    iou_metrics = compute_iou(pred, target, num_classes, ignore_index, class_dict)
    dice_metrics = compute_dice_coefficient(pred, target, num_classes, ignore_index, class_dict)
    accuracy_metrics = compute_pixel_accuracy(pred, target, ignore_index)
    
    # Combine all metrics
    results = {
        'confusion_matrix': confusion_matrix,
        **iou_metrics,
        **dice_metrics,
        **accuracy_metrics
    }
    
    if verbose or logger:
        messages = format_evaluation_results(results)
        
        if verbose:
            for msg in messages:
                print(msg)
        
        if logger:
            for msg in messages:
                logger.info(msg)
    
    return results

# ====================== Inference-based Evaluation ======================

def inference_evaluate(
    model,
    dataloader: DataLoader,
    class_dict: Dict[int, Dict],
    ignore_index: int = 255,
    mode: str = 'resize',
    device: str = 'auto',
    verbose: bool = True,
    logger: Optional[logging.Logger] = None,
    patch_size: int = None,
    transform_cfg: Optional[Dict] = None
) -> Dict:
    """
    Evaluate model on a dataset using the inference module.
    Wrapper around evaluate() that performs inference first.
    
    Args:
        model: Trained segmentation model
        dataloader: DataLoader for evaluation dataset
        ignore_index: Label to ignore in evaluation
        mode: Inference mode ('resize' or 'sliding_window')
        class_dict: List of class dictionaries (mandatory)
        device: Device for inference ('auto', 'cuda', or 'cpu')
        verbose: Print progress and results to console
        logger: Optional logger instance for logging results
        patch_size: Patch size for sliding window mode
        transform_cfg: Optional dictionary for transformation configuration
    Returns:
        Dictionary containing all evaluation metrics
    """
    assert (mode == 'sliding_window' and patch_size is not None) or mode == 'resize', \
        "For 'sliding_window' mode, patch_size must be provided."

    num_classes = len(class_dict)

    inferencer = SegmentationInference(
        model=model,
        patch_size=patch_size,
        device=device,
        load_messages=False,
        transform_cfg=transform_cfg,
    )
    
    all_preds = []
    all_targets = []
    
    iterator = tqdm(dataloader, desc="Evaluating") if verbose else dataloader
    
    for images, targets in iterator:
        if isinstance(targets, torch.Tensor):
            targets_np = targets.cpu().numpy()
        else:
            targets_np = targets
        
        pred, _, _ = inferencer(images, mode=mode, verbose=False)
        all_preds.append(pred)
        all_targets.append(targets_np)
    
    # Stack all predictions and targets
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Call the main evaluate function
    results = evaluate(
        pred=all_preds,
        target=all_targets,
        class_dict=class_dict,
        ignore_index=ignore_index,
        verbose=verbose,
        logger=logger
    )
    
    return results
