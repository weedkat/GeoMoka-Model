import numpy as np
import logging
import os
from typing import Optional
import hashlib
import json

def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count

logs = set()

def init_log(
    name,
    level=logging.INFO,
    log_file: Optional[str] = None,
    add_console: bool = True,
    rank_filter: bool = True,
):
    """Initialize a logger once and optionally attach file/console handlers.

    Args:
        name: logger name.
        level: logging level.
        log_file: optional path to write logs; created only on rank 0.
        add_console: attach a console handler (rank 0 only if rank_filter=True).
        rank_filter: if True, drop records from non-zero ranks.
    """
    key = (name, level, log_file, add_console, rank_filter)
    if key in logs:
        return logging.getLogger(name)

    logs.add(key)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    rank = int(os.environ.get("SLURM_PROCID", 0)) if rank_filter else 0

    def _rank_filter(record):
        return (rank == 0) if rank_filter else True

    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)

    if add_console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.addFilter(_rank_filter)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if log_file and (rank == 0 or not rank_filter):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.addFilter(_rank_filter)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def generate_model_name(cfg):
    # Create a hashable copy excluding model_name to avoid circular dependency
    config_for_hash = {k: v for k, v in cfg.items() if k != 'model_name'}
    config_str = json.dumps(config_for_hash, sort_keys=True, default=str)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    model_cfg = cfg.get('model_cfg', {})
    base_name = model_cfg.get('model_name', 'model')
    return f"{base_name}_{config_hash}"