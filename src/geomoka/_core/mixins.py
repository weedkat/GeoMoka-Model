from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional
import logging
from tensorboard import SummaryWriter

import torch


class DeviceMixin:
    """Provide `to(device)` behavior for wrappers that expose `self.model`."""

    device: str

    def to(self, device: str = 'auto'):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device not in ('cuda', 'cpu'):
            raise ValueError("Device must be 'auto', 'cuda', or 'cpu'")
        self.model = self.model.to(device)
        self.device = device
        return self
    

class LoggingMixin:
    """Provide a configured logger instance as `self.logger`."""

    logger: logging.Logger

    def build_logger(self, name: str = 'geomoka', level: int = logging.INFO, log_file: str | Path | None = None) -> logging.Logger:
        """Create or reuse a configured logger."""
        if hasattr(self, 'logger'):
            return self.logger
        self.logger = build_logger(name=name, level=level, log_file=log_file)
        return self.logger


class TensorboardMixin:
    """Provide a TensorBoard SummaryWriter as `self.writer`."""

    writer: SummaryWriter

    def build_writer(self, log_dir: str | Path):
        """Create a TensorBoard SummaryWriter."""
        if hasattr(self, 'writer'):
            return self.writer
        self.writer = SummaryWriter(log_dir=log_dir)
        return self.writer


class MLFlowMixin:
    """Provide an MLFlow client and experiment tracking."""

    mlflow_client: Any  # Placeholder for actual MLFlow client type
    mlflow_experiment_id: Optional[str]

    def setup_mlflow(self, experiment_name: str):
        """Initialize MLFlow client and set experiment."""
        import mlflow
        self.mlflow_client = mlflow.tracking.MlflowClient()
        experiment = self.mlflow_client.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.mlflow_experiment_id = self.mlflow_client.create_experiment(experiment_name)
        else:
            self.mlflow_experiment_id = experiment.experiment_id


def build_logger(name: str = 'geomoka', level: int = logging.INFO, log_file: Optional[str | Path] = None) -> logging.Logger:
	"""Create or reuse a configured logger."""
	logger = logging.getLogger(name)
	logger.setLevel(level)
	logger.propagate = False

	if logger.handlers:
		return logger

	formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')

	stream_handler = logging.StreamHandler()
	stream_handler.setFormatter(formatter)
	logger.addHandler(stream_handler)

	if log_file is not None:
		log_file = Path(log_file)
		log_file.parent.mkdir(parents=True, exist_ok=True)
		file_handler = logging.FileHandler(log_file)
		file_handler.setFormatter(formatter)
		logger.addHandler(file_handler)

	return logger