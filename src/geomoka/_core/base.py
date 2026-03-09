from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from geomoka._core.mixins import LoggingMixin, DeviceMixin


@runtime_checkable
class BaseModel(Protocol):
    """Task-agnostic model interface."""

    model_id: str # Unique identifier for the model architecture
    model: Any # The actual model instance (e.g. PyTorch nn.Module)
    model_spec: Dict[str, Any] # Model specification to reconstruct architecture (e.g. model name, encoder, weights, etc.)
    metadata: Optional[Dict[str, Any]] # Dataset metadata for informed training/inference
    inferencer: Optional[BaseInferencer] # Inferencer for running predictions

    def compile(self, *args, **kwargs) -> None:
        """Prepare optimizer/loss/dataloader settings for training."""
        ...

    def fit(self, *args, **kwargs) -> Any:
        """Train the model. Return training history."""
        ...
    
    def predict(self, *args, **kwargs) -> Any:
        """Run inference and return raw predictions."""
        ...
    
    def predict_proba(self, *args, **kwargs) -> Any:
        """Run inference and return class probabilities."""
        ...

    def evaluate(self, *args, **kwargs) -> Dict[str, Any]:
        """Evaluate the model and return metrics."""
        ...

    @classmethod
    def load(cls, load_path: str | Path) -> "BaseModel":
        """Load model artifact."""
        ...

    def save(self, save_path: str | Path) -> None:
        """Save model artifact."""
        ...
    
    def summary(self) -> Dict[str, Any]:
        """Return a summary of the model architecture and parameters."""
        print("Model Summary:")
        print(f"Model ID: {self.model_id}")
        print(f"Model Spec: {self.model_spec}")
        if self.metadata:
            print(f"Metadata: {self.metadata}")


@runtime_checkable
class BaseTrainer(Protocol):
    """Lifecycle orchestrator interface for training loops."""

    model: BaseModel

    def validate_init(self) -> None:
        """Validate that all necessary components are set before training."""
        ...
    
    def validate_run(self) -> None:
        """Validate training state at each epoch/iteration."""
        ...

    def run(self) -> None:
        """Run training lifecycle."""
        ...

    def save_checkpoint(self, save_dir: str | Path) -> None:
        """Save latest trainer checkpoint."""
        ...

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load trainer checkpoint."""
        ...


class BaseInferencer(Protocol):
    """Interface for model inference methods."""
    model: BaseModel

    def preprocess(self, *args, **kwargs) -> Any:
        """Preprocess raw inputs for the model."""
        ...
    
    def postprocess(self, *args, **kwargs) -> Any:
        """Postprocess raw model outputs into desired format."""
        ...

    