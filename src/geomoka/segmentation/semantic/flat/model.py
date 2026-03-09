from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
from geomoka._core.base import BaseModel
from geomoka._core.registry import Registry

from . import models # invoke for model registration
from . import method # invoke for method/trainer registration

from .dataset.metadata import MetadataInterpreter
from .inference import Inference
from .evaluate import inference_evaluate


def generate_model_id(arch: str, dataset: str) -> str:
    """Generate a unique model ID based on architecture and timestamp."""
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    return f"{arch}_{dataset}_{timestamp}"


class FlatSemSegModel(BaseModel):
    def __init__(self, arch, metadata, device='auto', **kwargs):
        self.metadata = metadata
        self.meta_inter = MetadataInterpreter(metadata)
        
        dataset = self.meta_inter.dataset
        self.model_id = generate_model_id(arch, dataset)
        
        self.model_spec = {
            'arch': arch,
            'in_channels': self.meta_inter.num_bands,
            'nclass': self.meta_inter.num_classes,
            **kwargs
        }

        self.model = Registry.build_model(arch, module=__name__, **self.model_spec)
        self.inferencer = None
        self.device = device

    def compile(self, method:str, transform_cfg:dict, **kwargs):
        kwargs['transform_cfg'] = transform_cfg
        self.trainer = Registry.build_trainer(method, self, module=__name__, **kwargs)
        self.inferencer = Inference(self, transform_cfg=transform_cfg)

    def fit(self, *args, **kwargs):
        # Paramerter validation in trainer, because it may have specific requirements (e.g. train_ulnabeled for semi-supervised)
        self.trainer.run(*args, **kwargs)

    def predict(self, data):
        return self.inferencer(data)
    
    def predict_proba(self, data):
        return self.inferencer(data)

    def evaluate(self, dataset):
        return inference_evaluate(dataset)

    def save(self, save_path: str | Path):
        # Save model state dict and metadata
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_spec': self.model_spec,
            'metadata': self.metadata,
            'transform_cfg': self.transform_cfg
        }, save_path)

    @classmethod
    def load(cls, load_path: str | Path):
        state = torch.load(load_path, map_location='cpu')
        model_spec = state['model_spec']
        metadata = state['metadata']
        model = cls(model_spec['arch'], metadata, **model_spec)
        model.model.load_state_dict(state['model_state_dict'])
        model.transform_cfg = state.get('transform_cfg')
        model.inferencer = Inference(model, model.transform_cfg)
        
        return model

    def lock_encoder(self):
        enc, _ = self.get_encoder_decoder_params()
        for p in enc:
            p.requires_grad = False

    def get_encoder_decoder_params(self):
        """Split parameters into encoder / decoder groups."""
        if hasattr(self.model, 'backbone'):
            enc = [p for p in self.model.backbone.parameters() if p.requires_grad]
            dec = [p for n, p in self.model.named_parameters() if 'backbone' not in n]
        elif hasattr(self.model, 'encoder'):
            enc = list(self.model.encoder.parameters())
            dec = [p for n, p in self.model.named_parameters() if not n.startswith('encoder')]
        else:
            raise ValueError("Model must have either 'backbone' or 'encoder' attribute to split parameters.")
        return enc, dec

    def summary(self):
        pass
