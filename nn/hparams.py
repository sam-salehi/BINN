"""
Central Hyperparameters config for graph models. One object replaces the many
kwargs that were being threaded through factory → training → model.train().
"""
from __future__ import annotations

import torch
from typing import Any


class Hyperparameters:
    """
    Everything that is decided before training starts. Passed as a single object
    to the factory, training loop, and encoder constructors (which read .num_node_features,
    .num_classes, .method, .lr, .w_decay).
    """
    def __init__(
        self,
        *,
        num_node_features: int | None = None,
        num_classes: int | None = None,
        lr: float = 0.05,
        w_decay: float = 5e-2,
        dropout: float = 0.3,
        epochs: int = 1000,
        patience: int = 5,
        min_delta: float = 0.0,
        cuda: bool = True,
        device: Any = None,
        method: str = "GCNConv",
        heads: int = 1,
        bias: bool = False,
        save_dir: str | None = None,
    ):
        self.num_node_features = num_node_features
        self.num_classes = num_classes
        self.lr = lr
        self.w_decay = w_decay
        self.dropout = dropout
        self.epochs = epochs
        self.patience = patience
        self.min_delta = min_delta
        self.cuda = cuda
        self.method = method
        self.heads = heads
        self.bias = bias
        self.save_dir = save_dir

        if device is not None:
            self.device = device
        elif cuda and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

    def for_method(self, method: str) -> Hyperparameters:
        """Return a shallow copy with .method set to the encoder's expected value (e.g. 'GCNConv')."""
        import copy
        hp = copy.copy(self)
        hp.method = method
        return hp

    @property
    def weight_decay(self) -> float:
        """Alias so code that reads .weight_decay still works."""
        return self.w_decay

    def save_path(self, method_name: str) -> str | None:
        """Build per-model save path from save_dir, or None if save_dir is unset."""
        if self.save_dir is None:
            return None
        return f"{self.save_dir}/{method_name}_scvi.pt"
