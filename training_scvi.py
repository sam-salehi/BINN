"""
Training module for graph models using scvi-tools BaseModelClass.
Uses GraphModelFactory so one function trains any registered method (ANN, GCN, GAT, ...).
"""

from typing import Optional
import torch
from nn import GraphModel, GraphModelFactory


class TrainingArgs:
    """Arguments class for encoder compatibility (must be defined at module level for pickling)."""
    def __init__(self, num_node_features: int, num_classes: int, method: str, lr: float, w_decay: float):
        self.num_node_features = num_node_features
        self.num_classes = num_classes
        self.method = method
        self.lr = lr
        self.w_decay = w_decay


def train_graph_model(
    method: str,
    graph_data,
    map_df,
    num_node_features: int,
    num_classes: int,
    device: torch.device,
    *,
    epochs: int = 400,
    patience: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    bias: bool = False,
    save_path: Optional[str] = None,
    adata=None,
    registry=None,
    **encoder_kwargs,
):
    """
    Train a graph model by method name (ANN, GCN, GAT, or any registered encoder).

    Args:
        method: Registered name ("ANN", "GCN", "GAT"). Add more via GraphModelFactory.register(...).
        graph_data: PyTorch Geometric data object with x, edge_index, y, train_mask, val_mask
        map_df: Pathway mapping dataframe
        num_node_features: Number of input node features
        num_classes: Number of output classes
        device: Torch device (CPU or CUDA)
        epochs: Maximum number of training epochs
        patience: Early stopping patience
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        bias: Whether to use bias in layers
        save_path: Optional path to save the trained model
        adata: AnnData object (optional, for scvi-tools compatibility)
        registry: Registry object (optional)
        **encoder_kwargs: Passed to the encoder (e.g. heads=2 for GAT)

    Returns:
        Trained GraphModel instance
    """
    args = TrainingArgs(
        num_node_features=num_node_features,
        num_classes=num_classes,
        method=GraphModelFactory.get_args_method(method),
        lr=lr,
        w_decay=weight_decay,
    )

    if adata is not None:
        GraphModel.setup_anndata(adata)

    model = GraphModelFactory.create(
        method=method,
        map_df=map_df,
        graph_data=graph_data,
        args=args,
        bias=bias,
        lr=lr,
        weight_decay=weight_decay,
        adata=adata,
        registry=registry,
        **encoder_kwargs,
    )

    train_kwargs = {
        "max_epochs": epochs,
        "patience": patience,
        "check_val_every_n_epoch": 1,
    }
    train_kwargs.update(GraphModelFactory.get_train_kwargs(method))

    device_msg = train_kwargs.get("accelerator", str(device))
    print(f"Training {method} model for up to {epochs} epochs with patience {patience}...")
    print(f"Using device: {device_msg}")
    if encoder_kwargs:
        print(f"Encoder kwargs: {encoder_kwargs}")

    model.train(**train_kwargs)

    if save_path:
        print(f"Saving model to {save_path}")
        torch.save(model, save_path)

    return model
