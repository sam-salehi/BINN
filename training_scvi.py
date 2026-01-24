"""
Training module for graph models using scvi-tools BaseModelClass.
This module provides functions to train models with PyTorch Lightning.
"""

from typing import Optional
import torch
from encoder import ANN, GCN, GAT, create_graph_model

class TrainingArgs:
    """Arguments class for encoder compatibility (must be defined at module level for pickling)."""
    def __init__(self, num_node_features: int, num_classes: int, method: str, lr: float, w_decay: float):
        self.num_node_features = num_node_features
        self.num_classes = num_classes
        self.method = method
        self.lr = lr
        self.w_decay = w_decay


def train_ann_model(
    graph_data,
    map_df,
    num_node_features: int,
    num_classes: int,
    device: torch.device,
    epochs: int = 400,
    patience: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    bias: bool = False,
    save_path: Optional[str] = None,
    adata=None,
    registry=None,
):
    """
    Train an ANN model using scvi-tools BaseModelClass with PyTorch Lightning.
    
    Args:
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
    Returns:
        Trained GraphModel instance
    """
    # Create args object for compatibility with encoder classes
    args = TrainingArgs(
        num_node_features=num_node_features,
        num_classes=num_classes,
        method="ANN",
        lr=lr,
        w_decay=weight_decay,
    )
    
    # Note: Don't move graph_data to device here.
    # PyTorch Lightning handles device management automatically.
    # Data is moved to self.device in training_step/validation_step.
    
    # Setup AnnData with GraphModel if adata is provided
    # This is required by scvi-tools BaseModelClass
    if adata is not None:
        from encoder import GraphModel
        GraphModel.setup_anndata(adata)
    
    # Create the model using the helper function
    model = create_graph_model(
        encoder_class=ANN,
        map_df=map_df,
        graph_data=graph_data,
        args=args,
        bias=bias,
        lr=lr,
        weight_decay=weight_decay,
        adata=adata,
        registry=registry,
    )
    
    # Train the model (PyTorch Lightning handles device management)
    print(f"Training ANN model for up to {epochs} epochs with patience {patience}...")
    print(f"Using device: {device}")
    model.train(
        max_epochs=epochs,
        patience=patience,
        check_val_every_n_epoch=1,
    )
    
    # Save model if path provided
    if save_path:
        print(f"Saving model to {save_path}")
        torch.save(model, save_path)
    
    return model


def train_gcn_model(
    graph_data,
    map_df,
    num_node_features: int,
    num_classes: int,
    device: torch.device,
    epochs: int = 400,
    patience: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    bias: bool = False,
    save_path: Optional[str] = None,
    adata=None,
    registry=None,
):
    """
    Train a GCN (Graph Convolutional Network) model using scvi-tools BaseModelClass with PyTorch Lightning.
    
    Args:
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
        registry: Registry object (optional, for scvi-tools compatibility)
    Returns:
        Trained GraphModel instance
    """
    # Create args object for compatibility with encoder classes
    args = TrainingArgs(
        num_node_features=num_node_features,
        num_classes=num_classes,
        method="GCNConv",
        lr=lr,
        w_decay=weight_decay,
    )
    
    # Setup AnnData with GraphModel if adata is provided
    if adata is not None:
        from encoder import GraphModel
        GraphModel.setup_anndata(adata)
    
    # Create the model using the helper function
    model = create_graph_model(
        encoder_class=GCN,
        map_df=map_df,
        graph_data=graph_data,
        args=args,
        bias=bias,
        lr=lr,
        weight_decay=weight_decay,
        adata=adata,
        registry=registry,
    )
    
    # Train the model
    print(f"Training GCN model for up to {epochs} epochs with patience {patience}...")
    print(f"Using device: {device}")
    model.train(
        max_epochs=epochs,
        patience=patience,
        check_val_every_n_epoch=1,
    )
    
    # Save model if path provided
    if save_path:
        print(f"Saving model to {save_path}")
        torch.save(model, save_path)
    
    return model


def train_gat_model(
    graph_data,
    map_df,
    num_node_features: int,
    num_classes: int,
    device: torch.device,
    epochs: int = 400,
    patience: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    bias: bool = False,
    heads: int = 1,
    save_path: Optional[str] = None,
    adata=None,
    registry=None,
):
    """
    Train a GAT (Graph Attention Network) model using scvi-tools BaseModelClass with PyTorch Lightning.
    
    Args:
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
        heads: Number of attention heads for GAT layers
        save_path: Optional path to save the trained model
        adata: AnnData object (optional, for scvi-tools compatibility)
        registry: Registry object (optional, for scvi-tools compatibility)
    Returns:
        Trained GraphModel instance
    """
    # Create args object for compatibility with encoder classes
    args = TrainingArgs(
        num_node_features=num_node_features,
        num_classes=num_classes,
        method="GATConv",
        lr=lr,
        w_decay=weight_decay,
    )
    
    # Setup AnnData with GraphModel if adata is provided
    if adata is not None:
        from encoder import GraphModel
        GraphModel.setup_anndata(adata)
    
    # Create the model using the helper function
    # Note: GAT encoder accepts heads parameter, but create_graph_model doesn't pass it
    # We create the encoder directly to support the heads parameter
    encoder = GAT(map_df, args=args, bias=bias, heads=heads)
    
    from encoder import GraphModel
    model = GraphModel(
        encoder=encoder,
        graph_data=graph_data,
        num_node_features=num_node_features,
        num_classes=num_classes,
        lr=lr,
        weight_decay=weight_decay,
        adata=adata,
        registry=registry,
    )
    
    # Train the model
    # Note: GAT uses scatter operations that aren't fully supported on MPS (Apple Metal),
    # so we force CPU training to avoid crashes
    print(f"Training GAT model for up to {epochs} epochs with patience {patience}...")
    print(f"Using device: cpu (forced - MPS scatter ops not supported) | Attention heads: {heads}")
    model.train(
        max_epochs=epochs,
        patience=patience,
        check_val_every_n_epoch=1,
        accelerator="cpu",  # Force CPU to avoid MPS scatter operation issues
    )
    
    # Save model if path provided
    if save_path:
        print(f"Saving model to {save_path}")
        torch.save(model, save_path)
    
    return model

