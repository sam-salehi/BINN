"""
Training module for graph models using scvi-tools BaseModelClass.
Single entry point: train_graph_model(name, graph_data, map_df, config).
"""

import torch
from nn import GraphModel, GraphModelFactory
from nn.hparams import Hyperparameters


def train_graph_model(
    name: str,
    graph_data,
    map_df,
    config: Hyperparameters,
    *,
    adata=None,
    **encoder_kwargs,
) -> GraphModel:
    """
    Train a graph model by registered name (ANN, GCN, GAT, ...).

    Args:
        name: Registered factory name ("ANN", "GCN", "GAT").
        graph_data: PyG data object (x, edge_index, y, train_mask, val_mask).
        map_df: Pathway mapping dataframe.
        config: Hyperparameters instance (all scalars live here).
        adata: AnnData object (optional, for scvi-tools compatibility).
        **encoder_kwargs: Override encoder defaults (e.g. heads=4 for GAT).

    Returns:
        Trained GraphModel instance.
    """
    if adata is not None:
        GraphModel.setup_anndata(adata)

    model = GraphModelFactory.create(
        name, map_df, graph_data, config,
        adata=adata,
        **encoder_kwargs,
    )

    train_kwargs = {
        "max_epochs": config.epochs,
        "patience": config.patience,
        "check_val_every_n_epoch": 1,
    }
    train_kwargs.update(GraphModelFactory.get_train_kwargs(name))

    device_msg = train_kwargs.get("accelerator", str(config.device))
    print(f"Training {name} | epochs={config.epochs} patience={config.patience} "
          f"lr={config.lr} w_decay={config.w_decay} device={device_msg}")

    model.train(**train_kwargs)

    save_path = config.save_path(name)
    if save_path:
        print(f"Saving model to {save_path}")
        torch.save(model, save_path)

    return model
