"""
scvi-tools integration: GraphEncoderModule, GraphTrainingPlan, GraphModel, create_graph_model.
Depends on .models for Encoder/ANN/GCN/GAT.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import LightningModule

from scvi.model.base import BaseModelClass
from scvi.module.base import BaseModuleClass
from scvi.data import AnnDataManager
from scvi.data.fields import LayerField

from .models import Encoder, ANN, GCN, GAT


# ---------------------------------------------------------------------------
# Graph model factory: register encoder types, create GraphModel by name
# ---------------------------------------------------------------------------

class GraphModelFactory:
    """
    Registry-based factory for GraphModel. Register encoder classes by method name;
    then create models with create_graph_model(method="GCN", ...) or create().
    Encoder-specific options (e.g. heads for GAT) are passed as kwargs and forwarded
    to the encoder constructor.
    """
    _registry: dict[str, type] = {}
    _default_encoder_kwargs: dict[str, dict] = {}

    @classmethod
    def register(cls, method: str, encoder_class: type, default_encoder_kwargs: dict | None = None):
        """Register an encoder class under a method name (e.g. 'ANN', 'GCN', 'GAT')."""
        cls._registry[method] = encoder_class
        cls._default_encoder_kwargs[method] = dict(default_encoder_kwargs or {})

    @classmethod
    def create(
        cls,
        method: str,
        map_df,
        graph_data,
        args,
        *,
        bias: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        adata=None,
        registry=None,
        **encoder_kwargs,
    ) -> "GraphModel":
        """Build a GraphModel by method name. encoder_kwargs (e.g. heads=2) are passed to the encoder."""
        encoder_class = cls._registry.get(method)
        if encoder_class is None:
            raise ValueError(f"Unknown method {method!r}. Registered: {list(cls._registry)}")
        defaults = cls._default_encoder_kwargs.get(method, {})
        kwargs = {**defaults, **encoder_kwargs}
        encoder = encoder_class(map_df, args=args, bias=bias, **kwargs)
        return GraphModel(
            encoder=encoder,
            graph_data=graph_data,
            num_node_features=args.num_node_features,
            num_classes=args.num_classes,
            lr=lr,
            weight_decay=weight_decay,
            adata=adata,
            registry=registry,
        )


GraphModelFactory.register("ANN", ANN)
GraphModelFactory.register("GCN", GCN)
GraphModelFactory.register("GAT", GAT, default_encoder_kwargs={"heads": 1})


class GraphEncoderModule(BaseModuleClass):
    """
    scvi-tools BaseModuleClass wrapper for graph-based encoders.
    Wraps the Encoder classes to work with scvi-tools training infrastructure.
    """
    def __init__(
        self,
        encoder: nn.Module,
        num_node_features: int,
        num_classes: int = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_node_features = num_node_features
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay

        if num_classes is None:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, edge_index, mask=None):
        """
        Forward pass through the encoder.

        Args:
            x: Node features tensor
            edge_index: Edge index tensor for graph convolutions
            mask: Optional mask to select specific nodes

        Returns:
            output: Model predictions
            outputs: Intermediate layer outputs
        """
        output, outputs = self.encoder(x, edge_index)
        return output, outputs

    def loss(self, output, y_true, mask=None):
        if mask is not None:
            return self.criterion(output[mask], y_true[mask])
        else:
            return self.criterion(output, y_true)

    def inference(self, x, edge_index):
        """Inference method for getting model predictions."""
        self.eval()
        with torch.no_grad():
            output, outputs = self.forward(x, edge_index)
        return output, outputs


class GraphTrainingPlan(LightningModule):
    """
    PyTorch Lightning module for graph-based models with train/val/test masks.
    """
    def __init__(
        self,
        module: GraphEncoderModule,
        graph_data,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.module = module
        self.graph_data = graph_data
        self.train_mask = graph_data.train_mask
        self.val_mask = graph_data.val_mask
        self.y = graph_data.y
        self.lr = lr
        self.weight_decay = weight_decay

    def training_step(self, batch, batch_idx):
        """Training step for graph data."""
        x = self.graph_data.x.to(self.device)
        edge_index = self.graph_data.edge_index.to(self.device)
        y_true = self.y.to(self.device)
        train_mask = self.train_mask.to(self.device)

        output, outputs = self.module(x, edge_index, mask=train_mask)
        loss = self.module.loss(output, y_true, mask=train_mask)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for graph data."""
        x = self.graph_data.x.to(self.device)
        edge_index = self.graph_data.edge_index.to(self.device)
        y_true = self.y.to(self.device)
        val_mask = self.val_mask.to(self.device)

        output, outputs = self.module(x, edge_index, mask=val_mask)
        loss = self.module.loss(output, y_true, mask=val_mask)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer for training."""
        optimizer = torch.optim.Adam(
            self.module.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer


class GraphModel(BaseModelClass):
    def __init__(
        self,
        encoder: nn.Module,
        graph_data,
        num_node_features: int,
        num_classes: int = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        adata=None,
        registry=None,
        **kwargs
    ):
        """
        Initialize the GraphModel.

        Args:
            encoder: The encoder model (ANN, GCN, or GAT instance)
            graph_data: PyTorch Geometric data object with x, edge_index, y, train_mask, val_mask
            num_node_features: Number of input node features
            num_classes: Number of output classes (None for regression)
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            adata: AnnData object (optional, for scvi-tools compatibility)
            registry: Registry object (optional, for scvi-tools compatibility)
        """
        super().__init__(adata=adata, registry=registry, **kwargs)

        self.module = GraphEncoderModule(
            encoder=encoder,
            num_node_features=num_node_features,
            num_classes=num_classes,
            lr=lr,
            weight_decay=weight_decay,
        )

        self.graph_data = graph_data
        self.num_node_features = num_node_features
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay

    @classmethod
    def setup_anndata(cls, adata, **kwargs):
        """
        Setup method for AnnData (required by scvi-tools BaseModelClass).
        This registers the AnnData with the model class for scvi-tools compatibility.
        """
        adata_manager = AnnDataManager(
            fields=[
                LayerField(registry_key="X", layer=None, is_count_data=False),
            ],
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
        return adata

    def train(
        self,
        max_epochs: int = 400,
        patience: int = 200,
        check_val_every_n_epoch: int = 1,
        **train_kwargs
    ):
        """
        Train the model using PyTorch Lightning Trainer directly.

        Args:
            max_epochs: Maximum number of training epochs
            patience: Early stopping patience
            check_val_every_n_epoch: How often to run validation
            **train_kwargs: Additional arguments for Trainer
        """
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import EarlyStopping

        training_plan = GraphTrainingPlan(
            module=self.module,
            graph_data=self.graph_data,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=patience,
            mode="min",
        )

        dummy_dataset = TensorDataset(torch.zeros(1))
        dummy_loader = DataLoader(dummy_dataset, batch_size=1)

        trainer = Trainer(
            max_epochs=max_epochs,
            check_val_every_n_epoch=check_val_every_n_epoch,
            callbacks=[early_stop_callback],
            enable_progress_bar=True,
            **train_kwargs
        )

        trainer.fit(training_plan, train_dataloaders=dummy_loader, val_dataloaders=dummy_loader)
        return trainer

    def get_latent_representation(self, x=None, edge_index=None):
        """
        Get latent representation from the model.

        Args:
            x: Optional node features (uses graph_data.x if None)
            edge_index: Optional edge index (uses graph_data.edge_index if None)

        Returns:
            latent: Latent representation
        """
        if x is None:
            x = self.graph_data.x
        if edge_index is None:
            edge_index = self.graph_data.edge_index

        self.module.eval()
        with torch.no_grad():
            output, outputs = self.module(x, edge_index)
            if len(outputs) > 0:
                return outputs[-1]
            return output

    def predict(self, x=None, edge_index=None, mask=None):
        """
        Get predictions from the model.

        Args:
            x: Optional node features (uses graph_data.x if None)
            edge_index: Optional edge index (uses graph_data.edge_index if None)
            mask: Optional mask to select specific nodes

        Returns:
            predictions: Model predictions
        """
        if x is None:
            x = self.graph_data.x
        if edge_index is None:
            edge_index = self.graph_data.edge_index

        self.module.eval()
        with torch.no_grad():
            output, outputs = self.module(x, edge_index, mask=mask)
            if self.num_classes is not None:
                return F.softmax(output, dim=-1) if mask is None else F.softmax(output[mask], dim=-1)
            else:
                return output if mask is None else output[mask]


# def create_graph_model(
#     method: str,
#     map_df,
#     graph_data,
#     args,
#     *,
#     bias: bool = True,
#     lr: float = 1e-3,
#     weight_decay: float = 1e-5,
#     adata=None,
#     registry=None,
#     **encoder_kwargs,
# ) -> GraphModel:
#     """
#     Create a GraphModel by registered method name. Encoder options (e.g. heads for GAT) go in encoder_kwargs.

#     Args:
#         method: Registered name ("ANN", "GCN", "GAT"). Add more via GraphModelFactory.register(...).
#         map_df: Pathway mapping dataframe
#         graph_data: PyTorch Geometric data object
#         args: Arguments with num_node_features, num_classes, etc.
#         bias: Whether to use bias in layers
#         lr: Learning rate
#         weight_decay: Weight decay for optimizer
#         adata: AnnData object (optional)
#         registry: Registry object (optional)
#         **encoder_kwargs: Passed to the encoder (e.g. heads=2 for GAT)

#     Returns:
#         GraphModel instance ready for training

#     Examples:
#         >>> model = create_graph_model("GCN", map_df, graph_data, args, adata=adata)
#         >>> model = create_graph_model("GAT", map_df, graph_data, args, heads=2)
#     """
#     return GraphModelFactory.create(
#         method=method,
#         map_df=map_df,
#         graph_data=graph_data,
#         args=args,
#         bias=bias,
#         lr=lr,
#         weight_decay=weight_decay,
#         adata=adata,
#         registry=registry,
#         **encoder_kwargs,
#     )
