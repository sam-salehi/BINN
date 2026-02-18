"""
scvi-tools integration: GraphEncoderModule, GraphTrainingPlan, GraphModel, GraphModelFactory.
Depends on .models for Encoder/ANN/GCN/GAT and .hparams for Hyperparameters.
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

from .models import ANN, GCN, GAT
from learn.hparams import Hyperparameters


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class GraphModelFactory:
    """
    Registry-based factory. Register encoder classes once; create fully-configured
    GraphModel instances from a Hyperparameters config + graph data.
    """
    _registry: dict[str, type] = {}
    _default_encoder_kwargs: dict[str, dict] = {}
    _args_method: dict[str, str] = {}
    _train_kwargs: dict[str, dict] = {}

    @classmethod
    def register(
        cls,
        name: str,
        encoder_class: type,
        *,
        args_method: str | None = None,
        default_encoder_kwargs: dict | None = None,
        train_kwargs: dict | None = None,
    ):
        """
        Register an encoder class.
        name: key callers use ("ANN", "GCN", "GAT").
        args_method: value the encoder asserts on args.method (e.g. "GCNConv").
        default_encoder_kwargs: merged into encoder kwargs unless overridden.
        train_kwargs: extra kwargs merged into model.train() (e.g. accelerator="cpu").
        """
        cls._registry[name] = encoder_class
        cls._args_method[name] = args_method if args_method is not None else name
        cls._default_encoder_kwargs[name] = dict(default_encoder_kwargs or {})
        cls._train_kwargs[name] = dict(train_kwargs or {})

    @classmethod
    def get_train_kwargs(cls, name: str) -> dict:
        return dict(cls._train_kwargs.get(name, {}))

    @classmethod
    def create(
        cls,
        name: str,
        map_df,
        graph_data,
        config: Hyperparameters,
        *,
        adata=None,
        **encoder_kwargs,
    ) -> "GraphModel":
        """
        Build a GraphModel. All hyperparams come from config; encoder_kwargs
        override defaults (e.g. heads=2).
        """
        encoder_class = cls._registry.get(name)
        if encoder_class is None:
            raise ValueError(f"Unknown method {name!r}. Registered: {list(cls._registry)}")

        args_method = cls._args_method[name]
        hp = config.for_method(args_method)

        defaults = cls._default_encoder_kwargs.get(name, {})
        enc_kw = {**defaults, **encoder_kwargs}
        if "heads" in defaults and "heads" not in encoder_kwargs:
            enc_kw["heads"] = config.heads

        encoder = encoder_class(map_df, args=hp, bias=hp.bias, **enc_kw)

        return GraphModel(
            encoder=encoder,
            graph_data=graph_data,
            config=config,
            adata=adata,
        )


GraphModelFactory.register("ANN", ANN, args_method="ANN")
GraphModelFactory.register("GCN", GCN, args_method="GCNConv")
GraphModelFactory.register(
    "GAT", GAT,
    args_method="GATConv",
    default_encoder_kwargs={"heads": 1},
    train_kwargs={"accelerator": "cpu"},
)


# scvi-tools wrappers


class GraphEncoderModule(BaseModuleClass):
    """Wraps an Encoder to work with scvi-tools training infrastructure."""
    def __init__(self, encoder: nn.Module, config: Hyperparameters):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.criterion = nn.MSELoss() if config.num_classes is None else nn.CrossEntropyLoss()

    def forward(self, x, edge_index, mask=None):
        output, outputs = self.encoder(x, edge_index)
        return output, outputs

    def loss(self, output, y_true, mask=None):
        if mask is not None:
            return self.criterion(output[mask], y_true[mask])
        return self.criterion(output, y_true)

    def inference(self, x, edge_index):
        self.eval()
        with torch.no_grad():
            return self.forward(x, edge_index)


class GraphTrainingPlan(LightningModule):
    """PyTorch Lightning module for full-batch graph training with train/val masks."""
    def __init__(self, module: GraphEncoderModule, graph_data, config: Hyperparameters):
        super().__init__()
        self.module = module
        self.graph_data = graph_data
        self.train_mask = graph_data.train_mask
        self.val_mask = graph_data.val_mask
        self.y = graph_data.y
        self.config = config

    def training_step(self, batch, batch_idx):
        x = self.graph_data.x.to(self.device)
        edge_index = self.graph_data.edge_index.to(self.device)
        y_true = self.y.to(self.device)
        mask = self.train_mask.to(self.device)
        output, _ = self.module(x, edge_index, mask=mask)
        loss = self.module.loss(output, y_true, mask=mask)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = self.graph_data.x.to(self.device)
        edge_index = self.graph_data.edge_index.to(self.device)
        y_true = self.y.to(self.device)
        mask = self.val_mask.to(self.device)
        output, _ = self.module(x, edge_index, mask=mask)
        loss = self.module.loss(output, y_true, mask=mask)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.module.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.w_decay,
        )


class GraphModel(BaseModelClass):
    """High-level API: create, train, predict, get_latent_representation."""
    def __init__(
        self,
        encoder: nn.Module,
        graph_data,
        config: Hyperparameters,
        adata=None,
        registry=None,
        **kwargs,
    ):
        super().__init__(adata=adata, registry=registry, **kwargs)
        self.module = GraphEncoderModule(encoder, config)
        self.graph_data = graph_data
        self.config = config

    @classmethod
    def setup_anndata(cls, adata, **kwargs):
        adata_manager = AnnDataManager(
            fields=[LayerField(registry_key="X", layer=None, is_count_data=False)],
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
        return adata

    def train(
        self,
        max_epochs: int | None = None,
        patience: int | None = None,
        check_val_every_n_epoch: int = 1,
        **train_kwargs,
    ):
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import EarlyStopping

        max_epochs = max_epochs if max_epochs is not None else self.config.epochs
        patience = patience if patience is not None else self.config.patience

        plan = GraphTrainingPlan(self.module, self.graph_data, self.config)
        early_stop = EarlyStopping(monitor="val_loss", patience=patience, mode="min")

        dummy = DataLoader(TensorDataset(torch.zeros(1)), batch_size=1)
        trainer = Trainer(
            max_epochs=max_epochs,
            check_val_every_n_epoch=check_val_every_n_epoch,
            callbacks=[early_stop],
            enable_progress_bar=True,
            **train_kwargs,
        )
        trainer.fit(plan, train_dataloaders=dummy, val_dataloaders=dummy)
        return trainer

    def get_latent_representation(self, x=None, edge_index=None):
        if x is None:
            x = self.graph_data.x
        if edge_index is None:
            edge_index = self.graph_data.edge_index
        self.module.eval()
        with torch.no_grad():
            _, outputs = self.module(x, edge_index)
            return outputs[-1] if outputs else self.module(x, edge_index)[0]

    def predict(self, x=None, edge_index=None, mask=None):
        if x is None:
            x = self.graph_data.x
        if edge_index is None:
            edge_index = self.graph_data.edge_index
        self.module.eval()
        with torch.no_grad():
            output, _ = self.module(x, edge_index, mask=mask)
            if self.config.num_classes is not None:
                return F.softmax(output, dim=-1) if mask is None else F.softmax(output[mask], dim=-1)
            return output if mask is None else output[mask]
