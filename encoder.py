import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, Linear
import torch.nn.functional as F
import pandas as pd 
import numpy as np

# scvi-tools imports for BaseModelClass integration
from scvi.model.base import BaseModelClass
from scvi.module.base import BaseModuleClass
from scvi.data import AnnDataManager
from scvi.data.fields import LayerField
from pytorch_lightning import LightningModule


class Encoder(nn.Module):
    def __init__(self,map,args,bias:bool=True):
        super().__init__()
        self.map = map 
        self.map_f = self.map.apply(lambda x: pd.factorize(x)[0])
        self.args = args
        self.bias = bias

 

        if self.map.shape[0] == 2:
            units = list(self.map_f.to_numpy()[0])
        else:
            units = list(self.map_f.nunique())
        units[0] = self.args.num_node_features
        self.units = units 

        self.masks = self.generate_masks(self.map_f)
        self.modules = []


    def forward(self, data, edge_index):
        """Forward method for graph-based encoders (GCN, GAT)"""
        x = data
        outputs = []

        for i, layers in enumerate(self.layers):
            for layer in layers:
                if isinstance(layer,(nn.BatchNorm1d,Linear,nn.Linear)):
                    x = layer(x)
                    if i < len(self.layers) - 1:
                        x = F.relu(x)
                else:
                    x = F.relu(layer(x,edge_index))
            # Keep tensors on device during forward pass (don't convert to numpy)
            outputs.append(x.detach())
        
        return x, outputs


    def generate_masks(self,map_f): 
        '''
        Generates a list of binary masks (torch.FloatTensor) to be used with torch.nn.utils.prune.
        For each consecutive pair of columns in the factorized map `map_f`, creates a mask matrix
        of shape (n_next, n_curr) where entry (to, from) == 1.0 if there exists at least one
        item mapping from group `from` at level i to group `to` at level i+1.
        '''

        num_levels = map_f.shape[1]
        masks: list[torch.Tensor] = []

        for level_idx in range(num_levels - 1):
            left = map_f.iloc[:, level_idx].to_numpy().astype(np.int64)
            right = map_f.iloc[:, level_idx + 1].to_numpy().astype(np.int64)

            # Determine group counts directly
            n_left = int(left.max() + 1) if left.size > 0 else 0
            n_right = int(right.max() + 1) if right.size > 0 else 0

            mask_np = np.zeros((n_right, n_left), dtype=np.float32)
            if left.size > 0:
                mask_np[right, left] = 1.0

            masks.append(torch.from_numpy(mask_np))

        # TODO: write test to make sure dimesnisons match with self.units        

        if self.args.num_classes is not None:
            output_size = masks[-1].shape[0]
            masks.append(torch.ones((self.args.num_classes,output_size),dtype=torch.float32))


        return masks




class ANN(Encoder):
    def __init__(self,map,args,bias:bool=True):
        super().__init__(map,args,bias)
        assert len(self.modules) == 0
        assert args.method == "ANN"

        for i in range(len(self.units) - 1):

            linear =  Linear(self.units[i],self.units[i+1],bias=self.bias)
            prune.custom_from_mask(linear,"weight",self.masks[i])

            self.modules.append(nn.Sequential(
                linear,
                nn.BatchNorm1d(self.units[i+1])
            ))
    
        if self.args.num_classes is not None:
            cls_linear = nn.Linear(self.units[-1],self.args.num_classes,bias = self.bias)
            # Apply final all-ones mask if provided
            if len(self.masks) == len(self.units):  # extra mask added for classifier
                prune.custom_from_mask(cls_linear, "weight", self.masks[-1])
            self.modules.append(nn.Sequential(cls_linear))
        self.layers = nn.Sequential(*self.modules)


class GCN(Encoder):
    def __init__(self,map,args,bias:bool=True):
        super().__init__(map,args,bias)
        assert len(self.modules) == 0
        assert args.method == "GCNConv"

        for i in range(len(self.units)-1):
            conv = GCNConv(self.units[i],self.units[i+1],bias=self.bias)
            # GCNConv stores parameters in the internal Linear module `lin`
            prune.custom_from_mask(conv.lin,"weight",self.masks[i])

            self.modules.append(nn.Sequential(conv, nn.BatchNorm1d(self.units[i+1])))

        if self.args.num_classes is not None:
            self.modules.append(nn.Sequential(nn.Linear(self.units[-1],self.args.num_classes,bias=self.bias)))
        self.layers = nn.Sequential(*self.modules)


class GAT(Encoder):
    def __init__(self,map,args,bias:bool=True,heads=1):
        super().__init__(map,args,bias)
        assert len(self.modules) == 0
        assert args.method == "GATConv"

        for i in range(len(self.units)-1):
            GAT = GATConv(self.units[i],self.units[i+1],bias=self.bias,heads=heads)
            prune.custom_from_mask(GAT.lin,"weight",self.masks[i])

            self.modules.append(nn.Sequential(
                GAT,
                nn.BatchNorm1d(self.units[i+1])
            ))

        if self.args.num_classes is not None:
            self.modules.append(nn.Sequential(nn.Linear(self.units[-1],self.args.num_classes,bias=self.bias)))
        self.layers = nn.Sequential(*self.modules)


# ============================================================================
# scvi-tools BaseModelClass Integration
# ============================================================================

class GraphEncoderModule(BaseModuleClass):
    """
    scvi-tools BaseModuleClass wrapper for graph-based encoders.
    This module wraps the Encoder classes to work with scvi-tools training infrastructure.
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
        
        # Define loss function based on task type
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
        """
        Compute loss for the model.
        
        Args:
            output: Model predictions
            y_true: Ground truth labels
            mask: Optional mask to select specific nodes
        
        Returns:
            loss: Computed loss value
        """
        if mask is not None:
            return self.criterion(output[mask], y_true[mask])
        else:
            return self.criterion(output, y_true)
    
    def inference(self, x, edge_index):
        """
        Inference method for getting model predictions.
        """
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
        """
        Training step for graph data.
        """
        # Move data to the same device as the model (handles MPS, CUDA, CPU)
        x = self.graph_data.x.to(self.device)
        edge_index = self.graph_data.edge_index.to(self.device)
        y_true = self.y.to(self.device)
        train_mask = self.train_mask.to(self.device)
        
        # Forward pass
        output, outputs = self.module(x, edge_index, mask=train_mask)
        
        # Compute loss on training mask
        loss = self.module.loss(output, y_true, mask=train_mask)
        
        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step for graph data.
        """
        # Move data to the same device as the model (handles MPS, CUDA, CPU)
        x = self.graph_data.x.to(self.device)
        edge_index = self.graph_data.edge_index.to(self.device)
        y_true = self.y.to(self.device)
        val_mask = self.val_mask.to(self.device)
        
        # Forward pass
        output, outputs = self.module(x, edge_index, mask=val_mask)
        
        # Compute loss on validation mask
        loss = self.module.loss(output, y_true, mask=val_mask)
        
        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """
        Configure optimizer for training.
        """
        optimizer = torch.optim.Adam(
            self.module.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer


class GraphModel(BaseModelClass):
    """
    scvi-tools BaseModelClass for graph-based classification/regression.
    This class provides a high-level API for training graph models with PyTorch Lightning.
    """
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
        # Initialize BaseModelClass first (signature: BaseModelClass(adata=None, registry=None))
        super().__init__(adata=adata, registry=registry, **kwargs)
        
        # Create and set the module after BaseModelClass initialization
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
        # Create a minimal setup - register the data layer
        # For graph models, we mainly need this for API compatibility
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
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create custom training plan (LightningModule)
        training_plan = GraphTrainingPlan(
            module=self.module,
            graph_data=self.graph_data,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        
        # Create early stopping callback
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            patience=patience,
            mode="min",
        )
        
        # Create a dummy dataloader (graph data is full-batch, handled in training_step)
        # We just need something to iterate over for Lightning
        dummy_dataset = TensorDataset(torch.zeros(1))
        dummy_loader = DataLoader(dummy_dataset, batch_size=1)
        
        # Create trainer
        trainer = Trainer(
            max_epochs=max_epochs,
            check_val_every_n_epoch=check_val_every_n_epoch,
            callbacks=[early_stop_callback],
            enable_progress_bar=True,
            **train_kwargs
        )
        
        # Train
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
            # Return the output before the final classification layer if available
            if len(outputs) > 0:
                return outputs[-1]  # Last intermediate output
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
                # For classification, return class probabilities or logits
                return F.softmax(output, dim=-1) if mask is None else F.softmax(output[mask], dim=-1)
            else:
                # For regression, return raw output
                return output if mask is None else output[mask]


def create_graph_model(
    encoder_class,
    map_df,
    graph_data,
    args,
    bias: bool = True,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    adata=None,
    registry=None,
):
    """
    Helper function to create a GraphModel from an encoder class.
    
    Args:
        encoder_class: The encoder class (ANN, GCN, or GAT)
        map_df: Pathway mapping dataframe
        graph_data: PyTorch Geometric data object
        args: Arguments object with num_node_features, num_classes, method, etc.
        bias: Whether to use bias in layers
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        adata: AnnData object (optional, for scvi-tools compatibility)
        registry: Registry object (optional, for scvi-tools compatibility)
    
    Returns:
        GraphModel instance ready for training
    
    Example:
        >>> from encoder import ANN, GCN, GAT, create_graph_model
        >>> args.method = "GCNConv"
        >>> model = create_graph_model(GCN, map_df, graph_data, args, adata=adata)
        >>> model.train(max_epochs=400, patience=200)
    """
    # Create the encoder instance
    encoder = encoder_class(map_df, args=args, bias=bias)
    
    # Create and return the GraphModel
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