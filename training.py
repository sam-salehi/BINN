from typing import Dict, Optional
import torch

from bruno.nn.modules import Encoder
from bruno.learn import Hyperparameters
from learn import TrainModel
from encoder import ANN, GAT, GCN

def train_models(
    data,
    map_df,
    device: torch.device,
    epochs: int = 2000,
    patience: int=50,
    save_dir: Optional[str] = None,
) -> Dict[str, torch.nn.Module]:
    args = Hyperparameters()
    args.epochs = epochs
    args.num_node_features = data.num_node_features
    args.num_classes = int(len(data.y.unique()))
    args.cuda = (device.type == "cuda")
    args.patience = patience
    args.device = device

    trained: Dict[str, torch.nn.Module] = {}
    trainers = {}

    for method, model in zip(["ANN", "GCNConv", "GATConv"],[ANN,GCN,GAT]):
        args.method = method
        model = Encoder(map_df, args=args, bias=False)
        trainer = TrainModel(model=model, graph=data, args=args)
        trainer.learn()
        trained[method] = model
        trainers[method] = trainer

    if save_dir:
        torch.save(trained["GCNConv"], f"{save_dir}/GCN.pt")
        torch.save(trained["GATConv"], f"{save_dir}/GAT.pt")
        torch.save(trained["ANN"], f"{save_dir}/ANN.pt")

    return trained, trainers