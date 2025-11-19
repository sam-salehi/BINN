import torch.nn as nn
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, Linear
import torch.nn.functional as F
import pandas as pd 


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
            outputs.append(x.cpu().detach().numpy())
        
        return x, outputs


class ANN(Encoder):
    def __init__(self,map,args,bias:bool=True):
        super().__init__(map,args,bias)
        assert len(self.modules) == 0
        assert args.method == "ANN"

        for i in range(len(self.units) - 1):
            self.modules.append(nn.Sequential(
                Linear(self.units[i],self.units[i+1],bias=self.bias),
                nn.BatchNorm1d(self.units[i+1])
            ))
    
        if self.args.num_classes is not None:
            self.modules.append(nn.Sequential(nn.Linear(self.units[-1],self.args.num_classes,bias = self.bias)))
        self.layers = nn.Sequential(*self.modules)

    
    # def forward(self,data,edge_index=None):
    #     x = data 

    #     outputs = [] # outputs of each layer on the way there. 

    #     for i, seq_module in enumerate(self.layers):
    #         x = seq_module(x)
    #         if i < len(self.layers) - 1:
    #             x = F.relu(x)
    #         outputs.append(x.cpu().detach())
    #     return x, outputs

class GCN(Encoder):
    def __init__(self,map,args,bias:bool=True):
        super().__init__(map,args,bias)
        assert len(self.modules) == 0
        assert args.method == "GCNConv"

    
        self.modules = [nn.Sequential(
            GCNConv(self.units[i],self.units[i+1],bias=self.bias),
            nn.BatchNorm1d(self.units[i+1])
        ) for i in range(len(self.units)-1)]

        if self.args.num_classes is not None:
            self.modules.append(nn.Sequential(nn.Linear(self.units[-1],self.args.num_classes,bias=self.bias)))
        self.layers = nn.Sequential(*self.modules)


class GAT(Encoder):
    def __init__(self,map,args,bias:bool=True,heads=1):
        super().__init__(map,args,bias)
        assert len(self.modules) == 0
        assert args.method == "GATConv"

        self.modules = [nn.Sequential(
            GATConv(self.units[i],self.units[i+1],bias=self.bias,heads=heads),
            nn.BatchNorm1d(self.units[i+1])
        ) for i in range(len(self.units)-1)]

        if self.args.num_classes is not None:
            self.modules.append(nn.Sequential(nn.Linear(self.units[-1],self.args.num_classes,bias=self.bias)))
        self.layers = nn.Sequential(*self.modules)


                        







