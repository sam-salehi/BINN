import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, Linear
import torch.nn.functional as F
import pandas as pd 
import numpy as np


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
            outputs.append(x.cpu().detach().numpy())
        
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


                        







