import torch
import torch.nn as nn
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
        self.masks = self.generate_masks(self.map_f)
 

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


    def generate_masks(self,map_f): 
        '''
        Generates a list of binary masks (torch.FloatTensor) to be used with torch.nn.utils.prune.
        For each consecutive pair of columns in the factorized map `map_f`, creates a mask matrix
        of shape (n_next, n_curr) where entry (to, from) == 1.0 if there exists at least one
        item mapping from group `from` at level i to group `to` at level i+1. Values of -1 are ignored.
        '''
        # Ensure DataFrame input
        if not isinstance(map_f, pd.DataFrame):
            map_f = pd.DataFrame(map_f)

        num_levels = map_f.shape[1]
        masks: list[torch.Tensor] = []

        for level_idx in range(num_levels - 1):
            left = map_f.iloc[:, level_idx].to_numpy()
            right = map_f.iloc[:, level_idx + 1].to_numpy()

            # Ignore missing (-1) codes from pandas.factorize
            valid = (left >= 0) & (right >= 0)
            left_codes = left[valid].astype(np.int64)
            right_codes = right[valid].astype(np.int64)

            # Determine group counts
            n_left = int(left_codes.max() + 1) if left_codes.size > 0 else int(max(0, (left[left >= 0].max() + 1)) if (left >= 0).any() else 0)
            n_right = int(right_codes.max() + 1) if right_codes.size > 0 else int(max(0, (right[right >= 0].max() + 1)) if (right >= 0).any() else 0)

            mask_np = np.zeros((n_right, n_left), dtype=np.float32)
            if left_codes.size > 0:
                mask_np[right_codes, left_codes] = 1.0

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


                        







