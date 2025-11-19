
# Just making Bruno Encoder and Decoder for learning purposes. 
import torch.nn as nn 
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, Linear 
import torch.nn.functional as F 
import pandas as pd 



class Encoder(nn.Module):
    def __init__(self,map,args,bias=True):
        super().__init__()
        self.map = map 
        self.map_f = self.map.apply(lambda x: pd.factorize(x)[0])
        self.args = args 

        if self.map.shape[0] == 2: # input top output
            units = list(self.map_f.to_numpy()[0])
            units[0] = self.args.num_node_features
            self.units = units 
        else: # taking number of unique values per column
            units = list(self.map_f.unique())
            units[0] = self.args.num_node_features
            self.units = units 
        self.bias = bias 
        self.transition = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=self.args.dropout)
        )
        
        self.module_list = []

        for i in range(len(self.units)-1):
            if self.args.method == "GCNConv":
                self.module_list.append(nn.Sequential(
                    GCNConv(
                        self.units[i],
                        self.units[i+1],
                        bias = self.bias
                    ),
                    nn.BatchNorm1d(self.units[i+1])
                ))
            elif self.args.method == "GATConv":
                self.module_list.append(nn.Sequential(
                    GATConv(
                        self.units[i],
                        self.units[i+1],
                        bias=self.bias,
                        heads=1
                    )
                ))
            elif self.args.method == "ANN":
                self.module_list.append(nn.Sequential(
                    Linear(
                        self.units[i],
                        self.units[i+1],
                        bias=self.bias
                    ),
                    nn.BatchNorm1d(self.units[i+1])
                ))
            else:
                raise ValueError("args.type should be one of ...")
            
        if self.args.num_classes is not None:
            self.module_list.append(nn.Sequential(nn.Linear(self.units[-1],self.args.num_classes,bias=self.bias)))

        self.layers = nn.Sequential(*self.module_list)

    
    def forward(self,data,edge_index=None):
        x = data 
        outputs = []
        if self.args.method == "ANN":

            for i,layers in enumerate(self.layers):
                for layer in layers:
                    if i == len(layers)-1:
                        x = layer(x)
                    else:
                        x = F.relu(layer(x))
                
                outputs.append(x.cpu().detach().numpy())
        else:
            for i, layers in enumerate(self.layers):
                for layer in layers:
                    if isinstance(layer,nn.Linear) or isinstance(layer,nn.BatchNorm1d):
                        if i == (len(layers) - 1):
                            x = layer(x)
                        else:
                            x = F.relu(layer(x))
                    else:
                        x = F.relu(layer(x,edge_index))
                
                outputs.append(x.cpu().detach().numpy())
        try:
            if self.args.simple:
                output = x
            else:
                output = (x, outputs)
        except(AttributeError):
                output = (x, outputs)
        return output 





