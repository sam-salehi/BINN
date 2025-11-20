import torch
import torch.nn as nn
import numpy as np
import re
from torch import linalg as LA
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import scprep as scprep
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
import plotly.graph_objects as go
import pandas as pd
import torch.nn.functional as F

class TrainModel():
    def __init__(self,graph,model,args):
        self.args = args
        self.model = model.to(self.args.device)
        self.graph = graph.to(self.args.device)

        print(model.map_f)
        # quit()
        # self.mask = self.convert_map(self.model,self.args)

        if self.args.num_classes == None:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.w_decay)

        self.train_loss = []
        self.val_loss = []
        self.train_complete = False
        self.cim = None
        self.weights = []




    def learn(self):
        for epoch in range(self.args.epochs):
            if self.train_complete: return
            tl = self.train_epoch()
            self.train_loss.append(tl)

            vl = self.val()
            self.val_loss.append(vl)
            if epoch % 100 == 0:
                print(f'Epoch: {epoch+1} Training loss: {tl:.4f}')
                print(f'Epoch: {epoch+1} Validation loss: {vl:.4f}')

    def train_epoch(self):
        self.model.train()
        y_true = self.graph.y[self.graph.train_mask]

        self.optim.zero_grad()
        y_pred, outputs = self.model(self.graph.x,self.graph.edge_index) # what do graph.x and graph.edge_index look like?
        loss = self.criterion(y_pred[self.graph.train_mask],y_true)
        loss.backward()
        self.optim.step()

        # setting weights to zero using map. # TODO replace by pruning in ecnoder. No convert_map. 
        w = {}
        for name, param in self.model.named_parameters():
            if self.args.method == "GCNConv":
                if re.search(".0.lin.weight", name):
                    weights = param.cpu().clone()
                    weights = weights * self.mask[name]
                    weights = F.normalize(weights, p=2, dim=1)
                    w[name] = weights
                    self.model.state_dict()[name].data.copy_(weights)
            elif self.args.method == "GATConv":
                if re.search(".0.lin_src.weight", name):
                    weights = param.cpu().clone()
                    weights = weights * self.mask[name]
                    weights = F.normalize(weights, p=2, dim=1)
                    w[name] = weights
                    self.model.state_dict()[name].data.copy_(weights)
            else:
                print("B")
                if re.search("0.weight", name):
                    print("C")
                    weights = param.cpu().clone()
                    weights = weights * self.mask[name]
                    weights = F.normalize(weights, p=2, dim=1)
                    w[name] = weights
                    self.model.state_dict()[name].data.copy_(weights)
    
        self.weights.append(w)
        return loss.data.item()


    def val(self) -> float:
        self.model.eval()
        labels = self.graph.y[self.graph.val_mask]
        output, outputs = self.model(self.graph.x,self.graph.edge_index)
        loss = self.criterion(output[self.graph.val_mask],labels)
        return loss.data.item()

    def plot_loss(self)->None:
        if self.args.num_classes == None:
            label = "Mean Square Error"
        else:
            label = "Cross Entropy Loss"

        plt.plot(self.train_loss,color="r")
        plt.plot(self.val_loss,color="b")
        plt.yscale("log",base=10)
        plt.xscale("log")
        plt.xlabel("epoch")
        plt.ylabel(label)
        plt.legend(["Training loss", "Validaton loss"])
        plt.show()


    def weights(self,map,index):
        w = map[[str(index), str(index+1)]].drop_duplicates()
        w = w.rename(columns = {str(index): '0', str(index+1):'1'})
        w.insert(2, "values", 1)
        return w.pivot(index=['0'], columns=['1']).fillna(0)

        

    def convert_map(self,model,args):
        map = model.map_f
        method =args.method 
        n_classes =  args.num_classes 
        p = map.nunique().tolist()
        if method == "GCNConv":
            mask = dict([(''.join(["layers.", str(i), ".0.lin.weight"]), torch.tensor(np.transpose(self.weights(map, i).to_numpy()))) for i in range(len(map.columns)-1)])
        elif method == "GATConv":
            mask = dict([(''.join(["layers.", str(i), ".0.lin_src.weight"]), torch.tensor(np.transpose(self.weights(map, i).to_numpy()))) for i in range(len(map.columns)-1)])
        else:
            mask = dict([(''.join(["layers.", str(i), ".0.weight"]), torch.tensor(np.transpose(self.weights(map, i).to_numpy()))) for i in range(len(map.columns)-1)])
        
        if n_classes is not None:
            mask['layers.'+str(len(mask))+'.0.weight'] = torch.ones(n_classes, p[len(p)-1])
        else:
            ## get weight names
            weight_names = []
            for name, param in model.named_parameters():
                if method == "GCNConv":
                    if re.search(".0.lin.weight", name):
                        weight_names.append(name)
                elif method == "GATConv":
                    if re.search(".0.lin_src.weight", name):
                        weight_names.append(name)
                else:
                    if re.search("0.weight", name):
                        weight_names.append(name)
            mask0 = {}
            for i, key in enumerate(mask.keys()):
                mask0[weight_names[i]] = mask[key]
            mask = mask0
        return mask
