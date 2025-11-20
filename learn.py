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
        y_pred, outputs = self.model(self.graph.x,self.graph.edge_index)
        loss = self.criterion(y_pred[self.graph.train_mask],y_true)
        loss.backward()
        self.optim.step()

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

        