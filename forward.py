import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
import torch.nn.functional as F
from util import *

class Apply_One_Hot_Label:
    def __init__(self, n_classes=10):
        self.n_classes = n_classes
    
    def __call__(self, x, label):
        y = torch.ones(size=(len(x),)).type(torch.int64) * label
        y = F.one_hot(y, num_classes=self.n_classes)
        y = y.to(x.device)
        return (x, y)

class FwFwLayer(nn.Module):
    def __init__(self, threshold=0):
        super().__init__()
        self.threshold = nn.Parameter(torch.ones(1)*threshold)
    
    def forward(self, x):
        next_in = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        logit = x.norm(2, 1) + self.threshold
        return logit, next_in.detach()

class FwFwNet(torch.nn.Module):
    
    def __init__(self, apply_label = None, **kwargs):
        if apply_label == None:
            apply_label = Apply_One_Hot_Label(self.y_classes)
        self.apply_label = apply_label
        super().__init__()
    
    def predict(self, x, apply_label=None):
        if apply_label == None:
            apply_label = self.apply_label
        goodness_per_label = []
        for label in range(self.y_classes):
            cur_x = apply_label(x, label)
            goodness = self.forward(cur_x)
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

class FCNet(FwFwNet):
    def __init__(self, dims, y_classes=10, dropout=0.2, **kwargs):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.y_classes = y_classes
        super().__init__(**kwargs)
        self.linears = []
        self.fwfws = []
        self.dropouts = []
        self.nonlins = []
        for d in range(len(dims) - 1):
            if d == 0:
                self.linears += [nn.Linear(dims[d]+y_classes, dims[d+1])]
            else:
                self.linears += [nn.Linear(dims[d], dims[d+1])]
            self.nonlins += [nn.ReLU()]
            self.fwfws += [FwFwLayer()]
            self.dropouts += [nn.Dropout(p=dropout)]
        self.linears = nn.ModuleList(self.linears)
        self.fwfws = nn.ModuleList(self.fwfws)
        self.dropouts = nn.ModuleList(self.dropouts)
        self.nonlins = nn.ModuleList(self.nonlins)

    def forward(self, x):
        x = torch.cat((x[0], x[1]), -1) # append x and label
        logits = []
        for i in range(len(self.linears)):
            x = self.linears[i](x)
            x = self.nonlins[i](x)
            logit, x = self.fwfws[i](x)
            x = self.dropouts[i](x)
            logits += [logit]
        return logits
    
