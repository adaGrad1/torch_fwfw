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

class FwFwLayer(nn.Module):
    def __init__(self, threshold=0):
        super().__init__()
        self.threshold = nn.Parameter(torch.ones(1)*threshold)
    
    def forward(self, x):
        next_in = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        logit = x.norm(2, 1) + self.threshold
        return logit, next_in.detach()
    
class FCNet(torch.nn.Module):
    def __init__(self, dims):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        super().__init__()
        self.linears = []
        self.fwfws = []
        self.dropouts = []
        self.nonlins = []
        for d in range(len(dims) - 1):
            self.linears += [nn.Linear(dims[d], dims[d+1])]
            self.fwfws += [FwFwLayer()]
            self.dropouts += [nn.Dropout(p=0.2)]
            self.nonlins += [nn.ReLU()]
        self.linears = nn.ModuleList(self.linears)
        self.fwfws = nn.ModuleList(self.fwfws)
        self.dropouts = nn.ModuleList(self.dropouts)
        self.nonlins = nn.ModuleList(self.nonlins)

    def forward(self, x):
        logits = []
        for i in range(len(self.linears)):
            x = self.linears[i](x)
            logit, x = self.fwfws[i](x)
            x = self.dropouts[i](x)
            x = self.nonlins[i](x)
            logits += [logit]
        return logits
    
    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness = self.forward(h)
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)
    

class FCNetBP(torch.nn.Module):
    def __init__(self, dims):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        super().__init__()
        self.linears = []
        self.dropouts = []
        for d in range(len(dims) - 1):
            self.linears += [nn.Linear(dims[d], dims[d+1])]
            self.dropouts += [nn.Dropout(p=0.2)]
        self.linears = nn.ModuleList(self.linears)
        self.dropouts = nn.ModuleList(self.dropouts)

    def forward(self, x):
        for i in range(len(self.linears)):
            x = self.linears[i](x)
            x = self.dropouts[i](x)

        return x.norm(2, 1)