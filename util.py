import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F



def MNIST_loaders(train_batch_size=50000, test_batch_size=10000):

    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])

    train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(
        MNIST('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader


def overlay_y_on_x(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_
    
def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize = (4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()

def fwfw_loss(true_labels : torch.Tensor, model_outs : list):
        to_return = None
        for probs in model_outs:
            if to_return == None:
                to_return = F.binary_cross_entropy_with_logits(probs, true_labels)
            else:
                to_return += F.binary_cross_entropy_with_logits(probs, true_labels)
        return to_return

class FwFw_Dataset(Dataset):
    
    def __init__(self, pos, neg):
        self.pos = pos
        self.neg = neg
    def __len__(self):
        return len(self.pos) + len(self.neg)
    def __getitem__(self, idx):
        if idx >= len(self.pos):
            return self.neg[idx-len(self.pos)], 0.00
        else:
            return self.pos[idx], 1.0