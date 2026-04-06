import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_mlp(in_dim=None, hid_dims=[], act=nn.ReLU(), dropout=0.,
               out_dim=None, end_with_fc=True, bias=True):

    layers = []
    if len(hid_dims) < 0:
        mlp = nn.Identity()
    elif len(hid_dims) >= 0:
        if len(hid_dims) > 0:
            for hid_dim in hid_dims:
                layers.append(nn.Linear(in_dim, hid_dim, bias=bias))
                layers.append(act)
                layers.append(nn.Dropout(dropout))
                in_dim = hid_dim
        layers.append(nn.Linear(in_dim, out_dim))
        if not end_with_fc:
            layers.append(act)
        mlp = nn.Sequential(*layers)
    return mlp


class Attn_Net(nn.Module):
    """
    Attention Network without Gating (2 fc layers)
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: dropout
        n_classes: number of classes
    """

    def __init__(self, L=1024, D=256, dropout=0., n_classes=1):
        super().__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(D, n_classes)]

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x)  # N x n_classes


class Attn_Net_Gated(nn.Module):
    """
    Attention Network with Sigmoid Gating (3 fc layers)
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: dropout
        n_classes: number of classes
    """

    def __init__(self, L=1024, D=256, dropout=0., n_classes=1):
        super().__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(dropout)]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid(),
                            nn.Dropout(dropout)]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A
