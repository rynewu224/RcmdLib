import torch
import torch.nn as nn
import torch.nn.functional as F


class RcmdModel(nn.Module):
    def __init__(self, num_user, num_item, dim):
        self.user_embs = nn.Embedding(num_user, dim)
        self.item_embs = nn.Embedding(num_item, dim)

    def forward(self, x):

        pass

    def train(self, loader):
        pass

    def test(self, loader):
        pass