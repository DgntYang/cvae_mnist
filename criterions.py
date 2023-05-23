import torch
import torch.nn as nn
import torch.nn.functional as F

class KL_loss(nn.Module):
    def __init__(self):
        super(KL_loss, self).__init__()

    def forward(self, x_mu, y_mu, log_var_2):
        return 0.5 * torch.sum((x_mu - y_mu) ** 2 + torch.exp(log_var_2) - log_var_2 - 1, dim=-1)

class BCEloss(nn.Module):
    def __init__(self):
        super(BCEloss, self).__init__()

    def forward(self, x, target):
        assert x.shape == target.shape, 'Inconsitent dimension!'
        return F.binary_cross_entropy(x, target, reduction='sum')

class MSEloss(nn.Module):
    def __init__(self):
        super(MSEloss, self).__init__()

    def forward(self, x, target):
        assert x.shape == target.shape, 'Inconsitent dimension!'
        return F.mse_loss(x, target, reduction='sum')

