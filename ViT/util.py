import torch
from torch import nn

class ScaledSigmoid(nn.Module):
    def __init__(self, bounds):
        super().__init__()

        self.bounds = bounds

    def forward(self, x):
        lower_bound = self.bounds[0]
        upper_bound = self.bounds[1]
        bounds_range = upper_bound - lower_bound
        return lower_bound + (bounds_range * torch.sigmoid(x))
