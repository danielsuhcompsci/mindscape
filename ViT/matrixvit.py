import torch
from vit_pytorch.simple_flash_attn_vit import SimpleViT
import numpy as np

class MatrixViTModel(SimpleViT):
    def __init__(self, *, output_dimensions, dim, **kwargs):
        
        super().__init__(num_classes=1, dim=dim, **kwargs)

        self.linear_head = torch.nn.Sequential(
            torch.nn.Linear(dim, np.prod(output_dimensions)), #linear projection layer
            torch.nn.Unflatten(1, output_dimensions), #reshaping layer, not always necessary
        )