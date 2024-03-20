import torch
from transformers import ViTConfig, ViTModel
import numpy as np
from util import ScaledSigmoid

class MatrixViTModel(ViTModel):
    #Bounds are exclusive
    def __init__(self, output_dimensions, bounds=None, config=None):
        if config is None:
            config = ViTConfig()
        super().__init__(config)

        
        if bounds == None:
            print("no bounds")
            self.regression_head = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, np.prod(output_dimensions)), #linear projection layer
                torch.nn.Unflatten(1, output_dimensions), #reshaping layer
                torch.nn.Sigmoid()
            )
        else:
            self.regression_head = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, np.prod(output_dimensions)), #linear projection layer
                torch.nn.Unflatten(1, output_dimensions), #reshaping layer
                ScaledSigmoid(bounds)
            )

    def forward(self, x, **kwargs):
        outputs = super().forward(x, **kwargs)
        class_token = outputs.last_hidden_state[:, 0, :]
        regression_output = self.regression_head(class_token)
        return regression_output