import torch
from transformers import ViTConfig, ViTModel
import numpy as np

class MatrixViTModel(ViTModel):
    def __init__(self, output_dimensions, config=None):
        if config is None:
            config = ViTConfig()
        super().__init__(config)

        
        self.regression_head = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, np.prod(output_dimensions)), #linear projection layer
            torch.nn.Unflatten(1, output_dimensions), #reshaping layer, not always necessary
        )

    
    def forward(self, x, **kwargs):
        outputs = super().forward(x, **kwargs)
        
        class_token = outputs.last_hidden_state[:, 0, :]  # class token is first token
        regression_output = self.regression_head(class_token)
        
        # include attentions if requested
        if 'return_dict' in kwargs and kwargs['return_dict'] == True:
            res = {
                "regression_output": regression_output
            }

            if 'output_attentions' in kwargs and kwargs['output_attentions'] == True:
                res['attentions'] = outputs.attentions
            return res
        else:
            return regression_output