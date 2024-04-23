import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTConfig, ViTModel
import numpy as np

class MatrixViTModel(ViTModel):
    def __init__(self, output_dimensions, num_rois, config=None):
        if config is None:
            config = ViTConfig()
        super().__init__(config)

        self.num_rois = num_rois
        
        # Initialize learnable parameters for ROI tokens
        self.roi_tokens = nn.Parameter(torch.randn(1, num_rois, config.hidden_size))

        # Initialize linear layers for each ROI token with Dropout
        self.roi_linear_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size, output_dim),
                nn.Dropout(0.1)  # Dropout probability of 0.1
            ) for output_dim in output_dimensions
        ])

        # Attention mechanism parameters
        self.num_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // self.num_heads

        # Query, key, and value linear transformations for attention mechanism
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)

        # Output dimension for each ROI token
        self.output_dimensions = output_dimensions

    def forward(self, x, **kwargs):
        outputs = super().forward(x, **kwargs)  # Obtain outputs from the transformer encoder

        # Extract class token from the output of the transformer encoder
        class_token = outputs.last_hidden_state[:, 0, :]  # class token is first token

        # Compute attention scores
        query = self.query(self.roi_tokens).repeat(outputs.last_hidden_state.size(0), 1, 1)
        key = self.key(outputs.last_hidden_state)
        value = self.value(outputs.last_hidden_state)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.attention_head_size)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention to obtain weighted sum of output embeddings
        weighted_sum = torch.matmul(attention_weights, value)

        # Compute voxel activations for each ROI token
        roi_outputs = []
        for i in range(self.num_rois):
            # Apply linear transformation for final output
            roi_output = self.roi_linear_layers[i](weighted_sum[:, i, :])
            roi_outputs.append(roi_output)

        # Concatenate outputs of all ROI linear layers
        flattened_outputs = torch.cat(roi_outputs, dim=1)

        return flattened_outputs
