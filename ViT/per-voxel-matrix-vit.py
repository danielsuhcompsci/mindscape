from matrixvit import MatrixViTModel
import torch
import numpy as np
from vit_pytorch.simple_flash_attn_vit import posemb_sincos_2d

class PerVoxelMatrixViTModel(MatrixViTModel):
    def __init__(self, *, output_dimensions, dim, depth, heads, mlp_dim, dim_head=64, use_flash=True, drop_rate=0.2, **kwargs):
        super.__init__(output_dimensions=output_dimensions, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dim_head=dim_head, use_flash=use_flash, drop_rate=drop_rate, **kwargs)
        
        self.class_token = None
        self.linear_head = None

        #class token for each voxel
        num_voxels = np.prod(output_dimensions)
        self.class_tokens = nn.Parameter(torch.randn(num_voxels, 1, dim))

        self.dropout = torch.nn.Dropout(drop_rate)

        #linear layer for each voxel
        self.voxel_heads = nn.ModuleList(
            [nn.Linear(dim, 1) for _ in range(num_voxels)]
        )

    def forward(x):
        *_, h, w, dtype = *img.shape, img.dtype

        x = self.to_patch_embedding(img)
        x = self.dropout(x)
        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe


        batch_size = img.shape[0]
        class_tokens = self.class_token.expand(-1, batch_size -1)

        x = torch.cat((class_tokens, x), dim=1)

        x = self.transformer(x)
        
        x = x[:, 0] # extract class tokens

        x = self.dropout(x)
        x = self.to_latent(x)

        outputs = [head(x[:, i]) for i, head in enumerate(self.voxel_heads)]
        outputs = torch.cat(outputs, dim=1)

        return outputs.view(batch_size, *self.output_dimensions)