import torch
from vit_pytorch.simple_flash_attn_vit import SimpleViT, posemb_sincos_2d, Transformer as FlashTransformer
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange

class FlashDropoutTransformer(FlashTransformer):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, use_flash, drop_rate = 0.2):
        super().__init__(dim=dim, depth=depth, heads=heads, dim_head = dim_head, mlp_dim = mlp_dim, use_flash=use_flash)
        self.dropout = torch.nn.Dropout(drop_rate)
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x  # apply attention and add residual
            x = self.dropout(x)  # apply dropout after residual (optional location)
            x = ff(x) + x  # apply feed-forward and add residual
            x = self.dropout(x)  # apply dropout after residual (standard location)
        return x


class PerVoxelMatrixViTModel(SimpleViT):
    def __init__(self, *, output_dimensions, dim, depth, heads, mlp_dim, dim_head=64, use_flash=True, drop_rate=0.2, **kwargs):
        
        super().__init__(num_classes=1, dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, use_flash=use_flash, **kwargs)

        #class token for each voxel
        num_voxels = np.prod(output_dimensions)
        self.class_tokens = nn.Parameter(torch.randn(num_voxels, 1, dim))

        self.dropout = torch.nn.Dropout(drop_rate)

        #linear layer for each voxel
        self.voxel_heads = nn.ModuleList(
            [nn.Linear(dim, 1) for _ in range(num_voxels)]
        )

        self.transformer = FlashDropoutTransformer(dim, depth, heads, dim_head, mlp_dim, use_flash, drop_rate)

    def forward(self, img):
        *_, h, w, dtype = *img.shape, img.dtype

        x = self.to_patch_embedding(img)
        x = self.dropout(x)
        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe


        batch_size = img.shape[0]
        class_tokens = self.class_token.expand(-1, batch_size -1)

        x = torch.cat((class_tokens, x), dim=1)


        x = self.transformer(x)
        
        x = x[:, 0] # extract class token

        x = self.dropout(x)
        x = self.to_latent(x)

        outputs = [head(x[:, i]) for i, head in enumerate(self.voxel_heads)]
        outputs = torch.cat(outputs, dim=1)

        return outputs.view(batch_size, *self.output_dimensions)