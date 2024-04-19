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
            x = self.dropout(attn(x)) + x  # apply dropout after attention and add residual
            x = self.dropout(ff(x)) + x    # apply dropout after feedforward and add residual
        return x


class MatrixViTModel(SimpleViT):
    def __init__(self, *, output_dimensions, dim, depth, heads, mlp_dim, dim_head=64, use_flash=True, drop_rate=0.2, **kwargs):
        
        super().__init__(num_classes=1, dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, use_flash=use_flash, **kwargs)

        self.class_token = torch.nn.Parameter(torch.randn(1,1,dim)) #randomly initialize class token
        self.dropout = torch.nn.Dropout(drop_rate)

        self.linear_head = torch.nn.Sequential(
            torch.nn.Linear(dim, np.prod(output_dimensions)), #linear projection layer
            torch.nn.Unflatten(1, output_dimensions), #reshaping layer, not always necessary
        )

        self.transformer = FlashDropoutTransformer(dim, depth, heads, dim_head, mlp_dim, use_flash, drop_rate)

    def forward(self, img):
        *_, h, w, dtype = *img.shape, img.dtype

        x = self.to_patch_embedding(img)
        x = self.dropout(x)
        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe


        batch_size = img.shape[0]
        class_tokens = self.class_token.expand(batch_size, -1, -1)

        x = torch.cat((class_tokens, x), dim=1)


        x = self.transformer(x)
        
        x = x[:, 0] # extract class token

        x = self.dropout(x)
        x = self.to_latent(x)
        return self.linear_head(x)