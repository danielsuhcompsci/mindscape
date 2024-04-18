import torch
from vit_pytorch.simple_flash_attn_vit import SimpleViT
import numpy as np

class MatrixViTModel(SimpleViT):
    def __init__(self, *, output_dimensions, dim, **kwargs):
        
        super().__init__(num_classes=1, dim=dim, **kwargs)

        self.class_token = torch.nn.Parameter(torch.randn(1,1,dim)) #randomly initialize class token

        self.linear_head = torch.nn.Sequential(
            torch.nn.Linear(dim, np.prod(output_dimensions)), #linear projection layer
            torch.nn.Unflatten(1, output_dimensions), #reshaping layer, not always necessary
        )

    def forward(self, img):
        *_, h, w, dtype = *img.shape, img.dtype

        x = self.to_patch_embedding(img)

        batch_size = img.shape[0]
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat((class_tokens, x), dim=1)

        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.transformer(x)
        
        x = x[:, 0] # extract class token

        x = self.to_latent(x)
        return self.linear_head(x)