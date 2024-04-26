import torch
from vit_pytorch.simple_flash_attn_vit import SimpleViT, posemb_sincos_2d, Transformer as FlashTransformer
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn as nn
from packaging import version
from collections import namedtuple
import torch.nn.functional as F

Config = namedtuple('FlashAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])


class Attend(nn.Module):
    def __init__(self, use_flash = False, drop_rate = None):
        super().__init__()
        self.use_flash = use_flash
        self.drop_rate = drop_rate
        assert not (use_flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = Config(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not use_flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            self.cuda_config = Config(True, False, False)
        else:
            self.cuda_config = Config(False, True, True)

    def flash_attn(self, q, k, v):
        config = self.cuda_config if q.is_cuda else self.cpu_config

        # flash attention - https://arxiv.org/abs/2205.14135
        
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop_rate)

        return out

    def forward(self, q, k, v):
        n, device, scale = q.shape[-2], q.device, q.shape[-1] ** -0.5

        if self.use_flash:
            return self.flash_attn(q, k, v)

        # similarity

        sim = einsum("b h i d, b j d -> b h i j", q, k) * scale

        # attention

        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum("b h i j, b j d -> b h i d", attn, v)

        return out

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, use_flash = True, drop_rate=None):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = Attend(use_flash = use_flash, drop_rate=drop_rate)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)



class FlashDropoutTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, use_flash, drop_rate = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(drop_rate)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, use_flash = use_flash, drop_rate=drop_rate),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x  # apply attention and add residual
            x = self.dropout(x)  # apply dropout after residual (optional location)
            x = ff(x) + x  # apply feed-forward and add residual
            x = self.dropout(x)  # apply dropout after residual (standard location)
        return x


class MatrixViTModel(SimpleViT):
    def __init__(self, *, output_dimensions, dim, depth, heads, mlp_dim, dim_head=64, use_flash=True, drop_rate=0.2, **kwargs):
        
        super().__init__(num_classes=1, dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, use_flash=use_flash, **kwargs)

        self.class_token = nn.Parameter(torch.randn(1,1,dim)) #randomly initialize class token
        self.dropout = nn.Dropout(drop_rate)

        self.linear_head = nn.Sequential(
            nn.Linear(dim, np.prod(output_dimensions)), #linear projection layer
            nn.Unflatten(1, output_dimensions), #reshaping layer, not always necessary
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