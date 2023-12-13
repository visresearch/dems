import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import  Block, PatchEmbed, Mlp
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import time 
from random import choice
from einops import rearrange
from transform import get_patch_contrast_labels

from timm.models.layers.helpers import to_2tuple
class PatchEmbed_DW(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=32,
            patch_size=2,
            depth=2,
            in_chans=3,
            embed_dim=192,
            #norm_layer=nn.BatchNorm2d,
            flatten=True,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // (patch_size**depth), img_size // (patch_size**depth))
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        #self.batch_norm = norm_layer(in_chans)
        #self.act = nn.ReLU()

        self.proj = nn.Sequential()
        for i in range(depth):
            self.proj.add_module('proj'+str(i), nn.Conv2d(in_chans, in_chans, kernel_size=patch_size, stride=patch_size, groups=in_chans, bias=False))
            #self.proj.add_module('batch_norm'+str(i), self.batch_norm)
            #self.proj.add_module('RELU'+str(i), self.act)
            #if i != depth - 1:
            #    self.proj.add_module('point_wise'+str(i), nn.Conv2d(in_chans, in_chans, kernel_size=1, stride=1))
        self.proj.add_module('point_wise', nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=1))
                                        
    def forward(self, x):
        B, C, H, W = x.shape
        #_assert(H == self.img_size, f"Input image height ({H}) doesn't match model ({self.img_size}).")
        #_assert(W == self.img_size, f"Input image width ({W}) doesn't match model ({self.img_size}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        return x

class ViT(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, image_size=32, patch_size=16, num_classes=10, dim=768, depth=12, heads=12, 
                 channels=3, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=image_size, patch_size=patch_size, in_chans=channels, embed_dim=dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = self.build_2d_sincos_position_embedding(self.patch_embed.grid_size, True)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=dim, num_heads=heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(dim)
        self.head = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def build_2d_sincos_position_embedding(self, grid_size, w_cls, temperature=10000.):
        h, w = grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        if w_cls:
            pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
            pos_emb = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1)) 
        else:
            pos_emb = nn.Parameter(pos_emb)
        pos_emb.requires_grad = False
        return pos_emb

    def forward(self, x):
        x_init = x
        x = self.patch_embed(x_init)
        B, N, C = x.shape

        cls_tokens = self.cls_token.expand(B, 1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        x = self.pos_drop(x)
        for _, blk in enumerate(self.blocks):
            x = blk(x)

        x = self.norm(x)
        x0 = x[:, 0]

        return self.head(x0)

# def vit_tiny(patch_size=2, **kwargs):
#     model = Multi_scale_merge(
#         patch_size=patch_size, dim=192, depth=12, num_heads=3, mlp_ratio=4,
#         qkv_bias=False, **kwargs)
#     return model

# def vit_small(patch_size=2, **kwargs):
#     model = Multi_scale_merge(
#         patch_size=patch_size, dim=384, depth=12, num_heads=6, mlp_ratio=4,
#         qkv_bias=False, **kwargs)
#     return model

# def vit_base(patch_size=2, **kwargs):
#     model = Multi_scale_merge(
#         patch_size=patch_size, dim=768, depth=12, num_heads=12, mlp_ratio=4,
#         qkv_bias=False, **kwargs)
#     return model