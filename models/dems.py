import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple, trunc_normal_
from timm.models.layers.helpers import to_2tuple
from timm.models.vision_transformer import  Block, PatchEmbed
import numpy as np
from einops import rearrange
from transforms import get_patch_contrast_labels

class PatchEmbed_DW(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=32,
            init_patch_size=2,
            s = 4,
            in_chans=3,
            embed_dim=192,
            # norm_layer=nn.BatchNorm2d,
            flatten=True,
    ):
        super().__init__()

        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(init_patch_size*s)
        self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[0] // self.patch_size[0])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        # self.batch_norm = norm_layer(in_chans)
        # self.act = nn.ReLU(inplace=True)

        self.proj = nn.Sequential()
        self.proj.add_module('proj-1', nn.Conv2d(in_chans, in_chans, kernel_size=self.patch_size[0] // 4, stride=self.patch_size[0] // 4, bias=False, groups=in_chans))
        self.proj.add_module('proj-2', nn.Conv2d(in_chans, in_chans, kernel_size=4, stride=4, bias=False, groups=in_chans))
        self.proj.add_module('point_wise', nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=1))
                                        
    def forward(self, x):
        B, C, H, W = x.shape
        #_assert(H == self.img_size, f"Input image height ({H}) doesn't match model ({self.img_size}).")
        #_assert(W == self.img_size, f"Input image width ({W}) doesn't match model ({self.img_size}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        return x

class Interaction(nn.Module):
    def __init__(self, in_feature, out_feature, norm, act_layer=nn.GELU, bias=True):
        super().__init__()
        
        hidden_feature = in_feature * 2
        hidden_feature2 = in_feature // 2
        bias = to_2tuple(bias)

        self.norm = norm
        self.fc1 = nn.Linear(in_feature, hidden_feature, bias=bias[0])
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_feature, hidden_feature2, bias=bias[1])
        self.fc3 = nn.Linear(hidden_feature2, out_feature, bias=bias[1])
        self.scale = nn.Parameter(torch.ones(1, 1, 1, 1))

    def forward(self, x):
        x_origin = x
        x_origin = rearrange(x_origin, 'b n1 n2 c->b n1 c n2')

        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)

        x = F.softmax(x * self.scale, dim=-2)

        x_merge = torch.einsum('bnik,bnkj->bnij',x_origin, x)
        x_merge = rearrange(x_merge, 'b n1 c n2->b n1 n2 c')
        return x_merge

class DEMS_ViT(nn.Module):
    def __init__(self, image_size=32, patch_size=2, num_classes=100, dim=768, depth=12,
                 heads=12, merge_num = [64, 32], merge_layer = [6, 10], channels=3, mlp_ratio=[4., 4., 4.], 
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=image_size, patch_size=patch_size, in_chans=channels, embed_dim=dim)
        self.patch_embed_2 = PatchEmbed(
            img_size=image_size, patch_size=patch_size*2, in_chans=channels, embed_dim=dim)
        self.patch_embed_4 = PatchEmbed_DW(
           img_size=image_size, init_patch_size=patch_size, s=4, in_chans=channels, embed_dim=dim)
        self.patch_embed_8 = PatchEmbed_DW(
            img_size=image_size, init_patch_size=patch_size, s=8, in_chans=channels, embed_dim=dim)


        self.num_region = self.patch_embed_8.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        
        self.pos_embed = self.build_2d_sincos_position_embedding(self.patch_embed.grid_size, False)#.requires_grad(False)
        self.pos_embed_2 = self.build_2d_sincos_position_embedding(self.patch_embed_2.grid_size, False)
        self.pos_embed_4 = self.build_2d_sincos_position_embedding(self.patch_embed_4.grid_size, False)
        self.pos_embed_8 = self.build_2d_sincos_position_embedding(self.patch_embed_8.grid_size, False)
        
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([])
        for i in range(depth):
            if i < merge_layer[0] - 1:
                m_ratio = mlp_ratio[0]
            elif i < merge_layer[1] - 1:
                m_ratio = mlp_ratio[1]
            else:
                m_ratio = mlp_ratio[2]
            self.blocks.append(
                Block(
                dim=dim, num_heads=heads, mlp_ratio=m_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer
                )
            )
        self.norm = norm_layer(dim)
        self.norm_merge1 = norm_layer(dim)
        self.norm_merge2 = norm_layer(dim)

        self.head = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()

        self.merge_num1 = merge_num[0]
        self.merge_num2 = merge_num[1]
        self.merge_layer1 = merge_layer[0] - 1
        self.merge_layer2 = merge_layer[1] - 1

        self.mlp1 = Interaction(in_feature=dim, out_feature=self.merge_num1, norm=self.norm_merge1)
        self.mlp2 = Interaction(in_feature=dim, out_feature=self.merge_num2, norm=self.norm_merge2)

        trunc_normal_(self.cls_token, std=.02)
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
        return {'pos_embed', 'pos_embed_2', 'pos_embed_4', 'pos_embed_8', 'cls_token'}

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

    def patch_convert(self, x, patch_label):
        B, _, C = x.shape
        patch_label = patch_label.reshape(B, -1)
        patch_label = patch_label.repeat(C, 1, 1)
        patch_label= patch_label.permute(1, 0, 2)
        x = rearrange(x, 'b n c->b c n')
        x = x.gather(dim=-1, index=patch_label)
        x = rearrange(x, 'b c n->b n c')
        x = x.reshape(B, self.num_region, -1, C)
        return x

    def patch_merge(self, x, mlp):
        B, _, _, C = x.shape
        x = mlp(x)
        x = x.reshape(B, -1, C)
        return x
    
    def forward(self, x):
        x_init = x
        x = self.patch_embed(x_init)
        B, N, C = x.shape
        x_2 = self.patch_embed_2(x_init)
        x_4 = self.patch_embed_4(x_init)
        x_8 = self.patch_embed_8(x_init)
        cls_tokens = self.cls_token.expand(B, 1, -1)
        
        x = x + self.pos_embed
        x_2 = x_2 + self.pos_embed_2
        x_4 = x_4 + self.pos_embed_4
        x_8 = x_8 + self.pos_embed_8

        patch_label1 = get_patch_contrast_labels(img_size=self.patch_embed.img_size[0], patch_size1=self.patch_embed.patch_size[0], 
                                                    patch_size2=self.patch_embed_8.patch_size[0], stride2=self.patch_embed_8.patch_size[0]).repeat(B, 1, 1).to(x.device)
        patch_label2 = get_patch_contrast_labels(img_size=self.patch_embed.img_size[0], patch_size1=self.patch_embed_2.patch_size[0], 
                                                    patch_size2=self.patch_embed_8.patch_size[0], stride2=self.patch_embed_8.patch_size[0]).repeat(B, 1, 1).to(x.device)
        patch_label3 = get_patch_contrast_labels(img_size=self.patch_embed.img_size[0], patch_size1=self.patch_embed_4.patch_size[0], 
                                                    patch_size2=self.patch_embed_8.patch_size[0], stride2=self.patch_embed_8.patch_size[0]).repeat(B, 1, 1).to(x.device)
        
        x = self.patch_convert(x, patch_label1)
        x_2 = self.patch_convert(x_2, patch_label2)
        x_4 = self.patch_convert(x_4, patch_label3)
        x_8 = x_8.unsqueeze(2)

        x = torch.cat((x,x_2,x_4,x_8),dim=2).reshape(B, -1, C)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)

        for layer, blk in enumerate(self.blocks):
            if layer == self.merge_layer1:
                x_cls = x[:, 0:1, :]
                x_patch = x[:, 1:, :].reshape(B, self.num_region, -1, C)
                x_patch = self.patch_merge(x_patch, mlp=self.mlp1)
                x = torch.cat((x_cls, x_patch), dim=1)
            if layer == self.merge_layer2:
                x_cls = x[:, 0:1, :]
                x_patch = x[:, 1:, :].reshape(B, self.num_region, -1, C)
                x_patch = self.patch_merge(x_patch, mlp=self.mlp2)
                x = torch.cat((x_cls, x_patch), dim=1)
            x = blk(x)

        x = self.norm(x)
        x0 = x[:, 0]

        return self.head(x0)

@register_model
def dems_tiny_patch2_32(patch_size=2, **kwargs):
    model = DEMS_ViT(
        image_size=32, patch_size=2, dim=192, depth=12, num_heads=3, mlp_ratio=[4, 4, 4],
        qkv_bias=False, **kwargs)
    return model

@register_model
def dems_small_patch2_32(patch_size=2, **kwargs):
    model = DEMS_ViT(
        image_size=32, patch_size=2, dim=384, depth=12, num_heads=6, mlp_ratio=[4, 4, 4],
        qkv_bias=False, **kwargs)
    return model

