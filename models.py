import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import time
from torch import Tensor
import collections.abc
from itertools import repeat
import torch.utils.checkpoint as checkpoint
from utils import _ntuple, iou, drop_path, DropPath, _no_grad_trunc_normal_, trunc_normal_, Mlp, PixelShuffle3d, window_partition, WindowAttention, window_reverse, SwinTransformerBlock, PatchMerging, BasicLayer, RSTB, PatchEmbed3D, PatchUnEmbed3D, Upsample, UpsampleOneStep

class tFUSFormer_5ch(nn.Module):
    r""" SwinIR
        A PyTorch impl of : `SwinIR: Image Restoration Using Swin Transformer`, based on Swin Transformer.
        Implementación promediando las representaciones a la entrada del transformer.
    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self, img_size=25, patch_size=1, in_chans=1,
                 embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=5, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, rpb=True ,patch_norm=True,
                 use_checkpoint=True, upscale=4, img_range=1., upsampler='', resi_connection='1conv',
                 output_type = "residual",num_feat=64,**kwargs):
        super(tFUSFormer_5ch, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = num_feat
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1, 1)
            #self.mean = torch.zeros(1, num_in_ch, 1, 1, 1) # not sure
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size
        self.patch_size = patch_size

        #####################################################################################################
        ################################### 3D shallow feature extraction ###################################
        self.conv_first = nn.Conv3d(num_in_ch, embed_dim, 3, 1, 1)
        #####################################################################################################
        ################################### 3D Deep Feature Extraction ######################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.rpb = rpb
        self.output_type = output_type
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio


        self.patch_embed_features = PatchEmbed3D(
            img_size = img_size, patch_size = patch_size, in_chans = embed_dim, embed_dim = embed_dim,
            norm_layer = norm_layer if self.patch_norm else None, level = "first")

        self.patch_embed_volume = PatchEmbed3D(
            img_size = img_size, patch_size = patch_size, in_chans = in_chans, embed_dim = embed_dim,
            norm_layer = norm_layer if self.patch_norm else None, level = "first")

        num_patches = self.patch_embed_volume.num_patches
        patches_resolution = self.patch_embed_volume.patches_resolution
        self.patches_resolution = patches_resolution

        self.patch_unembed = PatchUnEmbed3D(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)


        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1],
                                           patches_resolution[2]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=1,
                         resi_connection=resi_connection,
                         rpb = self.rpb
                         )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv3d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':

            self.conv_after_body = nn.Sequential(nn.Conv3d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv3d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv3d(embed_dim // 4, embed_dim, 3, 1, 1))

        ########################################
        # 3D high quality image reconstruction #
        ########################################
        if self.upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(nn.Conv3d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            #self.upsample = nn.Upsample(scale_factor=2, mode='nearest')#nn.Upsample(upscale, num_feat)
            self.conv_last = nn.Conv3d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(nn.Conv3d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.conv_up1  = nn.Conv3d(num_feat, num_feat,   3, 1, 1)
            self.conv_up2  = nn.Conv3d(num_feat, num_feat,   3, 1, 1)
            self.conv_hr   = nn.Conv3d(num_feat, num_feat,   3, 1, 1)
            self.conv_last = nn.Conv3d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu     = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            if self.patch_size>1:
                self.upsample_feat = nn.Sequential(nn.ConvTranspose3d(embed_dim,embed_dim, kernel_size=self.patch_size, stride=self.patch_size),
                                                  nn.Conv3d(embed_dim, embed_dim, 3, 1, 1),
                                                  nn.InstanceNorm3d(embed_dim),
                                                  nn.LeakyReLU(inplace=True))
            if self.output_type=='residual':
                self.conv_last = nn.Conv3d(embed_dim, num_out_ch, 3, 1, 1)
            else:
                self.conv_before_last = nn.Sequential(nn.Conv3d(embed_dim, num_feat,3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
                self.conv_last= nn.Conv3d(num_feat, num_out_ch, 1, 1, 0)


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
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        _, _, h, w, d = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        mod_pad_d = (self.window_size - d % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h, 0, mod_pad_d), 'replicate')

        return x

#     def forward_features(self, x, x_feat):
#         print('x.shape in forward_features (weird)',x.shape) # 4 1 50 50 50
#         print('x_feat.shape in forward_features (weird)',x_feat.shape) # 4 96 50 50 50
#         if self.patch_size>1:
#             x_size = self.patches_resolution
#             print('x_size if self.patch_size>1', x_size)
#         else:
#             x_size = (x.shape[2], x.shape[3], x.shape[4])
#             print('x_size if not self.patch_size>1', x_size)
#         print('x_size in forward_features', x_size)
#         x_feat = self.patch_embed_features(x_feat)
#         x_vol= self.patch_embed_volume(x)
#         print('x_feat in forward_features!!!!!!!!!!!', x_feat)
#         print('x_vol in forward_features!!!!!!!!!!', x_vol)

#         if self.ape:
#             x_feat = x_feat + self.absolute_pos_embed
#             x_vol = x_vol + self.absolute_pos_embed
#         x_feat = self.pos_drop(x_feat)
#         x_vol = self.pos_drop(x_vol)

#         for layer in self.layers:
#             x_feat = layer(x_feat, x_size)
#             x_vol = layer(x_vol, x_size)

#         x_feat = self.norm(x_feat)
#         x_vol = self.norm(x_vol)
#         x_feat = self.patch_unembed(x_feat, x_size)
#         x_vol = self.patch_unembed(x_vol, x_size)
#         return x_feat, x_vol


    def forward_features(self, x):
        if self.patch_size>1:
            x_size = self.patches_resolution
        else:
            x_size = (x.shape[2], x.shape[3], x.shape[4])
        x_feat  = self.patch_embed_features(x)

        #x_vol= self.patch_embed_volume(x)

        if self.ape:
            x_feat  = x_feat + self.absolute_pos_embed
            #x_vol = x_vol + self.absolute_pos_embed
        x_feat = self.pos_drop(x_feat)
        #x_vol = self.pos_drop(x_vol)

        for layer in self.layers:
            x_feat  = layer(x_feat, x_size)
            #x_vol = layer(x_vol, x_size)

        x_feat = self.norm(x_feat)

        #x_vol  = self.norm(x_vol)
        x_feat = self.patch_unembed(x_feat, x_size)

        #x_vol  = self.patch_unembed(x_vol, x_size)
        return x_feat #, x_vol



    def forward(self, x, x2, x3, x4, x5):
        H, W, D = x.shape[2:]
        x = self.check_image_size(x)
        self.mean = self.mean.type_as(x)
        self.mean2 = self.mean.type_as(x2)
        self.mean3 = self.mean.type_as(x3)
        self.mean4 = self.mean.type_as(x4)
        self.mean5 = self.mean.type_as(x5)
        x = (x - self.mean) * self.img_range
        x2 = (x2 - self.mean2) * self.img_range
        x3 = (x3 - self.mean3) * self.img_range
        x4 = (x4 - self.mean4) * self.img_range
        x5 = (x5 - self.mean5) * self.img_range

        if self.upsampler == 'pixelshuffle': # for now only pixelshuffle works!!!
            x = self.conv_first(x)
            x2 = self.conv_first(x2)
            x3 = self.conv_first(x3)
            x4 = self.conv_first(x4)
            x5 = self.conv_first(x5)
            #x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_after_body(self.forward_features(x)) + self.conv_after_body(self.forward_features(x2))
            + self.conv_after_body(self.forward_features(x3)) + self.conv_after_body(self.forward_features(x4))
            self.conv_after_body(self.forward_features(x5)) + + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))

        elif self.upsampler == 'pixelshuffledirect':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            x_first = self.conv_first(x)
            if self.patch_size>1:
                x_feat, x_vol = self.forward_features(x) # not sure which one is right!!!!!!!!!!!!!!!!!!
                #x_feat, x_vol = self.forward_features(x_first)
                res_deep_feat = self.conv_after_body(x_feat)
                res_deep_vol = self.conv_after_body(x_vol)
                res_deep = (res_deep_feat + res_deep_vol)/2
                res = self.upsample_feat(res_deep)
                res = res + x_first
            else:
                res = self.conv_after_body(self.forward_features(x_first)) + x_first
                #res = self.conv_after_body(self.forward_features(x,x_first)) + x_first
            if self.output_type == 'residual':
                x = x + self.conv_last(res)
            else:
                x = self.conv_last(self.conv_before_last(res))

        x = x / self.img_range + self.mean

        return x[:, :, :H*self.upscale, :W*self.upscale, :D*self.upscale]


    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops
