import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from tqdm import tqdm
import os
import time
from torch import Tensor
import collections.abc
from itertools import repeat
import torch.utils.checkpoint as checkpoint
from utils import _ntuple, iou, drop_path, DropPath, _no_grad_trunc_normal_, trunc_normal_, Mlp, PixelShuffle3d, window_partition, WindowAttention, window_reverse, SwinTransformerBlock, PatchMerging, BasicLayer, RSTB, PatchEmbed3D, PatchUnEmbed3D, Upsample, UpsampleOneStep

#========================================================================
# tFUSFormer_5ch
#========================================================================
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
    Effects of parameters:
	Embedding Dimension: Determines the size of the embedding vector for each patch. A higher embed_dim 
	increases the model's capacity and potentially its ability to capture more complex features, 
	but at the cost of higher computational requirements and memory usage.

	Depth of Each Swin Transformer Layer: Specifies the number of transformer blocks in each stage of 
	the model. Increasing the depth can improve the model's ability to learn complex representations 
	but also significantly increases the number of parameters, computation, and the risk of overfitting.

	Number of Attention Heads: Multi-head attention allows the model to focus on different parts of the 
	input simultaneously. More heads can lead to better learning of dependencies but also increase the 
	model's size and computational cost.

	Window Size: In Swin Transformers, self-attention is computed within local windows (sub-volumes). 
	A larger window size increases the receptive field per transformer block but also the computational 
	complexity of the self-attention mechanism.

	MLP Ratio: Dictates the size of the feed-forward network (FFN) within each transformer block relative 
	to the embedding dimension. A higher ratio increases the capacity of the FFN but also the model size and computation.

	Query/Key/Value Bias: Adding a learnable bias to the query, key, and value projections can help with 
	learning dynamics but has a minimal impact on computational complexity.

	Query/Key Scale Factor: Custom scaling of the dot product used in attention mechanisms. If not provided, 
	it defaults to 1/sqrt(embed_dim), which is standard. Adjusting this can impact the stability and efficiency 
	of attention computations.

	Dropout Rates: Dropout rates for the embeddings and attention weights. Using dropout can help prevent 
	overfitting by adding regularization, but setting it too high might hinder the model's ability to learn effectively.

	Drop Path Rate: Stochastic depth rate for dropping entire transformer blocks, which helps in regularization and preventing overfitting. Similar to dropout, but acts on the block level.    
    """

    def __init__(self, img_size=25, patch_size=1, in_chans=1,
                 embed_dim=120, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6], #108,(96),84,72,60,36
                 window_size=5, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, rpb=True ,patch_norm=True,
                 use_checkpoint=True, upscale=4, img_range=1., upsampler='', resi_connection='1conv',
                 output_type = "residual",num_feat=64,**kwargs):
        super(tFUSFormer_5ch, self).__init__()
        # Initialize learnable weights for 5 channels, initialized to 1/5 to mimic averaging initially
        self.channel_weights = nn.Parameter(torch.ones(6) / 6)
                
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = num_feat
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1, 1)
        
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

    def forward_features(self, x):
        if self.patch_size>1:
            x_size = self.patches_resolution
        else:
            x_size = (x.shape[2], x.shape[3], x.shape[4])
        x_feat  = self.patch_embed_features(x)

        if self.ape:
            x_feat  = x_feat + self.absolute_pos_embed

        x_feat = self.pos_drop(x_feat)

        for layer in self.layers:
            x_feat  = layer(x_feat, x_size)

        x_feat = self.norm(x_feat)
        x_feat = self.patch_unembed(x_feat, x_size)

        return x_feat

    def forward_volume(self, x):
        if self.patch_size>1:
            x_size = self.patches_resolution
        else:
            x_size = (x.shape[2], x.shape[3], x.shape[4])
        x_vol  = self.patch_embed_volume(x)

        if self.ape:
            x_vol  = x_vol + self.absolute_pos_embed

        x_vol = self.pos_drop(x_vol)

        for layer in self.layers:
            x_vol  = layer(x_vol, x_size)

        x_vol = self.norm(x_vol)
        x_vol = self.patch_unembed(x_vol, x_size)

        return x_vol


    def forward(self, x, x2, x3, x4, x5):
        H, W, D = x.shape[2:]
        x = self.check_image_size(x)
        
        self.mean = self.mean.type_as(x)
        self.mean2 = self.mean.type_as(x2)
        self.mean3 = self.mean.type_as(x3)
        self.mean4 = self.mean.type_as(x4)
        self.mean5 = self.mean.type_as(x5)
        x  = (x - self.mean)   * self.img_range
        x2 = (x2 - self.mean2) * self.img_range
        x3 = (x3 - self.mean3) * self.img_range
        x4 = (x4 - self.mean4) * self.img_range
        x5 = (x5 - self.mean5) * self.img_range

        if self.upsampler == 'pixelshuffle':
            #x  = self.conv_first(x)
            #x2 = self.conv_first(x2)
            #x3 = self.conv_first(x3)
            #x4 = self.conv_first(x4)
            #x5 = self.conv_first(x5)
            
            #x_vol = self.forward_volume(x)
            
            #x  = self.conv_after_body(self.forward_features(x))
            #x2 = self.conv_after_body(self.forward_features(x2))
            #x3 = self.conv_after_body(self.forward_features(x3))
            #x4 = self.conv_after_body(self.forward_features(x4))
            #x5 = self.conv_after_body(self.forward_features(x5))
            
            x = (1.0/6.0)*(x+self.conv_after_body(self.forward_features(self.conv_first(x))) + self.conv_after_body(self.forward_features(self.conv_first(x2))) + self.conv_after_body(self.forward_features(self.conv_first(x3))) + self.conv_after_body(self.forward_features(self.conv_first(x4))) + self.conv_after_body(self.forward_features(self.conv_first(x5))))
            print(x.size())
            x = self.conv_before_upsample(x)
            x = self.upsample(x)
            x = self.conv_last(x)
            #x = self.conv_last(self.upsample(x))

        elif self.upsampler == 'pixelshuffledirect': #2  
            #x_vol = self.forward_volume(x)     
            # Apply learned weights
            weighted_sum = (self.channel_weights[0] * x + 
                            self.channel_weights[1] * self.conv_after_body(self.forward_features(self.conv_first(x)))  + 
                            self.channel_weights[2] * self.conv_after_body(self.forward_features(self.conv_first(x2))) + 
                            self.channel_weights[3] * self.conv_after_body(self.forward_features(self.conv_first(x3))) + 
                            self.channel_weights[4] * self.conv_after_body(self.forward_features(self.conv_first(x4))) + 
                            self.channel_weights[5] * self.conv_after_body(self.forward_features(self.conv_first(x5))))                       
            #print((self.forward_volume(x)).size())
            weighted_sum = weighted_sum / self.channel_weights.sum()
            x = x + weighted_sum
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
            x2_first = self.conv_first(x2)
            x3_first = self.conv_first(x3)
            x4_first = self.conv_first(x4)
            x5_first = self.conv_first(x5)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)
            #x = self.upsample_feat(x)
            
        '''           
            if self.patch_size>1:
                x_vol = self.forward_volume(x) # not sure which one is right!!!!!!!!!!!!!!!!!!
                x_feat = self.forward_features(x_first)
                x2_feat = self.forward_features(x2_first)
                x3_feat = self.forward_features(x3_first)
                x4_feat = self.forward_features(x4_first)
                x5_feat = self.forward_features(x5_first)
                
                #res_deep_feat = self.conv_after_body(x_feat)
                #res_deep_vol = self.conv_after_body(x_vol)
                #res_deep = (res_deep_feat + res_deep_vol)/2
                #res = self.upsample_feat(res_deep)
                #res = res + x_first
            else:
                res = self.conv_after_body(self.forward_features(x_first)) + x_first
                #res = self.conv_after_body(self.forward_features(x,x_first)) + x_first
            if self.output_type == 'residual':
                x = x + self.conv_last(res)
            else:
                x = self.conv_last(self.conv_before_last(res))
        '''
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

#========================================================================
# tFUSFormer_1ch
#========================================================================
class tFUSFormer_1ch(nn.Module):
    def __init__(self, img_size=25, patch_size=1, in_chans=1,
                 embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6], #(96),84,72,60,36
                 window_size=5, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, rpb=True ,patch_norm=True,
                 use_checkpoint=True, upscale=4, img_range=1., upsampler='', resi_connection='1conv',
                 output_type = "residual",num_feat=64,**kwargs):
        super(tFUSFormer_1ch, self).__init__()
        # Initialize learnable weights for 5 channels, initialized to 1/5 to mimic averaging initially
        self.channel_weights = nn.Parameter(torch.ones(2) / 2)
                
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = num_feat
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1, 1)
        
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

    def forward_features(self, x):
        if self.patch_size>1:
            x_size = self.patches_resolution
        else:
            x_size = (x.shape[2], x.shape[3], x.shape[4])
        x_feat  = self.patch_embed_features(x)

        if self.ape:
            x_feat  = x_feat + self.absolute_pos_embed

        x_feat = self.pos_drop(x_feat)

        for layer in self.layers:
            x_feat  = layer(x_feat, x_size)

        x_feat = self.norm(x_feat)
        x_feat = self.patch_unembed(x_feat, x_size)

        return x_feat

    def forward_volume(self, x):
        if self.patch_size>1:
            x_size = self.patches_resolution
        else:
            x_size = (x.shape[2], x.shape[3], x.shape[4])
        x_vol  = self.patch_embed_volume(x)

        if self.ape:
            x_vol  = x_vol + self.absolute_pos_embed

        x_vol = self.pos_drop(x_vol)

        for layer in self.layers:
            x_vol  = layer(x_vol, x_size)

        x_vol = self.norm(x_vol)
        x_vol = self.patch_unembed(x_vol, x_size)

        return x_vol


    def forward(self, x):
        H, W, D = x.shape[2:]
        x = self.check_image_size(x)
        
        self.mean = self.mean.type_as(x)
        x  = (x - self.mean)   * self.img_range

        if self.upsampler == 'pixelshuffle':
            x = x+self.conv_after_body(self.forward_features(self.conv_first(x)))
            x = self.conv_before_upsample(x)
            x = self.upsample(x)
            x = self.conv_last(x)
            x = self.conv_last(self.upsample(x))

        elif self.upsampler == 'pixelshuffledirect': #2  
            #x_vol = self.forward_volume(x)     
            # Apply learned weights
            weighted_sum = (self.channel_weights[0] * x + 
                            self.channel_weights[1] * self.conv_after_body(self.forward_features(self.conv_first(x))))                       
            #print((self.forward_volume(x)).size())
            weighted_sum = weighted_sum / self.channel_weights.sum()
            x = x + weighted_sum
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
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)
            #x = self.upsample_feat(x)
            
        '''           
            if self.patch_size>1:
                x_vol = self.forward_volume(x) # not sure which one is right!!!!!!!!!!!!!!!!!!
                x_feat = self.forward_features(x_first)
                x2_feat = self.forward_features(x2_first)
                x3_feat = self.forward_features(x3_first)
                x4_feat = self.forward_features(x4_first)
                x5_feat = self.forward_features(x5_first)
                
                #res_deep_feat = self.conv_after_body(x_feat)
                #res_deep_vol = self.conv_after_body(x_vol)
                #res_deep = (res_deep_feat + res_deep_vol)/2
                #res = self.upsample_feat(res_deep)
                #res = res + x_first
            else:
                res = self.conv_after_body(self.forward_features(x_first)) + x_first
                #res = self.conv_after_body(self.forward_features(x,x_first)) + x_first
            if self.output_type == 'residual':
                x = x + self.conv_last(res)
            else:
                x = self.conv_last(self.conv_before_last(res))
        '''
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
        
#========================================================================
# FSRCNN
#========================================================================
class FSRCNN_1ch(nn.Module):
    def __init__(self, scale_factor=4, num_channels=1, d=56, s=12, m=4):
        super(FSRCNN_1ch, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv3d(num_channels, d, kernel_size=5, padding=5//2),
            nn.PReLU(d)
        )
        self.mid_part = [nn.Conv3d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            self.mid_part.extend([nn.Conv3d(s, s, kernel_size=3, padding=3//2), nn.PReLU(s)])
        self.mid_part.extend([nn.Conv3d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.ConvTranspose3d(d, num_channels, kernel_size=9, stride=scale_factor, padding=9//2,
                                            output_padding=scale_factor-1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.first_part:
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.mid_part:
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias.data)

    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)

        return x  
'''
#========================================================================
# SE-SRResNet_1ch
#========================================================================
class SESRResNet_1ch(nn.Module):
    class ReflectionPadding3D(nn.Module):
        def __init__(self, padding):
            super().__init__()
            #super(ReflectionPadding3D, self).__init__()
            # Assumption: padding = (d_pad, h_pad, w_pad)
            self.d_pad, self.h_pad, self.w_pad = padding

        def forward(self, x):
            # Padding depth
            x = F.pad(x, (0, 0, 0, 0, self.d_pad, self.d_pad), mode='reflect')
            # Padding height
            x = F.pad(x, (0, 0, self.h_pad, self.h_pad, 0, 0), mode='reflect')
            # Padding width
            x = F.pad(x, (self.w_pad, self.w_pad, 0, 0, 0, 0), mode='reflect')
            return x
            
    class SqueezeExcitationLayer(nn.Module):
        def __init__(self, channel, reduction_ratio):
            super().__init__()
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
            self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction_ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction_ratio, channel, bias=False),
                nn.Sigmoid()
            )

        def forward(self, x):
            b, c, _, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1, 1)
            return x * y.expand_as(x)

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size):
            super().__init__()
            padding = kernel_size // 2  # Ensure same spatial dimensions
            self.conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), bias=False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=(1, 1, 1), bias=False),
                nn.BatchNorm3d(out_channels),
            )
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), bias=False),
                nn.BatchNorm3d(out_channels)
            )
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            out = self.conv(x)
            shortcut = self.shortcut(x)
            return self.relu(out + shortcut)

    class IdentityBlock(nn.Module):
        def __init__(self, channels, kernel_size):
            super().__init__()
            padding = kernel_size // 2
            self.conv = nn.Sequential(
                nn.Conv3d(channels, channels, kernel_size=(1, 1, 1), bias=False),
                nn.BatchNorm3d(channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False),
                nn.BatchNorm3d(channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(channels, channels, kernel_size=(1, 1, 1), bias=False),
                nn.BatchNorm3d(channels)
            )
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            out = self.conv(x)
            return self.relu(out + x)

    def __init__(self):
        super(SESRResNet_1ch, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.squeeze_excite = self.SqueezeExcitationLayer(64, 4)
        self.model = nn.Sequential(
            nn.Conv3d(64, 64, 3),
            self.ReflectionPadding3D((1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            self.ConvBlock(64, 64, 3),
            self.IdentityBlock(64, 3),
            self.IdentityBlock(64, 3),
            self.IdentityBlock(64, 3),
            #PixelShuffle3d(2),
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(64, 64, 3),
            self.ReflectionPadding3D((1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            self.ConvBlock(64, 64, 3),
            self.IdentityBlock(64, 3),
            self.IdentityBlock(64, 3),
            self.IdentityBlock(64, 3),            
            # ConvBlock(32, 32, 3),
            # IdentityBlock(32, 3),
            # IdentityBlock(32, 3),
            # IdentityBlock(32, 3),
            #PixelShuffle3d(2),
            nn.Upsample(scale_factor=2, mode='trilinear'),
            #nn.Conv3d(32, 16, 3),
            nn.Conv3d(64, 64, 3),
            self.ReflectionPadding3D((1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            self.ConvBlock(64, 64, 3),
            self.IdentityBlock(64, 3),
            self.IdentityBlock(64, 3),
            self.IdentityBlock(64, 3),   
            nn.Conv3d(64, 1, 3),
            # ConvBlock(16, 16, 3),
            # IdentityBlock(16, 3),
            # IdentityBlock(16, 3),
            # IdentityBlock(16, 3),            
            # nn.Conv3d(16, 1, 3),
            self.ReflectionPadding3D((1, 1, 1)),
            nn.BatchNorm3d(1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.squeeze_excite(x)
        x = self.model(x)
        return x
'''
#========================================================================
# SE-SRResNet_1ch_reduced
#========================================================================
class SESRResNet_1ch(nn.Module):
    class ReflectionPadding3D(nn.Module):
        def __init__(self, padding):
            super().__init__()
            #super(ReflectionPadding3D, self).__init__()
            # Assumption: padding = (d_pad, h_pad, w_pad)
            self.d_pad, self.h_pad, self.w_pad = padding

        def forward(self, x):
            # Padding depth
            x = F.pad(x, (0, 0, 0, 0, self.d_pad, self.d_pad), mode='reflect')
            # Padding height
            x = F.pad(x, (0, 0, self.h_pad, self.h_pad, 0, 0), mode='reflect')
            # Padding width
            x = F.pad(x, (self.w_pad, self.w_pad, 0, 0, 0, 0), mode='reflect')
            return x
            
    class SqueezeExcitationLayer(nn.Module):
        def __init__(self, channel, reduction_ratio):
            super().__init__()
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
            self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction_ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction_ratio, channel, bias=False),
                nn.Sigmoid()
            )

        def forward(self, x):
            b, c, _, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1, 1)
            return x * y.expand_as(x)

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size):
            super().__init__()
            padding = kernel_size // 2  # Ensure same spatial dimensions
            self.conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), bias=False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=(1, 1, 1), bias=False),
                nn.BatchNorm3d(out_channels),
            )
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), bias=False),
                nn.BatchNorm3d(out_channels)
            )
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            out = self.conv(x)
            shortcut = self.shortcut(x)
            return self.relu(out + shortcut)

    class IdentityBlock(nn.Module):
        def __init__(self, channels, kernel_size):
            super().__init__()
            padding = kernel_size // 2
            self.conv = nn.Sequential(
                nn.Conv3d(channels, channels, kernel_size=(1, 1, 1), bias=False),
                nn.BatchNorm3d(channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False),
                nn.BatchNorm3d(channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(channels, channels, kernel_size=(1, 1, 1), bias=False),
                nn.BatchNorm3d(channels)
            )
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            out = self.conv(x)
            return self.relu(out + x)

    def __init__(self):
        super(SESRResNet_1ch, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.squeeze_excite = self.SqueezeExcitationLayer(32, 4)
        self.model = nn.Sequential(
            nn.Conv3d(32, 32, 3),
            self.ReflectionPadding3D((1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            self.ConvBlock(32, 32, 3),
            self.IdentityBlock(32, 3),
            self.IdentityBlock(32, 3),
            self.IdentityBlock(32, 3),
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(32, 16, 3),
            self.ReflectionPadding3D((1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            self.ConvBlock(16, 16, 3),
            self.IdentityBlock(16, 3),
            self.IdentityBlock(16, 3),
            self.IdentityBlock(16, 3),            
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(16, 16, 3),
            self.ReflectionPadding3D((1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            #self.ConvBlock(32, 32, 3),
            #self.IdentityBlock(32, 3),
            #self.IdentityBlock(32, 3),
            #self.IdentityBlock(32, 3),   
            nn.Conv3d(16, 1, 3),
            self.ReflectionPadding3D((1, 1, 1)),
            nn.BatchNorm3d(1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.squeeze_excite(x)
        x = self.model(x)
        return x

#========================================================================
# SRGAN_1ch
#========================================================================

class SRGAN_1ch(nn.Module):
    class ResidualBlock(nn.Module):
        def __init__(self, in_channels):
            super(SRGAN_1ch.ResidualBlock, self).__init__()
            self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm3d(in_channels)
            self.prelu = nn.PReLU()
            self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm3d(in_channels)

        def forward(self, x):
            residual = self.conv1(x)
            residual = self.bn1(residual)
            residual = self.prelu(residual)
            residual = self.conv2(residual)
            residual = self.bn2(residual)
            return x + residual

    class Generator(nn.Module):
        def __init__(self, num_residual_blocks=5):
            super(SRGAN_1ch.Generator, self).__init__()
            self.conv1 = nn.Conv3d(1, 32, kernel_size=9, stride=1, padding=4)
            self.prelu = nn.PReLU()

            res_blocks = [SRGAN_1ch.ResidualBlock(32) for _ in range(num_residual_blocks)]
            self.res_blocks = nn.Sequential(*res_blocks)

            self.conv2 = nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm3d(32)

            self.conv3 = nn.Conv3d(32, 128, kernel_size=3, stride=1, padding=1)           
            self.conv4 = nn.Conv3d(128, 1, kernel_size=9, stride=1, padding=4) # original

        def forward(self, x):
            x1 = self.prelu(self.conv1(x))
            x2 = self.res_blocks(x1)
            x3 = self.bn2(self.conv2(x2))
            x4 = F.interpolate(self.prelu(self.conv3(x3 + x1)), scale_factor=4, mode='trilinear')
            return self.conv4(x4)
            
    class Discriminator(nn.Module):
        def __init__(self):
            super(SRGAN_1ch.Discriminator, self).__init__()
            self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
            self.lrelu = nn.LeakyReLU(0.2)

            layers = []
            in_channels = 32
            out_channels = 64

            for _ in range(4):
                layers.extend([
                    nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm3d(out_channels),
                    nn.LeakyReLU(0.2)
                ])
                in_channels = out_channels
                out_channels *= 2

            self.layers = nn.Sequential(*layers)

            self.fc1 = nn.Linear(175616, 1024) #(562432, 1024)  # This size needs to be adjusted based on the input dimensions
            self.fc2 = nn.Linear(1024, 1)            

        def forward(self, x):
            x = self.lrelu(self.conv1(x))
            x = self.layers(x)
            x = x.view(x.size(0), -1)
            x = self.lrelu(self.fc1(x))
            return torch.sigmoid(self.fc2(x))

    def __init__(self, num_residual_blocks=5):
        super(SRGAN_1ch, self).__init__()
        self.generator = self.Generator(num_residual_blocks)
        self.discriminator = self.Discriminator()

    def forward(self, x, mode='generator'):
        if mode == 'generator':
            return self.generator(x)
        elif mode == 'discriminator':
            return self.discriminator(x)
        else:
            raise ValueError("Mode can be either 'generator' or 'discriminator'")

    def generate(self, x):
        return self.generator(x)

    def discriminate(self, x):
        return self.discriminator(x)

'''
class SRGAN_1ch(nn.Module):
    class ResidualBlock(nn.Module):
        def __init__(self, in_channels):
            super(SRGAN_1ch.ResidualBlock, self).__init__()
            self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm3d(in_channels)
            self.lrelu = nn.LeakyReLU(0.2)  # LeakyReLU
            self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm3d(in_channels)

        def forward(self, x):
            residual = self.lrelu(self.bn1(self.conv1(x)))  # Pre-activation
            residual = self.lrelu(self.bn2(self.conv2(residual)))
            return x + residual

    class Generator(nn.Module):
        def __init__(self, num_residual_blocks=3):
            super(SRGAN_1ch.Generator, self).__init__()
            self.conv1 = nn.Conv3d(1, 16, kernel_size=9, stride=1, padding=4)
            self.lrelu = nn.LeakyReLU(0.2)  # LeakyReLU

            res_blocks = [SRGAN_1ch.ResidualBlock(16) for _ in range(num_residual_blocks)]
            self.res_blocks = nn.Sequential(*res_blocks)

            self.conv2 = nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm3d(16)

            # Consider more efficient upsampling methods
            self.up = nn.ConvTranspose3d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.conv4 = nn.Conv3d(32, 1, kernel_size=9, stride=1, padding=4)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.res_blocks(x1)
            x3 = self.bn2(self.conv2(x2))
            x3 = self.lrelu(x3 + x1)  # Pre-activation
            x4 = self.up(x3) 
            return self.conv4(x4)

    class Discriminator(nn.Module):
        def __init__(self):
            super(SRGAN_1ch.Discriminator, self).__init__()
            self.conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1)
            self.lrelu = nn.LeakyReLU(0.2)

            layers = []
            in_channels = 16
            out_channels = 32

            for _ in range(3):
                layers.extend([
                    nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm3d(out_channels),
                    nn.LeakyReLU(0.2)
                ])
                in_channels = out_channels
                out_channels *= 2

            self.layers = nn.Sequential(*layers)

            self.fc1 = nn.Linear(13*13*13*4394, 1024)#(562432, 1024)  # Adjust size based on input!
            self.fc2 = nn.Linear(1024, 1)

        def forward(self, x):
            x = self.lrelu(self.conv1(x))
            x = self.layers(x)
            x = x.view(x.size(0), -1)
            x = self.lrelu(self.fc1(x))
            return torch.sigmoid(self.fc2(x))

    def __init__(self, num_residual_blocks=3):
        super(SRGAN_1ch, self).__init__()
        self.generator = self.Generator(num_residual_blocks)
        self.discriminator = self.Discriminator()

    def forward(self, x, mode='generator'):
        if mode == 'generator':
            return self.generator(x)
        elif mode == 'discriminator':
            return self.discriminator(x)
        else:
            raise ValueError("Mode can be either 'generator' or 'discriminator'")

    def generate(self, x):
        return self.generator(x)

    def discriminate(self, x):
        return self.discriminator(x)
'''     
