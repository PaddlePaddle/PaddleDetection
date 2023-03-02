# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This code is based on https://github.com/microsoft/FocalNet/blob/main/classification/focalnet.py
"""
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.modeling.shape_spec import ShapeSpec
from ppdet.core.workspace import register, serializable
from .transformer_utils import DropPath, Identity
from .transformer_utils import add_parameter, to_2tuple
from .transformer_utils import ones_, zeros_, trunc_normal_
from .swin_transformer import Mlp

__all__ = ['FocalNet']

MODEL_cfg = {
    'focalnet_T_224_1k_srf': dict(
        embed_dim=96,
        depths=[2, 2, 6, 2],
        focal_levels=[2, 2, 2, 2],
        focal_windows=[3, 3, 3, 3],
        drop_path_rate=0.2,
        use_conv_embed=False,
        use_postln=False,
        use_postln_in_modulation=False,
        use_layerscale=False,
        normalize_modulator=False,
        pretrained='https://bj.bcebos.com/v1/paddledet/models/pretrained/focalnet_tiny_srf_pretrained.pdparams',
    ),
    'focalnet_S_224_1k_srf': dict(
        embed_dim=96,
        depths=[2, 2, 18, 2],
        focal_levels=[2, 2, 2, 2],
        focal_windows=[3, 3, 3, 3],
        drop_path_rate=0.3,
        use_conv_embed=False,
        use_postln=False,
        use_postln_in_modulation=False,
        use_layerscale=False,
        normalize_modulator=False,
        pretrained='https://bj.bcebos.com/v1/paddledet/models/pretrained/focalnet_small_srf_pretrained.pdparams',
    ),
    'focalnet_B_224_1k_srf': dict(
        embed_dim=128,
        depths=[2, 2, 18, 2],
        focal_levels=[2, 2, 2, 2],
        focal_windows=[3, 3, 3, 3],
        drop_path_rate=0.5,
        use_conv_embed=False,
        use_postln=False,
        use_postln_in_modulation=False,
        use_layerscale=False,
        normalize_modulator=False,
        pretrained='https://bj.bcebos.com/v1/paddledet/models/pretrained/focalnet_base_srf_pretrained.pdparams',
    ),
    'focalnet_T_224_1k_lrf': dict(
        embed_dim=96,
        depths=[2, 2, 6, 2],
        focal_levels=[3, 3, 3, 3],
        focal_windows=[3, 3, 3, 3],
        drop_path_rate=0.2,
        use_conv_embed=False,
        use_postln=False,
        use_postln_in_modulation=False,
        use_layerscale=False,
        normalize_modulator=False,
        pretrained='https://bj.bcebos.com/v1/paddledet/models/pretrained/focalnet_tiny_lrf_pretrained.pdparams',
    ),
    'focalnet_S_224_1k_lrf': dict(
        embed_dim=96,
        depths=[2, 2, 18, 2],
        focal_levels=[3, 3, 3, 3],
        focal_windows=[3, 3, 3, 3],
        drop_path_rate=0.3,
        use_conv_embed=False,
        use_postln=False,
        use_postln_in_modulation=False,
        use_layerscale=False,
        normalize_modulator=False,
        pretrained='https://bj.bcebos.com/v1/paddledet/models/pretrained/focalnet_small_lrf_pretrained.pdparams',
    ),
    'focalnet_B_224_1k_lrf': dict(
        embed_dim=128,
        depths=[2, 2, 18, 2],
        focal_levels=[3, 3, 3, 3],
        focal_windows=[3, 3, 3, 3],
        drop_path_rate=0.5,
        use_conv_embed=False,
        use_postln=False,
        use_postln_in_modulation=False,
        use_layerscale=False,
        normalize_modulator=False,
        pretrained='https://bj.bcebos.com/v1/paddledet/models/pretrained/focalnet_base_lrf_pretrained.pdparams',
    ),
    'focalnet_L_384_22k_fl3': dict(
        embed_dim=192,
        depths=[2, 2, 18, 2],
        focal_levels=[3, 3, 3, 3],
        focal_windows=[5, 5, 5, 5],
        drop_path_rate=0.5,
        use_conv_embed=True,
        use_postln=True,
        use_postln_in_modulation=False,
        use_layerscale=True,
        normalize_modulator=False,
        pretrained='https://bj.bcebos.com/v1/paddledet/models/pretrained/focalnet_large_lrf_384_pretrained.pdparams',
    ),
    'focalnet_L_384_22k_fl4': dict(
        embed_dim=192,
        depths=[2, 2, 18, 2],
        focal_levels=[4, 4, 4, 4],
        focal_windows=[3, 3, 3, 3],
        drop_path_rate=0.5,
        use_conv_embed=True,
        use_postln=True,
        use_postln_in_modulation=False,
        use_layerscale=True,
        normalize_modulator=True,  #
        pretrained='https://bj.bcebos.com/v1/paddledet/models/pretrained/focalnet_large_lrf_384_fl4_pretrained.pdparams',
    ),
    'focalnet_XL_384_22k_fl3': dict(
        embed_dim=256,
        depths=[2, 2, 18, 2],
        focal_levels=[3, 3, 3, 3],
        focal_windows=[5, 5, 5, 5],
        drop_path_rate=0.5,
        use_conv_embed=True,
        use_postln=True,
        use_postln_in_modulation=False,
        use_layerscale=True,
        normalize_modulator=False,
        pretrained='https://bj.bcebos.com/v1/paddledet/models/pretrained/focalnet_xlarge_lrf_384_pretrained.pdparams',
    ),
    'focalnet_XL_384_22k_fl4': dict(
        embed_dim=256,
        depths=[2, 2, 18, 2],
        focal_levels=[4, 4, 4, 4],
        focal_windows=[3, 3, 3, 3],
        drop_path_rate=0.5,
        use_conv_embed=True,
        use_postln=True,
        use_postln_in_modulation=False,
        use_layerscale=True,
        normalize_modulator=False,
        pretrained='https://bj.bcebos.com/v1/paddledet/models/pretrained/focalnet_xlarge_lrf_384_fl4_pretrained.pdparams',
    ),
    'focalnet_H_224_22k_fl3': dict(
        embed_dim=352,
        depths=[2, 2, 18, 2],
        focal_levels=[3, 3, 3, 3],
        focal_windows=[3, 3, 3, 3],
        drop_path_rate=0.5,
        use_conv_embed=True,
        use_postln=True,
        use_postln_in_modulation=True,  #
        use_layerscale=True,
        normalize_modulator=False,
        pretrained='https://bj.bcebos.com/v1/paddledet/models/pretrained/focalnet_huge_lrf_224_pretrained.pdparams',
    ),
    'focalnet_H_224_22k_fl4': dict(
        embed_dim=352,
        depths=[2, 2, 18, 2],
        focal_levels=[4, 4, 4, 4],
        focal_windows=[3, 3, 3, 3],
        drop_path_rate=0.5,
        use_conv_embed=True,
        use_postln=True,
        use_postln_in_modulation=True,  #
        use_layerscale=True,
        normalize_modulator=False,
        pretrained='https://bj.bcebos.com/v1/paddledet/models/pretrained/focalnet_huge_lrf_224_fl4_pretrained.pdparams',
    ),
}


class FocalModulation(nn.Layer):
    """
    Args:
        dim (int): Number of input channels.
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        focal_level (int): Number of focal levels
        focal_window (int): Focal window size at focal level 1
        focal_factor (int): Step to increase the focal window. Default: 2
        use_postln_in_modulation (bool): Whether use post-modulation layernorm
        normalize_modulator (bool): Whether use normalize in modulator
    """

    def __init__(self,
                 dim,
                 proj_drop=0.,
                 focal_level=2,
                 focal_window=7,
                 focal_factor=2,
                 use_postln_in_modulation=False,
                 normalize_modulator=False):
        super().__init__()
        self.dim = dim

        # specific args for focalv3
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.focal_factor = focal_factor
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator

        self.f = nn.Linear(
            dim, 2 * dim + (self.focal_level + 1), bias_attr=True)
        self.h = nn.Conv2D(
            dim,
            dim,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias_attr=True)

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.focal_layers = nn.LayerList()

        if self.use_postln_in_modulation:
            self.ln = nn.LayerNorm(dim)

        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2D(
                        dim,
                        dim,
                        kernel_size=kernel_size,
                        stride=1,
                        groups=dim,
                        padding=kernel_size // 2,
                        bias_attr=False),
                    nn.GELU()))

    def forward(self, x):
        """ Forward function.
        Args:
            x: input features with shape of (B, H, W, C)
        """
        _, _, _, C = x.shape
        x = self.f(x)
        x = x.transpose([0, 3, 1, 2])
        q, ctx, gates = paddle.split(x, (C, C, self.focal_level + 1), 1)

        ctx_all = 0
        for l in range(self.focal_level):
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx * gates[:, l:l + 1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global * gates[:, self.focal_level:]
        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level + 1)

        x_out = q * self.h(ctx_all)
        x_out = x_out.transpose([0, 2, 3, 1])
        if self.use_postln_in_modulation:
            x_out = self.ln(x_out)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out


class FocalModulationBlock(nn.Layer):
    """ Focal Modulation Block.
    Args:
        dim (int): Number of input channels.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Layer, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Layer, optional): Normalization layer.  Default: nn.LayerNorm
        focal_level (int): number of focal levels
        focal_window (int): focal kernel size at level 1
        use_postln (bool): Whether use layernorm after modulation. Default: False.
        use_postln_in_modulation (bool): Whether use post-modulation layernorm. Default: False.
        normalize_modulator (bool): Whether use normalize in modulator
        use_layerscale (bool): Whether use layerscale proposed in CaiT. Default: False 
        layerscale_value (float): Value for layer scale. Default: 1e-4 
    """

    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 focal_level=2,
                 focal_window=9,
                 use_postln=False,
                 use_postln_in_modulation=False,
                 normalize_modulator=False,
                 use_layerscale=False,
                 layerscale_value=1e-4):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.use_postln = use_postln
        self.use_layerscale = use_layerscale

        self.norm1 = norm_layer(dim)
        self.modulation = FocalModulation(
            dim,
            proj_drop=drop,
            focal_level=self.focal_level,
            focal_window=self.focal_window,
            use_postln_in_modulation=use_postln_in_modulation,
            normalize_modulator=normalize_modulator)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)
        self.H = None
        self.W = None

        self.gamma_1 = 1.0
        self.gamma_2 = 1.0
        if self.use_layerscale:
            self.gamma_1 = add_parameter(self,
                                         layerscale_value * paddle.ones([dim]))
            self.gamma_2 = add_parameter(self,
                                         layerscale_value * paddle.ones([dim]))

    def forward(self, x):
        """
        Args:
            x: Input feature, tensor size (B, H*W, C).
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        if not self.use_postln:
            x = self.norm1(x)
        x = x.reshape([-1, H, W, C])

        # FM
        x = self.modulation(x).reshape([-1, H * W, C])
        if self.use_postln:
            x = self.norm1(x)

        # FFN
        x = shortcut + self.drop_path(self.gamma_1 * x)

        if self.use_postln:
            x = x + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x)))
        else:
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class BasicLayer(nn.Layer):
    """ A basic focal modulation layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Layer, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Layer | None, optional): Downsample layer at the end of the layer. Default: None
        focal_level (int): Number of focal levels
        focal_window (int): Focal window size at focal level 1
        use_conv_embed (bool): Whether use overlapped convolution for patch embedding
        use_layerscale (bool): Whether use layerscale proposed in CaiT. Default: False 
        layerscale_value (float): Value of layerscale
        use_postln (bool): Whether use layernorm after modulation. Default: False.
        use_postln_in_modulation (bool): Whether use post-modulation layernorm. Default: False.
        normalize_modulator (bool): Whether use normalize in modulator
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 focal_level=2,
                 focal_window=9,
                 use_conv_embed=False,
                 use_layerscale=False,
                 layerscale_value=1e-4,
                 use_postln=False,
                 use_postln_in_modulation=False,
                 normalize_modulator=False,
                 use_checkpoint=False):
        super().__init__()
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.LayerList([
            FocalModulationBlock(
                dim=dim,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i]
                if isinstance(drop_path, np.ndarray) else drop_path,
                act_layer=nn.GELU,
                norm_layer=norm_layer,
                focal_level=focal_level,
                focal_window=focal_window,
                use_postln=use_postln,
                use_postln_in_modulation=use_postln_in_modulation,
                normalize_modulator=normalize_modulator,
                use_layerscale=use_layerscale,
                layerscale_value=layerscale_value) for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                patch_size=2,
                in_chans=dim,
                embed_dim=2 * dim,
                use_conv_embed=use_conv_embed,
                norm_layer=norm_layer,
                is_stem=False)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """
        Args:
            x: Input feature, tensor size (B, H*W, C).
        """
        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x)

        if self.downsample is not None:
            x_reshaped = x.transpose([0, 2, 1]).reshape(
                [x.shape[0], x.shape[-1], H, W])
            x_down = self.downsample(x_reshaped)
            x_down = x_down.flatten(2).transpose([0, 2, 1])
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Layer):
    """ Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Layer, optional): Normalization layer. Default: None
        use_conv_embed (bool): Whether use overlapped convolution for patch embedding. Default: False
        is_stem (bool): Is the stem block or not. 
    """

    def __init__(self,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 norm_layer=None,
                 use_conv_embed=False,
                 is_stem=False):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if use_conv_embed:
            # if we choose to use conv embedding, then we treat the stem and non-stem differently
            if is_stem:
                kernel_size = 7
                padding = 2
                stride = 4
            else:
                kernel_size = 3
                padding = 1
                stride = 2
            self.proj = nn.Conv2D(
                in_chans,
                embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding)
        else:
            self.proj = nn.Conv2D(
                in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        _, _, H, W = x.shape

        if W % self.patch_size[1] != 0:
            # for 3D tensor: [pad_left, pad_right]
            # for 4D tensor: [pad_left, pad_right, pad_top, pad_bottom]
            x = F.pad(x, [0, self.patch_size[1] - W % self.patch_size[1], 0, 0])
            W += W % self.patch_size[1]
        if H % self.patch_size[0] != 0:
            x = F.pad(x, [0, 0, 0, self.patch_size[0] - H % self.patch_size[0]])
            H += H % self.patch_size[0]

        x = self.proj(x)
        if self.norm is not None:
            _, _, Wh, Ww = x.shape
            x = x.flatten(2).transpose([0, 2, 1])
            x = self.norm(x)
            x = x.transpose([0, 2, 1]).reshape([-1, self.embed_dim, Wh, Ww])

        return x


@register
@serializable
class FocalNet(nn.Layer):
    """ FocalNet backbone
    Args:
        arch (str): Architecture of FocalNet
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each FocalNet Transformer stage.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        drop_rate (float): Dropout rate.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Layer): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        focal_levels (Sequence[int]): Number of focal levels at four stages
        focal_windows (Sequence[int]): Focal window sizes at first focal level at four stages
        use_conv_embed (bool): Whether use overlapped convolution for patch embedding
        use_layerscale (bool): Whether use layerscale proposed in CaiT. Default: False 
        layerscale_value (float): Value of layerscale
        use_postln (bool): Whether use layernorm after modulation. Default: False.
        use_postln_in_modulation (bool): Whether use post-modulation layernorm. Default: False.
        normalize_modulator (bool): Whether use normalize in modulator
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
            self,
            arch='focalnet_T_224_1k_srf',
            out_indices=(0, 1, 2, 3),
            frozen_stages=-1,
            patch_size=4,
            in_chans=3,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.2,  # 0.5 better for large+ models
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            focal_levels=[2, 2, 2, 2],
            focal_windows=[3, 3, 3, 3],
            use_conv_embed=False,
            use_layerscale=False,
            layerscale_value=1e-4,
            use_postln=False,
            use_postln_in_modulation=False,
            normalize_modulator=False,
            use_checkpoint=False,
            pretrained=None):
        super(FocalNet, self).__init__()
        assert arch in MODEL_cfg.keys(), "Unsupported arch: {}".format(arch)

        embed_dim = MODEL_cfg[arch]['embed_dim']
        depths = MODEL_cfg[arch]['depths']
        drop_path_rate = MODEL_cfg[arch]['drop_path_rate']
        focal_levels = MODEL_cfg[arch]['focal_levels']
        focal_windows = MODEL_cfg[arch]['focal_windows']
        use_conv_embed = MODEL_cfg[arch]['use_conv_embed']
        use_layerscale = MODEL_cfg[arch]['use_layerscale']
        use_postln = MODEL_cfg[arch]['use_postln']
        use_postln_in_modulation = MODEL_cfg[arch]['use_postln_in_modulation']
        normalize_modulator = MODEL_cfg[arch]['normalize_modulator']
        if pretrained is None:
            pretrained = MODEL_cfg[arch]['pretrained']

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.num_layers = len(depths)
        self.patch_norm = patch_norm

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
            use_conv_embed=use_conv_embed,
            is_stem=True)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = np.linspace(0, drop_path_rate, sum(depths))

        # build layers
        self.layers = nn.LayerList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchEmbed
                if (i_layer < self.num_layers - 1) else None,
                focal_level=focal_levels[i_layer],
                focal_window=focal_windows[i_layer],
                use_conv_embed=use_conv_embed,
                use_layerscale=use_layerscale,
                layerscale_value=layerscale_value,
                use_postln=use_postln,
                use_postln_in_modulation=use_postln_in_modulation,
                normalize_modulator=normalize_modulator,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2**i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_sublayer(layer_name, layer)

        self.apply(self._init_weights)
        self._freeze_stages()
        if pretrained:
            if 'http' in pretrained:  #URL
                path = paddle.utils.download.get_weights_path_from_url(
                    pretrained)
            else:  #model in local path
                path = pretrained
            self.set_state_dict(paddle.load(path))

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.stop_gradient = True

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.stop_gradient = True

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def forward(self, x):
        x = self.patch_embed(x['image'])
        B, _, Wh, Ww = x.shape
        x = x.flatten(2).transpose([0, 2, 1])
        x = self.pos_drop(x)
        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                out = x_out.reshape([-1, H, W, self.num_features[i]]).transpose(
                    (0, 3, 1, 2))
                outs.append(out)

        return outs

    @property
    def out_shape(self):
        out_strides = [4, 8, 16, 32]
        return [
            ShapeSpec(
                channels=self.num_features[i], stride=out_strides[i])
            for i in self.out_indices
        ]
