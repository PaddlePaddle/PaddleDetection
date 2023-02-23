# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from paddle.nn.initializer import Constant, TruncatedNormal

from ppdet.modeling.shape_spec import ShapeSpec
from ppdet.core.workspace import register, serializable

from .transformer_utils import (DropPath, Identity, window_partition,
                                window_unpartition)
from ..initializer import linear_init_


class Mlp(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer='nn.GELU',
                 drop=0.,
                 lr_factor=1.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(
            in_features,
            hidden_features,
            weight_attr=ParamAttr(learning_rate=lr_factor),
            bias_attr=ParamAttr(learning_rate=lr_factor))
        self.act = eval(act_layer)()
        self.fc2 = nn.Linear(
            hidden_features,
            out_features,
            weight_attr=ParamAttr(learning_rate=lr_factor),
            bias_attr=ParamAttr(learning_rate=lr_factor))
        self.drop = nn.Dropout(drop)

        self._init_weights()

    def _init_weights(self):
        linear_init_(self.fc1)
        linear_init_(self.fc2)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class Attention(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=True,
                 attn_drop=0.,
                 proj_drop=0.,
                 use_rel_pos=False,
                 rel_pos_zero_init=True,
                 window_size=None,
                 input_size=None,
                 qk_scale=None,
                 lr_factor=1.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.use_rel_pos = use_rel_pos
        self.input_size = input_size
        self.rel_pos_zero_init = rel_pos_zero_init
        self.window_size = window_size
        self.lr_factor = lr_factor

        self.qkv = nn.Linear(
            dim,
            dim * 3,
            weight_attr=ParamAttr(learning_rate=lr_factor),
            bias_attr=ParamAttr(learning_rate=lr_factor) if qkv_bias else False)
        self.proj = nn.Linear(
            dim,
            dim,
            weight_attr=ParamAttr(learning_rate=lr_factor),
            bias_attr=ParamAttr(learning_rate=lr_factor))
        self.attn_drop = nn.Dropout(attn_drop)
        if window_size is None:
            self.window_size = self.input_size[0]

        self._init_weights()

    def _init_weights(self):
        linear_init_(self.qkv)
        linear_init_(self.proj)

        if self.use_rel_pos:
            self.rel_pos_h = self.create_parameter(
                [2 * self.window_size - 1, self.head_dim],
                attr=ParamAttr(learning_rate=self.lr_factor),
                default_initializer=Constant(value=0.))
            self.rel_pos_w = self.create_parameter(
                [2 * self.window_size - 1, self.head_dim],
                attr=ParamAttr(learning_rate=self.lr_factor),
                default_initializer=Constant(value=0.))

            if not self.rel_pos_zero_init:
                TruncatedNormal(self.rel_pos_h, std=0.02)
                TruncatedNormal(self.rel_pos_w, std=0.02)

    def get_rel_pos(self, seq_size, rel_pos):
        max_rel_dist = int(2 * seq_size - 1)
        # Interpolate rel pos if needed.
        if rel_pos.shape[0] != max_rel_dist:
            # Interpolate rel pos.
            rel_pos = rel_pos.reshape([1, rel_pos.shape[0], -1])
            rel_pos = rel_pos.transpose([0, 2, 1])
            rel_pos_resized = F.interpolate(
                rel_pos,
                size=(max_rel_dist, ),
                mode="linear",
                data_format='NCW'
            )
            rel_pos_resized = rel_pos_resized.reshape([-1, max_rel_dist])
            rel_pos_resized = rel_pos_resized.transpose([1, 0])
        else:
            rel_pos_resized = rel_pos

        coords = paddle.arange(seq_size, dtype='float32')
        relative_coords = coords.unsqueeze(-1) - coords.unsqueeze(0)
        relative_coords += (seq_size - 1)
        relative_coords = relative_coords.astype('int64').flatten()

        return paddle.index_select(rel_pos_resized, relative_coords).reshape(
            [seq_size, seq_size, self.head_dim])
      
    def add_decomposed_rel_pos(self, attn, q, h, w):
        """
        Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
        Args:
            attn (Tensor): attention map.
            q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        Returns:
            attn (Tensor): attention map with added relative positional embeddings.
        """
        Rh = self.get_rel_pos(h, self.rel_pos_h)
        Rw = self.get_rel_pos(w, self.rel_pos_w)

        B, _, dim = q.shape
        r_q = q.reshape([B, h, w, dim])
        # bhwc, hch->bhwh1
        # bwhc, wcw->bhw1w
        rel_h = paddle.einsum("bhwc,hkc->bhwk", r_q, Rh).unsqueeze(-1)
        rel_w = paddle.einsum("bhwc,wkc->bhwk", r_q, Rw).unsqueeze(-2)

        attn = attn.reshape([B, h, w, h, w]) + rel_h + rel_w
        return attn.reshape([B, h * w, h * w])

    def forward(self, x):
        B, H, W, C = paddle.shape(x)

        qkv = self.qkv(x).reshape(
            [B, H * W, 3, self.num_heads, self.head_dim]).transpose(
                [2, 0, 3, 1, 4]).reshape(
                    [3, B * self.num_heads, H * W, self.head_dim])
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q.matmul(k.transpose([0, 2, 1])) * self.scale

        if self.use_rel_pos:
            attn = self.add_decomposed_rel_pos(attn, q, H, W)

        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = attn.matmul(v).reshape(
            [B, self.num_heads, H * W, self.head_dim]).transpose(
                [0, 2, 1, 3]).reshape([B, H, W, C])
        x = self.proj(x)
        return x


class Block(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 init_values=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 use_rel_pos=True,
                 rel_pos_zero_init=True,
                 window_size=None,
                 input_size=None,
                 act_layer='nn.GELU',
                 norm_layer='nn.LayerNorm',
                 lr_factor=1.0,
                 epsilon=1e-5):
        super().__init__()
        self.window_size = window_size

        self.norm1 = eval(norm_layer)(dim,
                                      weight_attr=ParamAttr(
                                          learning_rate=lr_factor,
                                          regularizer=L2Decay(0.0)),
                                      bias_attr=ParamAttr(
                                          learning_rate=lr_factor,
                                          regularizer=L2Decay(0.0)),
                                      epsilon=epsilon)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            window_size=window_size,
            input_size=input_size,
            lr_factor=lr_factor)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = eval(norm_layer)(dim,
                                      weight_attr=ParamAttr(
                                          learning_rate=lr_factor,
                                          regularizer=L2Decay(0.0)),
                                      bias_attr=ParamAttr(
                                          learning_rate=lr_factor,
                                          regularizer=L2Decay(0.0)),
                                      epsilon=epsilon)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer,
                       drop=drop,
                       lr_factor=lr_factor)
        if init_values is not None:
            self.gamma_1 = self.create_parameter(
                shape=([dim]), default_initializer=Constant(value=init_values))
            self.gamma_2 = self.create_parameter(
                shape=([dim]), default_initializer=Constant(value=init_values))
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        _, H, W, _ = paddle.shape(x)
        y = self.norm1(x)
        if self.window_size is not None:
            y, pad_hw, num_hw = window_partition(y, self.window_size)
        y = self.attn(y)
        if self.gamma_1 is not None:
            y = self.gamma_1 * y
        
        if self.window_size is not None:
            y = window_unpartition(y, pad_hw, num_hw, (H, W))
        x = x + self.drop_path(y)
        if self.gamma_2 is None:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Layer):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=(224, 224), patch_size=16, in_chans=3, embed_dim=768,
                 lr_factor=0.01):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv2D(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            weight_attr=ParamAttr(learning_rate=lr_factor),
            bias_attr=ParamAttr(learning_rate=lr_factor))
    
    @property
    def num_patches_in_h(self):
        return self.img_size[1] // self.patch_size

    @property
    def num_patches_in_w(self):
        return self.img_size[0] // self.patch_size
    
    def forward(self, x):
        out = self.proj(x)
        return out


@register
@serializable
class VisionTransformer(nn.Layer):
    """ Vision Transformer with support for patch input
    """
    def __init__(self,
                 img_size=(1024, 1024),
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 init_values=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_layer='nn.GELU',
                 norm_layer='nn.LayerNorm',
                 lr_decay_rate=1.0,
                 global_attn_indexes=(2, 5, 8, 11),
                 use_abs_pos=False,
                 use_rel_pos=True,
                 use_learnt_abs_pos=False,
                 rel_pos_zero_init=True,
                 epsilon=1e-5,
                 final_norm=False,
                 pretrained=None,
                 window_size=14,
                 out_indices=(11, ),
                 with_fpn=False,
                 use_checkpoint=False,
                 use_cls_token=False,
                 *args,
                 **kwargs):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth
        self.global_attn_indexes = global_attn_indexes
        self.epsilon = epsilon
        self.with_fpn = with_fpn
        self.use_checkpoint = use_checkpoint

        self.patch_h = img_size[0] // patch_size
        self.patch_w = img_size[1] // patch_size
        self.num_patches = self.patch_h * self.patch_w
        self.use_abs_pos = use_abs_pos
        self.use_cls_token = use_cls_token
        self.use_learnt_abs_pos = use_learnt_abs_pos

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        dpr = np.linspace(0, drop_path_rate, depth)
        if use_checkpoint:
            paddle.seed(0)
        if use_cls_token:
            self.cls_token = self.create_parameter(
                shape=(1, 1, embed_dim),
                default_initializer=paddle.nn.initializer.Constant(value=0.))

        if use_learnt_abs_pos:
            self.pos_w = self.patch_embed.num_patches_in_w
            self.pos_h = self.patch_embed.num_patches_in_h
            self.pos_embed = self.create_parameter(
                shape=(1, self.pos_w * self.pos_h + 1, embed_dim),
                default_initializer=paddle.nn.initializer.TruncatedNormal(
                    std=.02))
        else:
            self.pos_embed = None

        self.blocks = nn.LayerList([
            Block(
                embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=None
                if i in self.global_attn_indexes else window_size,
                input_size=[self.patch_h, self.patch_w],
                act_layer=act_layer,
                lr_factor=self.get_vit_lr_decay_rate(i, lr_decay_rate),
                norm_layer=norm_layer,
                init_values=init_values,
                epsilon=epsilon) for i in range(depth)
        ])

        assert len(out_indices) <= 4, 'out_indices out of bound'
        self.out_indices = out_indices
        self.pretrained = pretrained
        self.init_weight()

        self.out_channels = [embed_dim for _ in range(len(out_indices))]
        self.out_strides = [4, 8, 16, 32][-len(out_indices):] if with_fpn else [patch_size for _ in range(len(out_indices))]
        self.norm = Identity()
        if self.with_fpn:
            self.init_fpn(embed_dim=embed_dim,
                          patch_size=patch_size, 
                          out_with_norm=final_norm)

    def get_vit_lr_decay_rate(self, layer_id, lr_decay_rate):
        return lr_decay_rate**(self.depth - layer_id)

    def init_weight(self):
        pretrained = self.pretrained
        if pretrained:
            if 'http' in pretrained:
                path = paddle.utils.download.get_weights_path_from_url(
                    pretrained)
            else: 
                path = pretrained

            load_state_dict = paddle.load(path)
            model_state_dict = self.state_dict()
            pos_embed_name = "pos_embed"

            if pos_embed_name in load_state_dict.keys() and self.use_learnt_abs_pos:
                load_pos_embed = paddle.to_tensor(
                    load_state_dict[pos_embed_name], dtype="float32")
                if self.pos_embed.shape != load_pos_embed.shape:
                    pos_size = int(math.sqrt(load_pos_embed.shape[1] - 1))
                    model_state_dict[pos_embed_name] = self.resize_pos_embed(
                        load_pos_embed, (pos_size, pos_size),
                        (self.pos_h, self.pos_w))

                    # self.set_state_dict(model_state_dict)
                    load_state_dict[pos_embed_name] = model_state_dict[
                        pos_embed_name]

                    print("Load pos_embed and resize it from {} to {} .".format(
                        load_pos_embed.shape, self.pos_embed.shape))

            self.set_state_dict(load_state_dict)
            print("Load load_state_dict....")

    def init_fpn(self, embed_dim=768, patch_size=16, out_with_norm=False):
        if patch_size == 16:
            self.fpn1 = nn.Sequential(
                nn.Conv2DTranspose(
                    embed_dim, embed_dim, kernel_size=2, stride=2),
                nn.BatchNorm2D(embed_dim),
                nn.GELU(),
                nn.Conv2DTranspose(
                    embed_dim, embed_dim, kernel_size=2, stride=2), )

            self.fpn2 = nn.Sequential(
                nn.Conv2DTranspose(
                    embed_dim, embed_dim, kernel_size=2, stride=2), )

            self.fpn3 = Identity()

            self.fpn4 = nn.MaxPool2D(kernel_size=2, stride=2)
        elif patch_size == 8:
            self.fpn1 = nn.Sequential(
                nn.Conv2DTranspose(
                    embed_dim, embed_dim, kernel_size=2, stride=2), )

            self.fpn2 = Identity()

            self.fpn3 = nn.Sequential(nn.MaxPool2D(kernel_size=2, stride=2), )

            self.fpn4 = nn.Sequential(nn.MaxPool2D(kernel_size=4, stride=4), )

        if not out_with_norm:
            self.norm = Identity()
        else:
            self.norm = nn.LayerNorm(embed_dim, epsilon=self.epsilon)
    
    def resize_pos_embed(self, pos_embed, old_hw, new_hw):
        """
        Resize pos_embed weight.
        Args:
            pos_embed (Tensor): the pos_embed weight
            old_hw (list[int]): the height and width of old pos_embed
            new_hw (list[int]): the height and width of new pos_embed
        Returns:
            Tensor: the resized pos_embed weight
        """
        cls_pos_embed = pos_embed[:, :1, :]
        pos_embed = pos_embed[:, 1:, :]

        pos_embed = pos_embed.transpose([0, 2, 1])
        pos_embed = pos_embed.reshape([1, -1, old_hw[0], old_hw[1]])
        pos_embed = F.interpolate(
            pos_embed, new_hw, mode='bicubic', align_corners=False)
        pos_embed = pos_embed.flatten(2).transpose([0, 2, 1])
        pos_embed = paddle.concat([cls_pos_embed, pos_embed], axis=1)

        return pos_embed

    def get_2d_sincos_position_embedding(self, h, w, temperature=10000.):
        grid_y, grid_x = paddle.meshgrid(
            paddle.arange(
                h, dtype=paddle.float32),
            paddle.arange(
                w, dtype=paddle.float32))
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim // 4
        omega = paddle.arange(pos_dim, dtype=paddle.float32) / pos_dim
        omega = (1. / (temperature**omega)).unsqueeze(0)

        out_x = grid_x.reshape([-1, 1]).matmul(omega)
        out_y = grid_y.reshape([-1, 1]).matmul(omega)

        pos_emb = paddle.concat(
            [
                paddle.sin(out_y), paddle.cos(out_y), paddle.sin(out_x),
                paddle.cos(out_x)
            ],
            axis=1)

        return pos_emb.reshape([1, h, w, self.embed_dim])

    def forward(self, inputs):
        x = self.patch_embed(inputs['image']).transpose([0, 2, 3, 1])
        B, Hp, Wp, _ = paddle.shape(x)
        
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(
                (B, self.cls_token.shape[-2], self.cls_token.shape[-1]))
            x = x.flatten(2).transpose([0, 2, 1])  # b * hw * c
            x = paddle.concat([cls_tokens, x], axis=1)
        
        if self.use_abs_pos:
            x = x + self.get_2d_sincos_position_embedding(Hp, Wp)

        if self.use_learnt_abs_pos:
            x = x + self.resize_pos_embed(self.pos_embed, (self.pos_h, self.pos_w), (Hp, Wp))

        feats = []
        for idx, blk in enumerate(self.blocks):
            if self.use_checkpoint and self.training:
                x = paddle.distributed.fleet.utils.recompute(
                    blk, x, **{"preserve_rng_state": True})
            else:
                x = blk(x)
            if idx in self.out_indices:
                feats.append(self.norm(x.transpose([0, 3, 1, 2])))

        if self.with_fpn:
            fpns = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
            for i in range(len(feats)):
                feats[i] = fpns[i](feats[i])
        return feats

    @property
    def num_layers(self):
        return len(self.blocks)

    @property
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=c, stride=s)
            for c, s in zip(self.out_channels, self.out_strides)
        ]