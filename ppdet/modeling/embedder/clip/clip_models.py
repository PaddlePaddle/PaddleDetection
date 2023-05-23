# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#
# This code is based on: https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import Normal, Constant, Assign

from ppdet.modeling.layers import MultiHeadAttention
from ppdet.modeling.initializer import zeros_, normal_
from ppdet.core.workspace import register

from .models import ModifiedResNet, ClipViT
from .layers import Transformer


@register
class CLIP(nn.Layer):
    def __init__(self,
                 embed_dim=512,
                 # vision
                 image_resolution=224,
                 vision_layers=12,
                 vision_width=768,
                 vision_patch_size=32,
                 # text
                 context_length=77,
                 vocab_size=49408,
                 transformer_width=512,
                 transformer_heads=8,
                 transformer_layers=12,
                 ):
        super().__init__()
        self.context_length = context_length
        self.embed_dim = embed_dim

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = ClipViT(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)

        positional_embedding = self.create_parameter(
            shape=(self.context_length, transformer_width),
            default_initializer=Assign(
                paddle.empty((self.context_length, transformer_width))
            )
        )
        self.add_parameter("positional_embedding", positional_embedding)

        self.ln_final = nn.LayerNorm(transformer_width)

        text_projection = self.create_parameter(
            shape=(transformer_width, embed_dim),
            default_initializer=Assign(
                paddle.empty((transformer_width, embed_dim))
            )
        )
        self.add_parameter("text_projection", text_projection)
        # self.text_projection = nn.Linear(
        #     transformer_width, embed_dim, bias_attr=False)

        logit_scale = self.create_parameter(
            shape=(1,),
            default_initializer=Assign(paddle.ones([1]))
        )
        self.add_parameter("logit_scale", logit_scale)

        self.initialize_parameters()

    def initialize_parameters(self):
        Normal(std=0.02)(self.token_embedding.weight)
        Normal(std=0.01)(self.positional_embedding)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.embed_dim ** -0.5
                # normal_ = Normal(std=std)
                normal_(self.visual.attnpool.attn.q_proj.weight, std=std)
                normal_(self.visual.attnpool.attn.k_proj.weight, std=std)
                normal_(self.visual.attnpool.attn.v_proj.weight, std=std)
                normal_(self.visual.attnpool.attn.out_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        Constant(value=0.0)(param)

        proj_std = (self.transformer.width ** -0.5) * \
            ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5

        for resblock in self.transformer.resblocks:
            # normal_ = Normal(std=attn_std)
            normal_(resblock.attn.q_proj.weight, std=attn_std)
            normal_(resblock.attn.k_proj.weight, std=attn_std)
            normal_(resblock.attn.v_proj.weight, std=attn_std)
            normal_(resblock.attn.out_proj.weight, std=proj_std)
            normal_(resblock.mlp.c_fc.weight, std=fc_std)
            normal_(resblock.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            Normal(std=self.transformer.width ** -0.5)(self.text_projection)

    def build_attention_mask(self):
        mask = paddle.full(
            (self.context_length, self.context_length), float("-inf")
        )
        mask = paddle.triu(mask, diagonal=1)
        return mask

    def encode_image(self, image):
        return self.visual(image)

    def encode_text(self, text):
        x = self.token_embedding(text)
        x = x + self.positional_embedding
        x = self.transformer(x)
        x = self.ln_final(x)

        select = []
        index = zip(
            paddle.arange(x.shape[0]).numpy(),
            text.argmax(axis=-1).numpy()
        )
        for i, j in index:
            select.append(x[int(i), int(j)])

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / \
            image_features.norm(axis=-1, keepdim=True)
        text_features = text_features / \
            text_features.norm(axis=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text