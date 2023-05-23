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
import os
import wget
import paddle
from paddle.vision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from .clip_models import CLIP
from ppdet.utils.checkpoint import match_state_dict


def get_transforms(image_resolution):
    transforms = Compose([
        Resize(image_resolution, interpolation='bicubic'),
        CenterCrop(image_resolution),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])
    return transforms


def clip_rn50():
    model = CLIP(
        embed_dim=1024,
        image_resolution=224,
        vision_layers=(3, 4, 6, 3),
        vision_width=64,
        vision_patch_size=None,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12
    )
    return model, get_transforms(224)


def clip_rn101():
    model = CLIP(
        embed_dim=512,
        image_resolution=224,
        vision_layers=(3, 4, 23, 3),
        vision_width=64,
        vision_patch_size=None,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12
    )
    return model, get_transforms(224)


def clip_rn50x4():
    model = CLIP(
        embed_dim=640,
        image_resolution=288,
        vision_layers=(4, 6, 10, 6),
        vision_width=80,
        vision_patch_size=None,
        context_length=77,
        vocab_size=49408,
        transformer_width=640,
        transformer_heads=10,
        transformer_layers=12
    )
    return model, get_transforms(288)


def clip_vit_b_32():
    model = CLIP(
        embed_dim=512,
        image_resolution=224,
        vision_layers=12,
        vision_width=768,
        vision_patch_size=32,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12
    )
    return model, get_transforms(224)


model_dict = {
    'RN50': [clip_rn50, r'https://bj.bcebos.com/v1/paddledet/models/clip/RN50_clip.pdparams', 'RN50_clip.pdparams'],
    'RN50x4': [clip_rn50x4, r'https://bj.bcebos.com/v1/paddledet/models/clip/RN50x4_clip.pdparams', 'RN50x4_clip.pdparams'],
    'RN101': [clip_rn101, r'https://bj.bcebos.com/v1/paddledet/models/clip/RN101_clip.pdparams', 'RN101_clip.pdparams'],
    'ViT_B_32': [clip_vit_b_32, r'https://bj.bcebos.com/v1/paddledet/models/clip/ViT-B-32_clip.pdparams', 'ViT-B-32_clip.pdparams']
}


