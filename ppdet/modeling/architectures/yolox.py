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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ppdet.core.workspace import register, create
from .meta_arch import BaseArch
import random
import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from ppdet.modeling.ops import paddle_distributed_is_initialized

__all__ = ['YOLOX']


@register
class YOLOX(BaseArch):
    __category__ = 'architecture'

    def __init__(self,
                 backbone='CSPDarkNet',
                 neck='YOLOCSPPAN',
                 head='YOLOXHead',
                 for_mot=False,
                 input_size=(640, 640),
                 size_stride=32,
                 size_range=(15, 25),
                 random_interval=10):
        """
        YOLOX network, see https://arxiv.org/abs/2107.08430

        Args:
            backbone (nn.Layer): backbone instance
            neck (nn.Layer): neck instance
            head (nn.Layer): head instance
            for_mot (bool): whether used for MOT or not
            input_size (tuple[int]): initial scale, will be reset by self._preprocess()
            size_stride (int): stride of the size range
            size_range (tuple[int]): multi-scale range for training
            random_interval (int): interval of iter to change self._input_size
        """
        super(YOLOX, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.for_mot = for_mot

        self.init_input_size = input_size
        self._input_size = paddle.to_tensor(input_size)
        self.size_stride = size_stride
        self.size_range = size_range
        self.random_interval = random_interval
        self._step = 0

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])

        # fpn
        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs)

        # head
        kwargs = {'input_shape': neck.out_shape}
        head = create(cfg['head'], **kwargs)

        return {
            'backbone': backbone,
            'neck': neck,
            "head": head,
        }

    def _forward(self):
        if self.training:
            self._preprocess()
        body_feats = self.backbone(self.inputs)
        neck_feats = self.neck(body_feats, self.for_mot)

        if self.training:
            return self.head(neck_feats, self.inputs)
        else:
            head_outs = self.head(neck_feats)
            bbox, bbox_num = self.head.post_process(
                head_outs, self.inputs['im_shape'], self.inputs['scale_factor'])
            return {'bbox': bbox, 'bbox_num': bbox_num}

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()

    def _preprocess(self):
        self._get_size()
        scale_y = self._input_size[0] / self.init_input_size[0]
        scale_x = self._input_size[1] / self.init_input_size[1]
        if scale_x != 1 or scale_y != 1:
            self.inputs['image'] = F.interpolate(
                self.inputs['image'],
                size=self._input_size,
                mode='bilinear',
                align_corners=False)
            gt_bboxes = self.inputs['gt_bbox']
            for i in range(len(gt_bboxes)):
                if len(gt_bboxes[i]) > 0:
                    gt_bboxes[i][:, 0::2] = gt_bboxes[i][:, 0::2] * scale_x
                    gt_bboxes[i][:, 1::2] = gt_bboxes[i][:, 1::2] * scale_y
            self.inputs['gt_bbox'] = gt_bboxes

    def _get_size(self):
        image_ratio = self._input_size[1] * 1.0 / self._input_size[0]
        if self._step % self.random_interval == 0:
            size_range = random.randint(*self.size_range)
            size = [
                self.size_stride * size_range,
                self.size_stride * int(size_range * image_ratio)
            ]
            size = paddle.to_tensor(size)
            if dist.get_world_size() > 1 and paddle_distributed_is_initialized(
            ):
                dist.barrier()
                dist.broadcast(size, 0)
            self._input_size = size
        self._step += 1
