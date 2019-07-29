# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from collections import OrderedDict

from ppdet.core.workspace import register

__all__ = ['YOLOv3']


@register
class YOLOv3(object):
    """
    YOLOv3 network, see https://arxiv.org/abs/1804.02767

    Args:
        backbone (object): an backbone instance
        yolo_head (object): an `YOLOv3Head` instance
    """

    __category__ = 'architecture'
    __inject__ = ['backbone', 'yolo_head']

    def __init__(self, backbone, yolo_head='YOLOv3Head'):
        super(YOLOv3, self).__init__()
        self.backbone = backbone
        self.yolo_head = yolo_head

    def build(self, feed_vars, mode='train'):
        im = feed_vars['image']
        body_feats = self.backbone(im)

        if isinstance(body_feats, OrderedDict):
            body_feat_names = list(body_feats.keys())
            body_feats = [body_feats[name] for name in body_feat_names]

        if mode == 'train':
            gt_box = feed_vars['gt_box']
            gt_label = feed_vars['gt_label']
            gt_score = feed_vars['gt_score']

            return {
                'loss': self.yolo_head.get_loss(body_feats, gt_box, gt_label,
                                                gt_score)
            }
        else:
            im_size = feed_vars['im_size']
            return self.yolo_head.get_prediction(body_feats, im_size)

    def train(self, feed_vars):
        return self.build(feed_vars, mode='train')

    def eval(self, feed_vars):
        return self.build(feed_vars, mode='test')

    def test(self, feed_vars):
        return self.build(feed_vars, mode='test')
