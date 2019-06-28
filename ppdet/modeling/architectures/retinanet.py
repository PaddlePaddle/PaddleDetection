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

import paddle.fluid as fluid

from ppdet.core.workspace import register

__all__ = ['RetinaNet']


@register
class RetinaNet(object):
    """
    RetinaNet architecture, see https://arxiv.org/abs/1708.02002

    Args:
        backbone (object): backbone instance
        fpn (object): feature pyramid network instance
        retina_head (object): `RetinaHead` instance
    """

    __category__ = 'architecture'
    __inject__ = ['backbone', 'fpn', 'retina_head']

    def __init__(self, backbone, fpn, retina_head):
        super(RetinaNet, self).__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.retina_head = retina_head

    def build(self, feed_vars, mode='train'):
        im = feed_vars['image']
        im_info = feed_vars['im_info']
        if mode == 'train':
            gt_box = feed_vars['gt_box']
            gt_label = feed_vars['gt_label']
            is_crowd = feed_vars['is_crowd']
        # backbone
        body_feats = self.backbone(im)

        # FPN
        body_feats, spatial_scale = self.fpn.get_output(body_feats)

        # retinanet head
        if mode == 'train':
            loss = self.retina_head.get_loss(body_feats, spatial_scale, im_info,
                                             gt_box, gt_label, is_crowd)
            total_loss = fluid.layers.sum(list(loss.values()))
            loss.update({'loss': total_loss})
            return loss
        else:
            pred = self.retina_head.get_prediction(body_feats, spatial_scale,
                                                   im_info)
            return pred

    def train(self, feed_vars):
        return self.build(feed_vars, 'train')

    def eval(self, feed_vars):
        return self.build(feed_vars, 'test')

    def test(self, feed_vars):
        return self.build(feed_vars, 'test')
