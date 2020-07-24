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

import numpy as np
from collections import OrderedDict

from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay

from ppdet.core.workspace import register
from ppdet.modeling.ops import SSDOutputDecoder
from ppdet.modeling.losses import SSDWithLmkLoss

__all__ = ['BlazeFace']


@register
class BlazeFace(object):
    """
    BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs,
               see https://arxiv.org/abs/1907.05047

    Args:
        backbone (object): backbone instance
        output_decoder (object): `SSDOutputDecoder` instance
        min_sizes (list|None): min sizes of generated prior boxes.
        max_sizes (list|None): max sizes of generated prior boxes. Default: None.
        steps (list|None): step size of adjacent prior boxes on each feature map.
        num_classes (int): number of output classes
        use_density_prior_box (bool): whether or not use density_prior_box
            instead of prior_box
        densities (list|None): the densities of generated density prior boxes,
            this attribute should be a list or tuple of integers
    """

    __category__ = 'architecture'
    __inject__ = ['backbone', 'output_decoder']
    __shared__ = ['num_classes']

    def __init__(self,
                 backbone="BlazeNet",
                 output_decoder=SSDOutputDecoder().__dict__,
                 min_sizes=[[16., 24.], [32., 48., 64., 80., 96., 128.]],
                 max_sizes=None,
                 steps=[8., 16.],
                 num_classes=2,
                 use_density_prior_box=False,
                 densities=[[2, 2], [2, 1, 1, 1, 1, 1]],
                 with_lmk=False,
                 lmk_loss=SSDWithLmkLoss().__dict__):
        super(BlazeFace, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.with_lmk = with_lmk
        self.output_decoder = output_decoder
        if isinstance(output_decoder, dict):
            if self.with_lmk:
                output_decoder['return_index'] = True
            self.output_decoder = SSDOutputDecoder(**output_decoder)
        self.min_sizes = min_sizes
        self.max_sizes = max_sizes
        self.steps = steps
        self.use_density_prior_box = use_density_prior_box
        self.densities = densities
        self.landmark = None
        if self.with_lmk and isinstance(lmk_loss, dict):
            self.lmk_loss = SSDWithLmkLoss(**lmk_loss)

    def build(self, feed_vars, mode='train'):
        im = feed_vars['image']

        body_feats = self.backbone(im)
        locs, confs, box, box_var = self._multi_box_head(
            inputs=body_feats,
            image=im,
            num_classes=self.num_classes,
            use_density_prior_box=self.use_density_prior_box)

        if mode == 'train':
            gt_bbox = feed_vars['gt_bbox']
            gt_class = feed_vars['gt_class']
            if self.with_lmk:
                lmk_labels = feed_vars['gt_keypoint']
                lmk_ignore_flag = feed_vars["keypoint_ignore"]
                loss = self.lmk_loss(locs, confs, gt_bbox, gt_class,
                                     self.landmark, lmk_labels, lmk_ignore_flag,
                                     box, box_var)
            else:
                loss = fluid.layers.ssd_loss(
                    locs,
                    confs,
                    gt_bbox,
                    gt_class,
                    box,
                    box_var,
                    overlap_threshold=0.35,
                    neg_overlap=0.35)

            loss = fluid.layers.reduce_sum(loss)
            return {'loss': loss}
        else:
            if self.with_lmk:
                pred, face_index = self.output_decoder(locs, confs, box,
                                                       box_var)
                return {
                    'bbox': pred,
                    'face_index': face_index,
                    'prior_boxes': box,
                    'landmark': self.landmark
                }
            else:
                pred = self.output_decoder(locs, confs, box, box_var)
                return {'bbox': pred}

    def _multi_box_head(self,
                        inputs,
                        image,
                        num_classes=2,
                        use_density_prior_box=False):
        def permute_and_reshape(input, last_dim):
            trans = fluid.layers.transpose(input, perm=[0, 2, 3, 1])
            compile_shape = [0, -1, last_dim]
            return fluid.layers.reshape(trans, shape=compile_shape)

        locs, confs = [], []
        boxes, vars = [], []
        lmk_locs = []
        b_attr = ParamAttr(learning_rate=2., regularizer=L2Decay(0.))

        for i, input in enumerate(inputs):
            min_size = self.min_sizes[i]

            if use_density_prior_box:
                densities = self.densities[i]
                box, var = fluid.layers.density_prior_box(
                    input,
                    image,
                    densities=densities,
                    fixed_sizes=min_size,
                    fixed_ratios=[1.],
                    clip=False,
                    offset=0.5,
                    steps=[self.steps[i]] * 2)
            else:
                box, var = fluid.layers.prior_box(
                    input,
                    image,
                    min_sizes=min_size,
                    max_sizes=None,
                    steps=[self.steps[i]] * 2,
                    aspect_ratios=[1.],
                    clip=False,
                    flip=False,
                    offset=0.5)

            num_boxes = box.shape[2]

            box = fluid.layers.reshape(box, shape=[-1, 4])
            var = fluid.layers.reshape(var, shape=[-1, 4])
            num_loc_output = num_boxes * 4
            num_conf_output = num_boxes * num_classes
            # get loc
            mbox_loc = fluid.layers.conv2d(
                input, num_loc_output, 3, 1, 1, bias_attr=b_attr)
            loc = permute_and_reshape(mbox_loc, 4)
            # get conf
            mbox_conf = fluid.layers.conv2d(
                input, num_conf_output, 3, 1, 1, bias_attr=b_attr)
            conf = permute_and_reshape(mbox_conf, num_classes)

            if self.with_lmk:
                # get landmark
                lmk_loc_output = num_boxes * 10
                lmk_box_loc = fluid.layers.conv2d(
                    input,
                    lmk_loc_output,
                    3,
                    1,
                    1,
                    param_attr=ParamAttr(name='lmk' + str(i) + '_weights'),
                    bias_attr=False)
                lmk_loc = permute_and_reshape(lmk_box_loc, 10)
                lmk_locs.append(lmk_loc)

            locs.append(loc)
            confs.append(conf)
            boxes.append(box)
            vars.append(var)

        face_mbox_loc = fluid.layers.concat(locs, axis=1)
        face_mbox_conf = fluid.layers.concat(confs, axis=1)
        prior_boxes = fluid.layers.concat(boxes)
        box_vars = fluid.layers.concat(vars)
        if self.with_lmk:
            self.landmark = fluid.layers.concat(lmk_locs, axis=1)
        return face_mbox_loc, face_mbox_conf, prior_boxes, box_vars

    def _inputs_def(self, image_shape):
        im_shape = [None] + image_shape
        # yapf: disable
        inputs_def = {
            'image':    {'shape': im_shape,  'dtype': 'float32', 'lod_level': 0},
            'im_id':    {'shape': [None, 1], 'dtype': 'int64',   'lod_level': 0},
            'gt_bbox':  {'shape': [None, 4], 'dtype': 'float32', 'lod_level': 1},
            'gt_class': {'shape': [None, 1], 'dtype': 'int32',   'lod_level': 1},
            'im_shape': {'shape': [None, 3], 'dtype': 'int32',   'lod_level': 0},
            'gt_keypoint':  {'shape': [None, 10], 'dtype': 'float32', 'lod_level': 1},
            'keypoint_ignore': {'shape': [None, 1], 'dtype': 'float32',   'lod_level': 1},
        }
        # yapf: enable
        return inputs_def

    def build_inputs(
            self,
            image_shape=[3, None, None],
            fields=['image', 'im_id', 'gt_bbox', 'gt_class'],  # for train
            use_dataloader=True,
            iterable=False):
        inputs_def = self._inputs_def(image_shape)
        feed_vars = OrderedDict([(key, fluid.data(
            name=key,
            shape=inputs_def[key]['shape'],
            dtype=inputs_def[key]['dtype'],
            lod_level=inputs_def[key]['lod_level'])) for key in fields])
        loader = fluid.io.DataLoader.from_generator(
            feed_list=list(feed_vars.values()),
            capacity=16,
            use_double_buffer=True,
            iterable=iterable) if use_dataloader else None
        return feed_vars, loader

    def train(self, feed_vars):
        return self.build(feed_vars, 'train')

    def eval(self, feed_vars):
        return self.build(feed_vars, 'eval')

    def test(self, feed_vars, exclude_nms=False):
        assert not exclude_nms, "exclude_nms for {} is not support currently".format(
            self.__class__.__name__)
        return self.build(feed_vars, 'test')

    def is_bbox_normalized(self):
        return True
