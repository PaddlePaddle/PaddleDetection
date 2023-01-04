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

import copy
import math
import numpy as np
import paddle
from ppdet.core.workspace import register, create
from .meta_arch import BaseArch

from ..keypoint_utils import affine_transform
from ppdet.data.transform.op_helper import gaussian_radius, gaussian2D, draw_umich_gaussian

__all__ = ['CenterTrack']


@register
class CenterTrack(BaseArch):
    """
    CenterTrack network, see http://arxiv.org/abs/2004.01177

    Args:
        detector (object): 'CenterNet' instance
        plugin_head (object): 'CenterTrackHead' instance
        tracker (object): 'CenterTracker' instance
    """
    __category__ = 'architecture'
    __shared__ = ['mot_metric']

    def __init__(self,
                 detector='CenterNet',
                 plugin_head='CenterTrackHead',
                 tracker='CenterTracker',
                 mot_metric=False):
        super(CenterTrack, self).__init__()
        self.detector = detector
        self.plugin_head = plugin_head
        self.tracker = tracker
        self.mot_metric = mot_metric
        self.pre_image = None
        self.deploy = False

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        detector = create(cfg['detector'])
        detector_out_shape = detector.neck and detector.neck.out_shape or detector.backbone.out_shape

        kwargs = {'input_shape': detector_out_shape}
        plugin_head = create(cfg['plugin_head'], **kwargs)
        tracker = create(cfg['tracker'])

        return {
            'detector': detector,
            'plugin_head': plugin_head,
            'tracker': tracker,
        }

    def _forward(self):
        if self.training:
            det_outs = self.detector(self.inputs)
            neck_feat = det_outs['neck_feat']

            losses = {}
            for k, v in det_outs.items():
                if 'loss' not in k: continue
                losses.update({k: v})

            plugin_outs = self.plugin_head(neck_feat, self.inputs)
            for k, v in plugin_outs.items():
                if 'loss' not in k: continue
                losses.update({k: v})

            losses['loss'] = det_outs['det_loss'] + plugin_outs['plugin_loss']
            return losses

        else:
            if not self.mot_metric:
                # detection, support bs>=1
                det_outs = self.detector(self.inputs)
                return {
                    'bbox': det_outs['bbox'],
                    'bbox_num': det_outs['bbox_num']
                }

            else:
                # MOT, only support bs=1
                if not self.deploy:
                    if self.pre_image is None:
                        self.pre_image = self.inputs['image']
                        # initializing tracker for the first frame
                        self.tracker.init_track([])
                    self.inputs['pre_image'] = self.pre_image
                    self.pre_image = self.inputs[
                        'image']  # Note: update for next image

                    # render input heatmap from tracker status
                    pre_hm = self.get_additional_inputs(
                        self.tracker.tracks, self.inputs, with_hm=True)
                    self.inputs['pre_hm'] = paddle.to_tensor(pre_hm)

                # model inference
                det_outs = self.detector(self.inputs)
                neck_feat = det_outs['neck_feat']
                result = self.plugin_head(
                    neck_feat, self.inputs, det_outs['bbox'],
                    det_outs['bbox_inds'], det_outs['topk_clses'],
                    det_outs['topk_ys'], det_outs['topk_xs'])

                if not self.deploy:
                    # convert the cropped and 4x downsampled output coordinate system
                    # back to the input image coordinate system
                    result = self.plugin_head.centertrack_post_process(
                        result, self.inputs, self.tracker.out_thresh)
                return result

    def get_pred(self):
        return self._forward()

    def get_loss(self):
        return self._forward()

    def reset_tracking(self):
        self.tracker.reset()
        self.pre_image = None

    def get_additional_inputs(self, dets, meta, with_hm=True):
        # Render input heatmap from previous trackings.
        trans_input = meta['trans_input'][0].numpy()
        inp_width, inp_height = int(meta['inp_width'][0]), int(meta[
            'inp_height'][0])
        input_hm = np.zeros((1, inp_height, inp_width), dtype=np.float32)

        for det in dets:
            if det['score'] < self.tracker.pre_thresh:
                continue
            bbox = affine_transform_bbox(det['bbox'], trans_input, inp_width,
                                         inp_height)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if (h > 0 and w > 0):
                radius = gaussian_radius(
                    (math.ceil(h), math.ceil(w)), min_overlap=0.7)
                radius = max(0, int(radius))
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                    dtype=np.float32)
                ct_int = ct.astype(np.int32)
                if with_hm:
                    input_hm[0] = draw_umich_gaussian(input_hm[0], ct_int,
                                                      radius)
        if with_hm:
            input_hm = input_hm[np.newaxis]
        return input_hm


def affine_transform_bbox(bbox, trans, width, height):
    bbox = np.array(copy.deepcopy(bbox), dtype=np.float32)
    bbox[:2] = affine_transform(bbox[:2], trans)
    bbox[2:] = affine_transform(bbox[2:], trans)
    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, width - 1)
    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, height - 1)
    return bbox
