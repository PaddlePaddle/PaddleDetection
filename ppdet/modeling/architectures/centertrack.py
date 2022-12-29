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
from ..keypoint_utils import get_affine_transform

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

        self.pre_images = None
        self.pre_hm = True

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
                det_outs = self.detector(self.inputs)
                # detection results
                return {
                    'bbox': det_outs['bbox'],
                    'bbox_num': det_outs['bbox_num']
                }

            else:
                meta = self.inputs

                for k in [
                        'center', 'scale', 'out_height', 'out_width',
                        'inp_height', 'inp_width', 'trans_input', 'trans_output'
                ]:
                    meta[k] = meta[k].numpy()

                # initializing tracker for the first frame
                if self.pre_images is None:
                    self.pre_images = meta['image']
                    self.tracker.init_track([])

                meta['pre_images'] = self.pre_images
                self.pre_images = meta['image']  # Note: update for next image

                pre_hms, pre_inds = None, None
                if self.pre_hm:
                    # render input heatmap from tracker status
                    # pre_inds is not used, it can learn an offset from previous
                    # image to the current image.
                    pre_hms, pre_inds = self.get_additional_inputs(
                        self.tracker.tracks, meta, with_hm=True)
                meta['pre_hms'] = pre_hms

                det_outs = self.detector(meta)
                neck_feat = det_outs['neck_feat']

                dets = self.plugin_head(
                    neck_feat, meta, det_outs['bbox'], det_outs['bbox_inds'],
                    det_outs['topk_clses'], det_outs['topk_ys'],
                    det_outs['topk_xs'])

                # convert the cropped and 4x downsampled output coordinate system
                # back to the input image coordinate system
                result = self.tracking_post_process(dets, meta)
                return result

    def get_pred(self):
        return self._forward()

    def get_loss(self):
        return self._forward()

    def reset_tracking(self):
        self.tracker.reset()
        self.pre_images = None

    def get_additional_inputs(self, dets, meta, with_hm=True):
        '''
        Render input heatmap from previous trackings.
        '''
        trans_input, trans_output = meta['trans_input'][0], meta[
            'trans_output'][0]
        inp_width, inp_height = meta['inp_width'][0], meta['inp_height'][0]
        out_width, out_height = meta['out_width'][0], meta['out_height'][0]
        input_hm = np.zeros((1, inp_height, inp_width), dtype=np.float32)

        output_inds = []
        for det in dets:
            if det['score'] < self.tracker.pre_thresh:  # or det['active'] == 0:
                continue
            bbox = affine_transform_bbox(det['bbox'], trans_input, inp_width,
                                         inp_height)
            bbox_out = affine_transform_bbox(det['bbox'], trans_output,
                                             out_width, out_height)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if (h > 0 and w > 0):
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                    dtype=np.float32)
                ct_int = ct.astype(np.int32)
                if with_hm:
                    input_hm[0] = draw_umich_gaussian(input_hm[0], ct_int,
                                                      radius)
                ct_out = np.array(
                    [(bbox_out[0] + bbox_out[2]) / 2,
                     (bbox_out[1] + bbox_out[3]) / 2],
                    dtype=np.int32)
                output_inds.append(ct_out[1] * out_width + ct_out[0])
        if with_hm:
            input_hm = paddle.to_tensor(input_hm[np.newaxis])
        output_inds = np.array(output_inds, np.int64).reshape(1, -1)
        output_inds = paddle.to_tensor(output_inds)
        return input_hm, output_inds

    def tracking_post_process(self, dets, meta):
        if not ('scores' in dets):
            return [{}], [{}]

        preds = []
        c, s, h, w = meta['center'], meta['scale'], meta['out_height'], meta[
            'out_width']
        trans = get_affine_transform(
            center=c[0],
            input_size=s[0],
            rot=0,
            output_size=[w[0], h[0]],
            shift=(0., 0.),
            inv=True).astype(np.float32)

        for j in range(len(dets['scores'])):
            if dets['scores'][j] < self.tracker.out_thresh:
                break
            item = {}
            item['score'] = dets['scores'][j]
            item['class'] = int(dets['clses'][j]) + 1
            item['ct'] = transform_preds_with_trans(
                dets['cts'][j].reshape([1, 2]), trans).reshape(2)

            if 'tracking' in dets:
                tracking = transform_preds_with_trans(
                    (dets['tracking'][j] + dets['cts'][j]).reshape([1, 2]),
                    trans).reshape(2)
                item['tracking'] = tracking - item['ct']

            if 'bboxes' in dets:
                bbox = transform_preds_with_trans(
                    dets['bboxes'][j].reshape([2, 2]), trans).reshape(4)
                item['bbox'] = bbox

            preds.append(item)
        return preds


def affine_transform_bbox(bbox, trans, width, height):
    bbox = np.array(copy.deepcopy(bbox), dtype=np.float32)
    bbox[:2] = affine_transform(bbox[:2], trans)
    bbox[2:] = affine_transform(bbox[2:], trans)
    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, width - 1)
    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, height - 1)
    return bbox


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:
                               radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def transform_preds_with_trans(coords, trans):
    target_coords = np.ones((coords.shape[0], 3), np.float32)
    target_coords[:, :2] = coords
    target_coords = np.dot(trans, target_coords.transpose()).transpose()
    return target_coords[:, :2]


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]
