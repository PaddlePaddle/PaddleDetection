# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import math
from scipy.optimize import linear_sum_assignment
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import KaimingUniform, Constant, Uniform
from ppdet.core.workspace import register
from ppdet.modeling.losses import CTFocalLoss
from ppdet.modeling.layers import DeformableConvV2
from ppdet.data.transform.op_helper import gaussian2D
from ppdet.modeling.losses.iou_loss import GIoULoss

class ConvLayer(nn.Layer):
    def __init__(self,
                 ch_in,
                 ch_out,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 use_dcn=False):
        super(ConvLayer, self).__init__()
        bias_attr = False
        fan_in = ch_in * kernel_size**2
        bound = 1 / math.sqrt(fan_in)
        param_attr = paddle.ParamAttr(initializer=Uniform(-bound, bound))
        if bias:
            bias_attr = paddle.ParamAttr(initializer=Constant(0.))
        if not use_dcn:
            self.conv = nn.Conv2D(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                weight_attr=param_attr,
                bias_attr=bias_attr)
        else:
            self.conv = DeformableConvV2(
                in_channels=ch_in,
                out_channels=ch_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                groups=groups,
                weight_attr=param_attr,
                bias_attr=bias_attr,
                lr_scale=1.,
                dcn_bias_regularizer=None,
                dcn_bias_lr_scale=1.)

    def forward(self, inputs):
        out = self.conv(inputs)

        return out


@register
class CenterNetHead(nn.Layer):
    """
    Args:
        in_channels (int): the channel number of input to CenterNetHead.
        num_classes (int): the number of classes, 80 by default.
        head_planes (int): the channel number in all head, 256 by default.
        heatmap_weight (float): the weight of heatmap loss, 1 by default.
        regress_ltrb (bool): whether to regress left/top/right/bottom or
            width/height for a box, true by default
        size_weight (float): the weight of box size loss, 0.1 by default.
        offset_weight (float): the weight of center offset loss, 1 by default.

    """

    __shared__ = ['num_classes']

    def __init__(self,
                 in_channels,
                 num_classes=80,
                 head_planes=256,
                 heatmap_weight=1,
                 regress_ltrb=True,
                 size_weight=0.1,
                 offset_weight=1,
		         use_dcn=False,
                 poto=False,
                 poto_alpha=0.8,
                 max_objs=500,
                 size_loss='l1',
                 poto_method='mul',
                 poto_heatmap_clip='True',
                 size_base=1.,
                 center_sampling_radius=0):
        super(CenterNetHead, self).__init__()
        self.weights = {
            'heatmap': heatmap_weight,
            'size': size_weight,
            'offset': offset_weight
        }
        self.heatmap = nn.Sequential(
            ConvLayer(
                in_channels, head_planes, kernel_size=3, padding=1, bias=True, use_dcn=use_dcn),
            nn.ReLU(),
            ConvLayer(
                head_planes,
                num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True))
        self.heatmap[2].conv.bias[:] = -2.19
        self.size = nn.Sequential(
            ConvLayer(
                in_channels, head_planes, kernel_size=3, padding=1, bias=True, use_dcn=use_dcn),
            nn.ReLU(),
            ConvLayer(
                head_planes,
                4 if regress_ltrb else 2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True))
        if not poto:
            self.offset = nn.Sequential(
                ConvLayer(
                    in_channels, head_planes, kernel_size=3, padding=1, bias=True, use_dcn=use_dcn),
                nn.ReLU(),
                ConvLayer(
                    head_planes, 2, kernel_size=1, stride=1, padding=0, bias=True))
        self.focal_loss = CTFocalLoss()
        self.poto = poto
        self.poto_alpha = poto_alpha
        self.max_objs = max_objs
        self.num_classes = num_classes
        if size_loss == 'giou':
            self.iou_loss = GIoULoss(reduction='sum')
        self.poto_method = poto_method
        self.poto_heatmap_clip = poto_heatmap_clip
        self.size_base = size_base
        self.center_sampling_radius = center_sampling_radius

    @classmethod
    def from_config(cls, cfg, input_shape):
        if isinstance(input_shape, (list, tuple)):
            input_shape = input_shape[0]
        return {'in_channels': input_shape.channels}

    def forward(self, feat, inputs):
        heatmap = self.heatmap(feat)
        size = self.size(feat)
        size = size * self.size_base
        if not self.poto:
            offset = self.offset(feat)
        else:
            offset = None
        if self.training:
            loss = self.get_loss(heatmap, size, offset, self.weights, inputs)
            return loss
        else:
            heatmap = F.sigmoid(heatmap)
            return {'heatmap': heatmap, 'size': size, 'offset': offset}

    def get_loss_w_poto(self, heatmap, size, offset, weights, inputs):
        inputs = self.get_ground_truth(heatmap, size, weights, inputs)
        return self.get_loss_wo_poto(heatmap, size, offset, weights, inputs)


    def get_loss_wo_poto(self, heatmap, size, offset, weights, inputs):
        heatmap_target = inputs['heatmap']
        size_target = inputs['size']
        if offset is not None:
            offset_target = inputs['offset']
        index = inputs['index']
        mask = inputs['index_mask']
        heatmap = paddle.clip(F.sigmoid(heatmap), 1e-4, 1 - 1e-4)
        heatmap_loss = self.focal_loss(heatmap, heatmap_target)

        size = paddle.transpose(size, perm=[0, 2, 3, 1])
        size_n, size_h, size_w, size_c = size.shape
        size = paddle.reshape(size, shape=[size_n, -1, size_c])
        index = paddle.unsqueeze(index, 2)
        batch_inds = list()
        for i in range(size_n):
            batch_ind = paddle.full(
                shape=[1, index.shape[1], 1], fill_value=i, dtype='int64')
            batch_inds.append(batch_ind)
        batch_inds = paddle.concat(batch_inds, axis=0)
        index = paddle.concat(x=[batch_inds, index], axis=2)
        pos_size = paddle.gather_nd(size, index=index)
        mask = paddle.unsqueeze(mask, axis=2)
        size_mask = paddle.expand_as(mask, pos_size)
        size_mask = paddle.cast(size_mask, dtype=pos_size.dtype)
        pos_num = size_mask.sum()
        size_mask.stop_gradient = True
        size_target.stop_gradient = True
        if hasattr(self, 'iou_loss'):
            symbol = np.ones(pos_size.shape, 'float32')
            symbol[:, :, 0:2] = -1
            symbol = paddle.to_tensor(symbol, dtype='float32')
            symbol.stop_gradient = True
            pred_size = pos_size * symbol
            gt_size = size_target * symbol
            gt_size.stop_gradient = True
            size_loss = self.iou_loss(pred_size * size_mask, gt_size * size_mask, iou_weight=size_mask)
        else:
            size_loss = F.l1_loss(
                pos_size * size_mask, size_target * size_mask, reduction='sum')
        size_loss = size_loss / (pos_num + 1e-4)


        det_loss = weights['heatmap'] * heatmap_loss + weights[
                'size'] * size_loss
        if offset is not None:
            offset = paddle.transpose(offset, perm=[0, 2, 3, 1])
            offset_n, offset_h, offset_w, offset_c = offset.shape
            offset = paddle.reshape(offset, shape=[offset_n, -1, offset_c])
            pos_offset = paddle.gather_nd(offset, index=index)
            offset_mask = paddle.expand_as(mask, pos_offset)
            offset_mask = paddle.cast(offset_mask, dtype=pos_offset.dtype)
            pos_num = offset_mask.sum()
            offset_mask.stop_gradient = True
            offset_target.stop_gradient = True
            offset_loss = F.l1_loss(
                pos_offset * offset_mask,
                offset_target * offset_mask,
                reduction='sum')
            offset_loss = offset_loss / (pos_num + 1e-4)
            det_loss = det_loss + weights['offset'] * offset_loss

        loss = {
            'det_loss': det_loss,
            'heatmap_loss': heatmap_loss,
            'size_loss': size_loss,
            'loss': det_loss
        }
        if offset is not None:
            loss.update({'offset_loss': offset_loss})
        return loss


    @paddle.no_grad()
    def pairwise_iou(self, gt_bbox, pred_bbox):
        num_gt = gt_bbox.shape[0]
        num_pred = pred_bbox.shape[0]
        gt_area = (gt_bbox[:, 2] - gt_bbox[:, 0]) * (gt_bbox[:, 3] - gt_bbox[:, 1])
        pred_area = (pred_bbox[:, 2] - pred_bbox[:, 0]) * (pred_bbox[:, 3] - pred_bbox[:, 1])
        expand_gt_area = paddle.transpose(paddle.expand(gt_area, shape=[num_pred, num_gt]), [1, 0])
        expand_pred_area = paddle.expand(pred_area, shape=[num_gt, num_pred])
        expand_gt_bbox = paddle.transpose(paddle.expand(gt_bbox, shape=[num_pred, num_gt, 4]), [1, 0, 2])
        expand_pred_bbox = paddle.expand(pred_bbox, shape=[num_gt, num_pred, 4])

        width_height = \
            paddle.minimum(expand_gt_bbox[:, :, 2:], expand_pred_bbox[:, :, 2:]) - paddle.maximum(expand_gt_bbox[:, :, :2], expand_pred_bbox[:, :, :2])

        width_height = paddle.clip(width_height, min=0)
        inter = paddle.prod(width_height, 2)
        iou = paddle.where(
            inter > 0,
            inter / (expand_gt_area + expand_pred_area - inter),
            paddle.zeros([1], dtype=inter.dtype))
        return iou 

    @paddle.no_grad()
    def _create_grid_offsets(self, size, stride, offset):
        bs, c, grid_height, grid_width = paddle.shape(size)
        shifts_start = offset * stride
        shifts_x = paddle.arange(
            shifts_start, grid_width * stride + shifts_start, step=stride,
            dtype=size.dtype
        )
        shifts_y = paddle.arange(
            shifts_start, grid_height * stride + shifts_start, step=stride,
            dtype=size.dtype
        )
        shift_y, shift_x = paddle.meshgrid(shifts_y, shifts_x)
        shift_x = paddle.reshape(shift_x, [-1, 1])
        shift_y = paddle.reshape(shift_y, [-1, 1])
        return shift_x, shift_y

    @paddle.no_grad()
    def get_ground_truth(self, heatmap, size, weights, inputs):
        gt_class = inputs['gt_class']
        gt_bbox = inputs['gt_bbox']
        gt_ide = inputs['gt_ide']
        index_mask = inputs['mask']
        #np.save('gt_class.npy', gt_class.numpy())
        #np.save('gt_bbox.npy', gt_bbox.numpy())
        #np.save('gt_ide.npy', gt_ide.numpy())
        #np.save('mask.npy', index_mask.numpy())


        bs, c, output_h, output_w = paddle.shape(heatmap).numpy()
        #bs = heatmap.shape[0]
        center_x, center_y = self._create_grid_offsets(size, 4, 0.5)
        center_xyxy = paddle.concat([center_x, center_y, center_x, center_y], 1)
        symbol = np.ones(center_xyxy.shape, 'float32')
        symbol[:, 0:2] = -1
        symbol = paddle.to_tensor(symbol, dtype='float32')
        
        target_heatmap_list = []
        target_bbox_size_list = []
        target_index_list = []
        target_mask_list = []
        target_reid_list = []        

        for i in range(bs):
            target_heatmap = np.zeros(
                (self.num_classes, output_h, output_w), dtype='float32')
            target_bbox_size = np.zeros((self.max_objs, 4), dtype=np.float32)
            target_index = np.zeros((self.max_objs, ), dtype=np.int64)
            target_mask = np.zeros((self.max_objs, ), dtype=np.int32)
            target_reid = np.zeros((self.max_objs, ), dtype=np.int64)
            

            gt_class_per_img = paddle.slice(gt_class, axes=[0], starts=[i], ends=[i+1])
            gt_bbox_per_img = paddle.slice(gt_bbox, axes=[0], starts=[i], ends=[i+1])
            gt_ide_per_img = paddle.slice(gt_ide, axes=[0], starts=[i], ends=[i+1])
            index_mask_per_img = paddle.slice(index_mask, axes=[0], starts=[i], ends=[i+1])
            index_mask_per_img = paddle.cast(index_mask_per_img, dtype='bool')
            gt_class_per_img = paddle.masked_select(gt_class_per_img, index_mask_per_img)
            gt_ide_per_img = paddle.masked_select(gt_ide_per_img, index_mask_per_img)
            index_mask_per_img = paddle.unsqueeze(index_mask_per_img, 2)
            index_mask_per_img = paddle.expand(index_mask_per_img, gt_bbox_per_img.shape)
            gt_bbox_per_img = paddle.masked_select(gt_bbox_per_img, index_mask_per_img)
            gt_bbox_per_img = paddle.reshape(gt_bbox_per_img, [-1, 4])
            if len(gt_class_per_img.shape) == 2:
                gt_class_per_img = paddle.squeeze(gt_class_per_img, 0)
                gt_bbox_per_img = paddle.squeeze(gt_bbox_per_img, 0)

            heatmap_per_img = paddle.slice(heatmap, axes=[0], starts=[i], ends=[i+1])
            size_per_img = paddle.slice(size, axes=[0], starts=[i], ends=[i+1])

            size_per_img = paddle.transpose(size_per_img, [0, 2, 3, 1])
            #np.save('size_per_img.npy', size_per_img.numpy())
            size_per_img = paddle.reshape(size_per_img, [-1, 4])
            
            pred_bbox_per_img = center_xyxy + symbol * size_per_img
            
            if self.poto_heatmap_clip:
                heatmap_per_img = paddle.clip(F.sigmoid(heatmap_per_img), 1e-4, 1 - 1e-4)
            else:
                heatmap_per_img = F.sigmoid(heatmap_per_img)
            heatmap_per_img = paddle.transpose(heatmap_per_img, [0, 2, 3, 1]) 
            #np.save('heatmap_per_img.npy', heatmap_per_img.numpy())
            heatmap_per_img = paddle.reshape(heatmap_per_img, [-1, self.num_classes])
            
            prob = paddle.gather(heatmap_per_img, gt_class_per_img, axis=1)
            #np.save('prob.npy', prob.numpy())
            prob = paddle.transpose(prob, [1, 0])
            gt_x1 = gt_bbox_per_img[:, 0] - gt_bbox_per_img[:, 2] / 2.
            gt_y1 = gt_bbox_per_img[:, 1] - gt_bbox_per_img[:, 3] / 2.
            gt_x2 = gt_x1 + gt_bbox_per_img[:, 2]
            gt_y2 = gt_y1 + gt_bbox_per_img[:, 3]
            gt_x1 = paddle.unsqueeze(gt_x1, 1) 
            gt_y1 = paddle.unsqueeze(gt_y1, 1)
            gt_x2 = paddle.unsqueeze(gt_x2, 1)
            gt_y2 = paddle.unsqueeze(gt_y2, 1)
            gt_xxyy = paddle.concat([gt_x1, gt_y1, gt_x2, gt_y2], 1)
            #np.save('gt_xxyy.npy', gt_xxyy.numpy())
            #np.save('pred_bbox_per_img.npy', pred_bbox_per_img.numpy())
            iou = self.pairwise_iou(gt_xxyy, pred_bbox_per_img)
            #np.save('iou.npy', iou.numpy())
            quality_cls = prob ** (1 - self.poto_alpha)
            quality_iou = iou ** self.poto_alpha
            if self.poto_method == 'mul':
                quality = quality_cls * quality_iou
            elif self.poto_method == 'add':
                quality = quality_cls + quality_iou
            #np.save('quality.npy', quality.numpy())
            num_gt, num_pred = quality.shape
            if self.center_sampling_radius > 0:
                gt_c_x1 = gt_bbox_per_img[:, 0] - self.center_sampling_radius * 4
                gt_c_y1 = gt_bbox_per_img[:, 1] - self.center_sampling_radius * 4
                gt_c_x2 = gt_c_x1 + self.center_sampling_radius * 4
                gt_c_y2 = gt_c_y1 + self.center_sampling_radius * 4
                gt_c_x1 = paddle.maximum(gt_c_x1, gt_xxyy[:, 0])                
                gt_c_y1 = paddle.maximum(gt_c_y1, gt_xxyy[:, 1])
                gt_c_x2 = paddle.minimum(gt_c_x2, gt_xxyy[:, 2])
                gt_c_y2 = paddle.minimum(gt_c_y2, gt_xxyy[:, 3])
                gt_c_x1 = paddle.unsqueeze(gt_c_x1, 1) 
                gt_c_y1 = paddle.unsqueeze(gt_c_y1, 1)
                gt_c_x2 = paddle.unsqueeze(gt_c_x2, 1)
                gt_c_y2 = paddle.unsqueeze(gt_c_y2, 1)
                gt_c_xxyy = paddle.concat([gt_c_x1, gt_c_y1, gt_c_x2, gt_c_y2], 1)
                expand_gt_x1 = paddle.transpose(paddle.expand(gt_c_x1, shape=[num_pred, num_gt, 1]), [1, 0, 2])
                expand_gt_y1 = paddle.transpose(paddle.expand(gt_c_y1, shape=[num_pred, num_gt, 1]), [1, 0, 2])
                expand_gt_x2 = paddle.transpose(paddle.expand(gt_c_x2, shape=[num_pred, num_gt, 1]), [1, 0, 2])
                expand_gt_y2 = paddle.transpose(paddle.expand(gt_c_y2, shape=[num_pred, num_gt, 1]), [1, 0, 2])
            else:
                expand_gt_x1 = paddle.transpose(paddle.expand(gt_x1, shape=[num_pred, num_gt, 1]), [1, 0, 2])
                expand_gt_y1 = paddle.transpose(paddle.expand(gt_y1, shape=[num_pred, num_gt, 1]), [1, 0, 2])
                expand_gt_x2 = paddle.transpose(paddle.expand(gt_x2, shape=[num_pred, num_gt, 1]), [1, 0, 2])
                expand_gt_y2 = paddle.transpose(paddle.expand(gt_y2, shape=[num_pred, num_gt, 1]), [1, 0, 2])
            expand_center_x = paddle.expand(center_x, shape=[num_gt, num_pred, 1])
            expand_center_y = paddle.expand(center_y, shape=[num_gt, num_pred, 1])
            x1_is_in = expand_center_x >= expand_gt_x1
            x2_is_in = expand_center_x <= expand_gt_x2
            y1_is_in = expand_center_y >= expand_gt_y1
            y2_is_in = expand_center_y <= expand_gt_y2
            is_in = paddle.concat([x1_is_in, y1_is_in, x2_is_in, y2_is_in], 2) 
            is_in = paddle.cast(is_in, dtype='int32')
            is_in = paddle.prod(is_in, axis=2)
            not_in = (1 - is_in) * -1
            not_in = paddle.cast(not_in, quality.dtype)
            #np.save('not_in.npy', not_in.numpy())
            quality = paddle.where(not_in == -1, not_in, quality)
            #np.save('quality_in.npy', quality.numpy())
            gt_idxs, shift_idxs = linear_sum_assignment(quality.numpy(), maximize=True)
            #np.save('gt_idxs.npy', gt_idxs)
            #np.save('shift_idxs.npy', shift_idxs)
            
            gt_bbox_per_img_data = gt_bbox_per_img.numpy()
            center_x_data = center_x.numpy()
            center_y_data = center_y.numpy()
            center_data = np.concatenate([center_x_data, center_y_data], 1)
            gt_class_per_img_data = gt_class_per_img.numpy()
            gt_xxyy_data = gt_xxyy.numpy()
            gt_ide_per_img_data = gt_ide_per_img.numpy()
            
            #print('shift_idxs: ', shift_idxs)
            for k, (gt_idx, shift_idx) in enumerate(zip(gt_idxs, shift_idxs)):
                bbox_w, bbox_h = gt_bbox_per_img_data[gt_idx, 2:]
                radius = self.gaussian_radius((math.ceil(bbox_h / 4.), math.ceil(bbox_w / 4.)))
                radius = max(0, int(radius))
                ct = center_data[shift_idx, :]
                ct_int = ((ct - 0.5 * 4 ) / 4).astype('int32')
                cls_id = gt_class_per_img_data[gt_idx]
                self.draw_truncate_gaussian(target_heatmap[cls_id], ct_int, radius,
                                                    radius)
                bbox_amodal = gt_xxyy_data[gt_idx]
                target_bbox_size[k] = ct[0] - bbox_amodal[0], ct[1] - bbox_amodal[1], \
                        bbox_amodal[2] - ct[0], bbox_amodal[3] - ct[1]

                target_index[k] = shift_idx
                target_mask[k] = 1
                target_reid[k] = gt_ide_per_img_data[gt_idx]
            target_heatmap_list.append(target_heatmap)
            target_bbox_size_list.append(target_bbox_size)
            target_index_list.append(target_index)
            target_mask_list.append(target_mask)
            target_reid_list.append(target_reid)
 
            import cv2
            image = inputs['image'].numpy()[i] * 255
            show_image = image.transpose((1, 2, 0)).astype('uint8')
            show_heatmap = target_heatmap * 255
            show_heatmap = show_heatmap.astype('uint8')
            show_heatmap = show_heatmap.transpose((1, 2, 0))
            show_heatmap = cv2.resize(show_heatmap, (show_image.shape[1], show_image.shape[0]))
            pseudo_image = np.zeros(show_image.shape, show_image.dtype)
            pseudo_image[:, :, 0] = show_heatmap
            pseudo_image[:, :, 1] = show_heatmap
            pseudo_image[:, :, 2] = show_heatmap
            show_heatmap = cv2.addWeighted(show_image, 0.3,
                                           pseudo_image, 0.7,
                                           0)
            
            cv2.imwrite('centernet_mot17half_coco_poto_imscale_giou_center/fairmot_heatmap.jpg', show_heatmap)
            
           
        inputs['heatmap'] = paddle.to_tensor(target_heatmap_list)
        inputs['size'] = paddle.to_tensor(target_bbox_size_list)
        inputs['index'] = paddle.to_tensor(target_index_list)
        inputs['index_mask'] = paddle.to_tensor(target_mask_list)
        inputs['reid'] = paddle.to_tensor(target_reid_list)
        
        return inputs

    def draw_truncate_gaussian(self, heatmap, center, h_radius, w_radius):
        h, w = 2 * h_radius + 1, 2 * w_radius + 1
        sigma_x = w / 6
        sigma_y = h / 6
        gaussian = gaussian2D((h, w), sigma_x, sigma_y)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, w_radius), min(width - x, w_radius + 1)
        top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[h_radius - top:h_radius + bottom, w_radius -
                                   left:w_radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            heatmap[y - top:y + bottom, x - left:x + right] = np.maximum(
                masked_heatmap, masked_gaussian)
        return heatmap


    def gaussian_radius(self, det_size, min_overlap=0.7):
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


    def get_loss(self, heatmap, size, offset, weights, inputs):
        if self.poto:
            return self.get_loss_w_poto(heatmap, size, offset, weights, inputs)
        else:
            return self.get_loss_wo_poto(heatmap, size, offset, weights, inputs)
