# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved. 
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

import math
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register
from paddle.nn.initializer import Normal, Constant
from IPython import embed


def _de_sigmoid(x, eps=1e-7):
    x = paddle.clip(x, eps, 1. / eps)
    x = paddle.clip(1. / x - 1., eps, 1. / eps)
    x = -paddle.log(x)
    return x


class LossParam(nn.Layer):
    def __init__(self, init_value=0.):
        super(LossParam, self).__init__()
        self.loss_param = self.create_parameter(
            shape=[1],
            attr=ParamAttr(initializer=Constant(value=init_value)),
            dtype="float32")

    def forward(self, inputs):
        out = paddle.exp(-self.loss_param) * inputs + self.loss_param
        return out


@register
class JDEHead(nn.Layer):
    __shared__ = ['num_classes']
    __inject__ = ['jde_loss']
    """
    JDEHead
    Args:
        anchors(list): Anchor parameters.
        anchor_masks(list): Anchor parameters.
        num_classes(int): Number of classes. Only support one class tracking.
        num_identifiers(int): Number of identifiers.
        embedding_dim(int): Embedding dimension. Default: 512.
        jde_loss    : 
        img_size(list): Input size of JDE network.
        ide_thresh  : Identification positive threshold. Default: 0.5.
        obj_thresh  : Objectness positive threshold. Default: 0.5.
        bkg_thresh  : Background positive threshold. Default: 0.4.
        s_box       : Weight for the box regression task.
        s_cls       : Weight for the classification task.
        s_ide       : Weight for the identifier classification task.
    """

    def __init__(
            self,
            anchors=[[8, 24], [11, 34], [16, 48], [23, 68], [32, 96],
                     [45, 135], [64, 192], [90, 271], [128, 384], [180, 540],
                     [256, 640], [512, 640]],
            anchor_masks=[[8, 9, 10, 11], [4, 5, 6, 7], [0, 1, 2, 3]],
            num_classes=1,
            num_identifiers=1,  # defined by dataset.nID
            embedding_dim=512,
            jde_loss='JDELoss',
            # img_size=[1888, 608],
            iou_aware=False,
            iou_aware_factor=0.4):
        super(JDEHead, self).__init__()
        self.num_classes = num_classes
        self.num_identifiers = num_identifiers
        self.embedding_dim = embedding_dim
        self.jde_loss = jde_loss
        # self.img_size = img_size
        self.iou_aware = iou_aware
        self.iou_aware_factor = iou_aware_factor

        self.shift = [1, 3, 5]
        self.emb_scale = math.sqrt(2) * math.log(
            self.num_identifiers - 1) if self.num_identifiers > 1 else 1

        self.parse_anchor(anchors, anchor_masks)
        self.num_outputs = len(self.anchors)

        self.yolo_outputs = []
        self.identify_outputs = []
        self.loss_params_cls = []
        self.loss_params_reg = []
        self.loss_params_ide = []
        for i in range(len(self.anchors)):
            if self.iou_aware:
                num_filters = len(self.anchors[i]) * (self.num_classes + 6)
            else:
                num_filters = len(self.anchors[i]) * (self.num_classes + 5)
            name = 'yolo_output.{}'.format(i)
            yolo_output = self.add_sublayer(
                name,
                nn.Conv2D(
                    in_channels=128 * (2**self.num_outputs) // (2**i),
                    out_channels=num_filters,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    weight_attr=ParamAttr(name=name + '.conv.weights'),
                    bias_attr=ParamAttr(
                        name=name + '.conv.bias', regularizer=L2Decay(0.))))
            self.yolo_outputs.append(yolo_output)

            name = 'identify_output.{}'.format(i)
            identify_output = self.add_sublayer(
                name,
                nn.Conv2D(
                    in_channels=64 * (2**self.num_outputs) // (2**i),
                    out_channels=embedding_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    weight_attr=ParamAttr(name=name + '.conv.weights'),
                    bias_attr=ParamAttr(
                        name=name + '.conv.bias', regularizer=L2Decay(0.))))
            self.identify_outputs.append(identify_output)

            loss_p_cls = self.add_sublayer('cls.{}'.format(i), LossParam(-4.15))
            self.loss_params_cls.append(loss_p_cls)
            loss_p_reg = self.add_sublayer('reg.{}'.format(i), LossParam(-4.85))
            self.loss_params_reg.append(loss_p_reg)
            loss_p_ide = self.add_sublayer('ide.{}'.format(i), LossParam(-2.3))
            self.loss_params_ide.append(loss_p_ide)

        self.classifier = self.add_sublayer(
            'classifier',
            nn.Linear(
                self.embedding_dim,
                self.num_identifiers,
                weight_attr=ParamAttr(
                    learning_rate=1., initializer=Normal(
                        mean=0.0, std=0.01)),
                bias_attr=ParamAttr(
                    learning_rate=2., regularizer=L2Decay(0.))))

    def parse_anchor(self, anchors, anchor_masks):
        self.anchors = [[anchors[i] for i in mask] for mask in anchor_masks]
        self.mask_anchors = []
        anchor_num = len(anchors)
        for masks in anchor_masks:
            self.mask_anchors.append([])
            for mask in masks:
                assert mask < anchor_num, "anchor mask index overflow"
                self.mask_anchors[-1].extend(anchors[mask])

    def forward(self, yolo_feats, identify_feats, targets=None, test_emb=False):
        assert len(yolo_feats) == len(identify_feats) == len(self.anchors)
        det_outs = []
        ide_outs = []
        for yolo_feat, ide_feat, yolo_head, ide_head in zip(
                yolo_feats, identify_feats, self.yolo_outputs,
                self.identify_outputs):
            det_out = yolo_head(yolo_feat)
            ide_out = ide_head(ide_feat)
            det_outs.append(det_out)
            ide_outs.append(ide_out)

        if self.training:
            return self.jde_loss(det_outs, ide_outs, targets, self.anchors,
                                 self.emb_scale, self.classifier,
                                 self.loss_params_cls, self.loss_params_reg,
                                 self.loss_params_ide)
        else:
            if test_emb:
                assert targets != None
                #embs_and_gts = self.get_emb_and_gt_outs(ide_outs, targets)
                #return embs_and_gts
                return targets
            else:
                yolo_outs = self.get_det_outs(det_outs)
                return yolo_outs

    def get_det_outs(self, det_outs):
        if self.iou_aware:
            y = []
            for i, out in enumerate(det_outs):
                na = len(self.anchors[i])
                ioup, x = out[:, 0:na, :, :], out[:, na:, :, :]
                b, c, h, w = x.shape
                no = c // na
                x = x.reshape((b, na, no, h * w))
                ioup = ioup.reshape((b, na, 1, h * w))
                obj = x[:, :, 4:5, :]
                ioup = F.sigmoid(ioup)
                obj = F.sigmoid(obj)
                obj_t = (obj**(1 - self.iou_aware_factor)) * (
                    ioup**self.iou_aware_factor)
                obj_t = _de_sigmoid(obj_t)
                loc_t = x[:, :, :4, :]
                cls_t = x[:, :, 5:, :]
                y_t = paddle.concat([loc_t, obj_t, cls_t], axis=2)
                y_t = y_t.reshape((b, c, h, w))
                y.append(y_t)
            return y
        else:
            return det_outs

    '''
    def get_emb_and_gt_outs(self, ide_outs, targets):
        embeddings, id_labels = [], []
        for emb_out, anchor in zip(ide_outs, self.anchors):
            nA = len(anchor)
            nGh, nGw = emb_out.shape[-2], emb_out.shape[-1]
            id_lbl = self.build_ide_targets_thres(targets, anchor, nA, nGh, nGw)

            id_mask = id_lbl != -1:
            embeddings.append(emb_out[id_mask])
            id_labels.append(id_lbl[id_mask])
        print('Computing pairwise similairity...')
        if len(embedding) <1 :
            return None

        embedding = paddle.stack(embedding, dim=0)
        n = len(id_labels)
        print(n, len(embedding))
        assert len(embedding) == n

        # return (embeddings, id_labels)
        embedding = F.normalize(embedding, dim=1) # 
        pdist = paddle.mm(embedding, embedding.T).cpu().numpy()
        gt = id_labels.expand(n,n).eq(id_labels.expand(n,n).T).numpy()
        
        up_triangle = np.where(np.triu(pdist)- np.eye(n)*pdist !=0)
        pdist = pdist[up_triangle]
        gt = gt[up_triangle]

        far_levels = [ 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        far,tar,threshold = metrics.roc_curve(gt, pdist)
        interp = interpolate.interp1d(far, tar)
        tar_at_far = [interp(x) for x in far_levels]
        for f,fa in enumerate(far_levels):
            print('TPR@FAR={:.7f}: {:.4f}'.format(fa, tar_at_far[f]))
        return tar_at_far
    
    def build_targets_thres(self, target, anchor_wh, nA, nGh, nGw):
        ID_THRESH = 0.5
        FG_THRESH = 0.5
        BG_THRESH = 0.4
        nB = len(target)  # number of images in batch
        assert(len(anchor_wh)==nA)

        tbox = paddle.zeros(nB, nA, nGh, nGw, 4).cuda()  # batch size, anchors, grid size
        tconf = torch.LongTensor(nB, nA, nGh, nGw).fill_(0).cuda()
        tid = torch.LongTensor(nB, nA, nGh, nGw, 1).fill_(-1).cuda() 
        for b in range(nB):
            t = target[b]
            t_id = t[:, 1].clone().long().cuda()
            t = t[:,[0,2,3,4,5]]
            nTb = len(t)  # number of targets
            if nTb == 0:
                continue

            gxy, gwh = t[: , 1:3].clone() , t[:, 3:5].clone()
            gxy[:, 0] = gxy[:, 0] * nGw
            gxy[:, 1] = gxy[:, 1] * nGh
            gwh[:, 0] = gwh[:, 0] * nGw
            gwh[:, 1] = gwh[:, 1] * nGh
            gxy[:, 0] = torch.clamp(gxy[:, 0], min=0, max=nGw -1)
            gxy[:, 1] = torch.clamp(gxy[:, 1], min=0, max=nGh -1)

            gt_boxes = torch.cat([gxy, gwh], dim=1)                                            # Shape Ngx4 (xc, yc, w, h)
            
            anchor_mesh = generate_anchor(nGh, nGw, anchor_wh)
            anchor_list = anchor_mesh.permute(0,2,3,1).contiguous().view(-1, 4)              # Shpae (nA x nGh x nGw) x 4
            #print(anchor_list.shape, gt_boxes.shape)
            iou_pdist = bbox_iou(anchor_list, gt_boxes)                                      # Shape (nA x nGh x nGw) x Ng
            iou_max, max_gt_index = torch.max(iou_pdist, dim=1)                              # Shape (nA x nGh x nGw), both

            iou_map = iou_max.view(nA, nGh, nGw)       
            gt_index_map = max_gt_index.view(nA, nGh, nGw)

            #nms_map = pooling_nms(iou_map, 3)
            
            id_index = iou_map > ID_THRESH
            fg_index = iou_map > FG_THRESH                                                    
            bg_index = iou_map < BG_THRESH 
            ign_index = (iou_map < FG_THRESH) * (iou_map > BG_THRESH)
            tconf[b][fg_index] = 1
            tconf[b][bg_index] = 0
            tconf[b][ign_index] = -1

            gt_index = gt_index_map[fg_index]
            gt_box_list = gt_boxes[gt_index]
            gt_id_list = t_id[gt_index_map[id_index]]
            #print(gt_index.shape, gt_index_map[id_index].shape, gt_boxes.shape)
            if torch.sum(fg_index) > 0:
                tid[b][id_index] =  gt_id_list.unsqueeze(1)
                fg_anchor_list = anchor_list.view(nA, nGh, nGw, 4)[fg_index] 
                delta_target = encode_delta(gt_box_list, fg_anchor_list)
                tbox[b][fg_index] = delta_target
        return tconf, tbox, tid
        '''
