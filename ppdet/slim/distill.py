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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr

from ppdet.core.workspace import register, create, load_config
from ppdet.modeling import ops
from ppdet.utils.checkpoint import load_pretrain_weight
from ppdet.utils.logger import setup_logger

logger = setup_logger(__name__)


class DistillModel(nn.Layer):
    def __init__(self, cfg, slim_cfg):
        super(DistillModel, self).__init__()

        self.student_model = create(cfg.architecture)
        logger.debug('Load student model pretrain_weights:{}'.format(
            cfg.pretrain_weights))
        load_pretrain_weight(self.student_model, cfg.pretrain_weights)

        slim_cfg = load_config(slim_cfg)

        self.teacher_model = create(slim_cfg.architecture)
        self.distill_loss = create(slim_cfg.distill_loss)
        logger.debug('Load teacher model pretrain_weights:{}'.format(
            slim_cfg.pretrain_weights))
        load_pretrain_weight(self.teacher_model, slim_cfg.pretrain_weights)

        for param in self.teacher_model.parameters():
            param.trainable = False

    def parameters(self):
        return self.student_model.parameters()

    def forward(self, inputs):
        if self.training:
            teacher_loss = self.teacher_model(inputs)
            student_loss = self.student_model(inputs)
            loss = self.distill_loss(self.teacher_model, self.student_model)
            student_loss['distill_loss'] = loss
            student_loss['teacher_loss'] = teacher_loss['loss']
            student_loss['loss'] += student_loss['distill_loss']
            return student_loss
        else:
            return self.student_model(inputs)


class FGDDistillModel(nn.Layer):
    """
    Build FGD distill model.
    Args:
        cfg: The student config.
        slim_cfg: The teacher and distill config.
    """

    def __init__(self, cfg, slim_cfg):
        super(FGDDistillModel, self).__init__()

        self.is_inherit = True
        # build student model before load slim config
        self.student_model = create(cfg.architecture)
        self.arch = cfg.architecture
        stu_pretrain = cfg['pretrain_weights']
        slim_cfg = load_config(slim_cfg)
        self.teacher_cfg = slim_cfg
        self.loss_cfg = slim_cfg
        tea_pretrain = cfg['pretrain_weights']

        self.teacher_model = create(self.teacher_cfg.architecture)
        self.teacher_model.eval()

        for param in self.teacher_model.parameters():
            param.trainable = False

        if 'pretrain_weights' in cfg and stu_pretrain:
            if self.is_inherit and 'pretrain_weights' in self.teacher_cfg and self.teacher_cfg.pretrain_weights:
                load_pretrain_weight(self.student_model,
                                     self.teacher_cfg.pretrain_weights)
                logger.debug(
                    "Inheriting! loading teacher weights to student model!")

            load_pretrain_weight(self.student_model, stu_pretrain)

        if 'pretrain_weights' in self.teacher_cfg and self.teacher_cfg.pretrain_weights:
            load_pretrain_weight(self.teacher_model,
                                 self.teacher_cfg.pretrain_weights)

        self.fgd_loss_dic = self.build_loss(
            self.loss_cfg.distill_loss,
            name_list=self.loss_cfg['distill_loss_name'])

    def build_loss(self,
                   cfg,
                   name_list=[
                       'neck_f_4', 'neck_f_3', 'neck_f_2', 'neck_f_1',
                       'neck_f_0'
                   ]):
        loss_func = dict()
        for idx, k in enumerate(name_list):
            loss_func[k] = create(cfg)
        return loss_func

    def forward(self, inputs):
        if self.training:
            s_body_feats = self.student_model.backbone(inputs)
            s_neck_feats = self.student_model.neck(s_body_feats)

            with paddle.no_grad():
                t_body_feats = self.teacher_model.backbone(inputs)
                t_neck_feats = self.teacher_model.neck(t_body_feats)

            loss_dict = {}
            for idx, k in enumerate(self.fgd_loss_dic):
                loss_dict[k] = self.fgd_loss_dic[k](s_neck_feats[idx],
                                                    t_neck_feats[idx], inputs)
            if self.arch == "RetinaNet":
                loss = self.student_model.head(s_neck_feats, inputs)
            elif self.arch == "PicoDet":
                head_outs = self.student_model.head(
                    s_neck_feats, self.student_model.export_post_process)
                loss_gfl = self.student_model.head.get_loss(head_outs, inputs)
                total_loss = paddle.add_n(list(loss_gfl.values()))
                loss = {}
                loss.update(loss_gfl)
                loss.update({'loss': total_loss})
            else:
                raise ValueError(f"Unsupported model {self.arch}")
            for k in loss_dict:
                loss['loss'] += loss_dict[k]
                loss[k] = loss_dict[k]
            return loss
        else:
            body_feats = self.student_model.backbone(inputs)
            neck_feats = self.student_model.neck(body_feats)
            head_outs = self.student_model.head(neck_feats)
            if self.arch == "RetinaNet":
                bbox, bbox_num = self.student_model.head.post_process(
                    head_outs, inputs['im_shape'], inputs['scale_factor'])
                return {'bbox': bbox, 'bbox_num': bbox_num}
            elif self.arch == "PicoDet":
                head_outs = self.student_model.head(
                    neck_feats, self.student_model.export_post_process)
                scale_factor = inputs['scale_factor']
                bboxes, bbox_num = self.student_model.head.post_process(
                    head_outs,
                    scale_factor,
                    export_nms=self.student_model.export_nms)
                return {'bbox': bboxes, 'bbox_num': bbox_num}
            else:
                raise ValueError(f"Unsupported model {self.arch}")


class CWDDistillModel(nn.Layer):
    """                                                                                                                                                    
    Build CWD distill model.                                                                                                                               
    Args:                                                                                                                                                  
        cfg: The student config.                                                                                                                           
        slim_cfg: The teacher and distill config.                                                                                                          
    """

    def __init__(self, cfg, slim_cfg):
        super(CWDDistillModel, self).__init__()

        self.is_inherit = False
        # build student model before load slim config                                                                                                      
        self.student_model = create(cfg.architecture)
        self.arch = cfg.architecture
        stu_pretrain = cfg['pretrain_weights']
        slim_cfg = load_config(slim_cfg)
        self.teacher_cfg = slim_cfg
        self.loss_cfg = slim_cfg
        tea_pretrain = cfg['pretrain_weights']

        self.teacher_model = create(self.teacher_cfg.architecture)
        self.teacher_model.eval()

        for param in self.teacher_model.parameters():
            param.trainable = False
        if 'pretrain_weights' in cfg and stu_pretrain:
            if self.is_inherit and 'pretrain_weights' in self.teacher_cfg and self.teacher_cfg.pretrain_weights:
                load_pretrain_weight(self.student_model,
                                     self.teacher_cfg.pretrain_weights)
                logger.debug(
                    "Inheriting! loading teacher weights to student model!")

            load_pretrain_weight(self.student_model, stu_pretrain)

        if 'pretrain_weights' in self.teacher_cfg and self.teacher_cfg.pretrain_weights:
            load_pretrain_weight(self.teacher_model,
                                 self.teacher_cfg.pretrain_weights)

        self.loss_dic = self.build_loss(
            self.loss_cfg.distill_loss,
            name_list=self.loss_cfg['distill_loss_name'])

    def build_loss(self,
                   cfg,
                   name_list=[
                       'neck_f_4', 'neck_f_3', 'neck_f_2', 'neck_f_1',
                       'neck_f_0'
                   ]):
        loss_func = dict()
        for idx, k in enumerate(name_list):
            loss_func[k] = create(cfg)
        return loss_func

    def get_loss_retinanet(self, stu_fea_list, tea_fea_list, inputs):
        loss = self.student_model.head(stu_fea_list, inputs)
        distill_loss = {}
        # cwd kd loss
        for idx, k in enumerate(self.loss_dic):
            distill_loss[k] = self.loss_dic[k](stu_fea_list[idx],
                                               tea_fea_list[idx])

            loss['loss'] += distill_loss[k]
            loss[k] = distill_loss[k]
        return loss

    def get_loss_gfl(self, stu_fea_list, tea_fea_list, inputs):
        loss = {}
        head_outs = self.student_model.head(stu_fea_list)
        loss_gfl = self.student_model.head.get_loss(head_outs, inputs)
        loss.update(loss_gfl)
        total_loss = paddle.add_n(list(loss.values()))
        loss.update({'loss': total_loss})
        # cwd kd loss
        feat_loss = {}
        loss_dict = {}

        s_cls_feat, t_cls_feat = [], []
        for s_neck_f, t_neck_f in zip(stu_fea_list, tea_fea_list):
            conv_cls_feat, _ = self.student_model.head.conv_feat(s_neck_f)
            cls_score = self.student_model.head.gfl_head_cls(conv_cls_feat)
            t_conv_cls_feat, _ = self.teacher_model.head.conv_feat(t_neck_f)
            t_cls_score = self.teacher_model.head.gfl_head_cls(t_conv_cls_feat)
            s_cls_feat.append(cls_score)
            t_cls_feat.append(t_cls_score)

        for idx, k in enumerate(self.loss_dic):
            loss_dict[k] = self.loss_dic[k](s_cls_feat[idx], t_cls_feat[idx])
            feat_loss[f"neck_f_{idx}"] = self.loss_dic[k](stu_fea_list[idx],
                                                          tea_fea_list[idx])

        for k in feat_loss:
            loss['loss'] += feat_loss[k]
            loss[k] = feat_loss[k]

        for k in loss_dict:
            loss['loss'] += loss_dict[k]
            loss[k] = loss_dict[k]
        return loss

    def forward(self, inputs):
        if self.training:
            s_body_feats = self.student_model.backbone(inputs)
            s_neck_feats = self.student_model.neck(s_body_feats)

            with paddle.no_grad():
                t_body_feats = self.teacher_model.backbone(inputs)
                t_neck_feats = self.teacher_model.neck(t_body_feats)

            if self.arch == "RetinaNet":
                loss = self.get_loss_retinanet(s_neck_feats, t_neck_feats,
                                               inputs)
            elif self.arch == "GFL":
                loss = self.get_loss_gfl(s_neck_feats, t_neck_feats, inputs)
            else:
                raise ValueError(f"unsupported arch {self.arch}")
            return loss
        else:
            body_feats = self.student_model.backbone(inputs)
            neck_feats = self.student_model.neck(body_feats)
            head_outs = self.student_model.head(neck_feats)
            if self.arch == "RetinaNet":
                bbox, bbox_num = self.student_model.head.post_process(
                    head_outs, inputs['im_shape'], inputs['scale_factor'])
                return {'bbox': bbox, 'bbox_num': bbox_num}
            elif self.arch == "GFL":
                bbox_pred, bbox_num = head_outs
                output = {'bbox': bbox_pred, 'bbox_num': bbox_num}
                return output
            else:
                raise ValueError(f"unsupported arch {self.arch}")


@register
class ChannelWiseDivergence(nn.Layer):
    def __init__(
            self,
            student_channels,
            teacher_channels,
            name,
            tau=1.0,
            weight=1.0, ):
        super(ChannelWiseDivergence, self).__init__()
        self.tau = tau
        self.loss_weight = weight

        if student_channels != teacher_channels:
            self.align = nn.Conv2D(
                student_channels,
                teacher_channels,
                kernel_size=1,
                stride=1,
                padding=0)
        else:
            self.align = None

    def distill_softmax(self, x, t):
        _, _, w, h = paddle.shape(x)
        x = paddle.reshape(x, [-1, w * h])
        x /= t
        return F.softmax(x, axis=1)

    def forward(self, preds_s, preds_t):
        assert preds_s.shape[-2:] == preds_t.shape[
            -2:], 'the output dim of teacher and student differ'
        N, C, W, H = preds_s.shape
        eps = 1e-5
        if self.align is not None:
            preds_s = self.align(preds_s)

        softmax_pred_s = self.distill_softmax(preds_s, self.tau)
        softmax_pred_t = self.distill_softmax(preds_t, self.tau)

        loss = paddle.sum(-softmax_pred_t * paddle.log(eps + softmax_pred_s) +
                          softmax_pred_t * paddle.log(eps + softmax_pred_t))
        return self.loss_weight * loss / (C * N)


@register
class DistillYOLOv3Loss(nn.Layer):
    def __init__(self, weight=1000):
        super(DistillYOLOv3Loss, self).__init__()
        self.weight = weight

    def obj_weighted_reg(self, sx, sy, sw, sh, tx, ty, tw, th, tobj):
        loss_x = ops.sigmoid_cross_entropy_with_logits(sx, F.sigmoid(tx))
        loss_y = ops.sigmoid_cross_entropy_with_logits(sy, F.sigmoid(ty))
        loss_w = paddle.abs(sw - tw)
        loss_h = paddle.abs(sh - th)
        loss = paddle.add_n([loss_x, loss_y, loss_w, loss_h])
        weighted_loss = paddle.mean(loss * F.sigmoid(tobj))
        return weighted_loss

    def obj_weighted_cls(self, scls, tcls, tobj):
        loss = ops.sigmoid_cross_entropy_with_logits(scls, F.sigmoid(tcls))
        weighted_loss = paddle.mean(paddle.multiply(loss, F.sigmoid(tobj)))
        return weighted_loss

    def obj_loss(self, sobj, tobj):
        obj_mask = paddle.cast(tobj > 0., dtype="float32")
        obj_mask.stop_gradient = True
        loss = paddle.mean(
            ops.sigmoid_cross_entropy_with_logits(sobj, obj_mask))
        return loss

    def forward(self, teacher_model, student_model):
        teacher_distill_pairs = teacher_model.yolo_head.loss.distill_pairs
        student_distill_pairs = student_model.yolo_head.loss.distill_pairs
        distill_reg_loss, distill_cls_loss, distill_obj_loss = [], [], []
        for s_pair, t_pair in zip(student_distill_pairs, teacher_distill_pairs):
            distill_reg_loss.append(
                self.obj_weighted_reg(s_pair[0], s_pair[1], s_pair[2], s_pair[
                    3], t_pair[0], t_pair[1], t_pair[2], t_pair[3], t_pair[4]))
            distill_cls_loss.append(
                self.obj_weighted_cls(s_pair[5], t_pair[5], t_pair[4]))
            distill_obj_loss.append(self.obj_loss(s_pair[4], t_pair[4]))
        distill_reg_loss = paddle.add_n(distill_reg_loss)
        distill_cls_loss = paddle.add_n(distill_cls_loss)
        distill_obj_loss = paddle.add_n(distill_obj_loss)
        loss = (distill_reg_loss + distill_cls_loss + distill_obj_loss
                ) * self.weight
        return loss


def parameter_init(mode="kaiming", value=0.):
    if mode == "kaiming":
        weight_attr = paddle.nn.initializer.KaimingUniform()
    elif mode == "constant":
        weight_attr = paddle.nn.initializer.Constant(value=value)
    else:
        weight_attr = paddle.nn.initializer.KaimingUniform()

    weight_init = ParamAttr(initializer=weight_attr)
    return weight_init


@register
class FGDFeatureLoss(nn.Layer):
    """
    The code is reference from https://github.com/yzd-v/FGD/blob/master/mmdet/distillation/losses/fgd.py
    Paddle version of `Focal and Global Knowledge Distillation for Detectors`
   
    Args:
        student_channels(int): The number of channels in the student's FPN feature map. Default to 256.
        teacher_channels(int): The number of channels in the teacher's FPN feature map. Default to 256.
        temp (float, optional): The temperature coefficient. Defaults to 0.5.
        alpha_fgd (float, optional): The weight of fg_loss. Defaults to 0.001
        beta_fgd (float, optional): The weight of bg_loss. Defaults to 0.0005
        gamma_fgd (float, optional): The weight of mask_loss. Defaults to 0.001
        lambda_fgd (float, optional): The weight of relation_loss. Defaults to 0.000005
    """

    def __init__(self,
                 student_channels=256,
                 teacher_channels=256,
                 temp=0.5,
                 alpha_fgd=0.001,
                 beta_fgd=0.0005,
                 gamma_fgd=0.001,
                 lambda_fgd=0.000005):
        super(FGDFeatureLoss, self).__init__()
        self.temp = temp
        self.alpha_fgd = alpha_fgd
        self.beta_fgd = beta_fgd
        self.gamma_fgd = gamma_fgd
        self.lambda_fgd = lambda_fgd

        kaiming_init = parameter_init("kaiming")
        zeros_init = parameter_init("constant", 0.0)

        if student_channels != teacher_channels:
            self.align = nn.Conv2D(
                student_channels,
                teacher_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                weight_attr=kaiming_init)
            student_channels = teacher_channels
        else:
            self.align = None

        self.conv_mask_s = nn.Conv2D(
            student_channels, 1, kernel_size=1, weight_attr=kaiming_init)
        self.conv_mask_t = nn.Conv2D(
            teacher_channels, 1, kernel_size=1, weight_attr=kaiming_init)

        self.stu_conv_block = nn.Sequential(
            nn.Conv2D(
                student_channels,
                student_channels // 2,
                kernel_size=1,
                weight_attr=zeros_init),
            nn.LayerNorm([student_channels // 2, 1, 1]),
            nn.ReLU(),
            nn.Conv2D(
                student_channels // 2,
                student_channels,
                kernel_size=1,
                weight_attr=zeros_init))
        self.tea_conv_block = nn.Sequential(
            nn.Conv2D(
                teacher_channels,
                teacher_channels // 2,
                kernel_size=1,
                weight_attr=zeros_init),
            nn.LayerNorm([teacher_channels // 2, 1, 1]),
            nn.ReLU(),
            nn.Conv2D(
                teacher_channels // 2,
                teacher_channels,
                kernel_size=1,
                weight_attr=zeros_init))

    def spatial_channel_attention(self, x, t=0.5):
        shape = paddle.shape(x)
        N, C, H, W = shape

        _f = paddle.abs(x)
        spatial_map = paddle.reshape(
            paddle.mean(
                _f, axis=1, keepdim=True) / t, [N, -1])
        spatial_map = F.softmax(spatial_map, axis=1, dtype="float32") * H * W
        spatial_att = paddle.reshape(spatial_map, [N, H, W])

        channel_map = paddle.mean(
            paddle.mean(
                _f, axis=2, keepdim=False), axis=2, keepdim=False)
        channel_att = F.softmax(channel_map / t, axis=1, dtype="float32") * C
        return [spatial_att, channel_att]

    def spatial_pool(self, x, mode="teacher"):
        batch, channel, width, height = x.shape
        x_copy = x
        x_copy = paddle.reshape(x_copy, [batch, channel, height * width])
        x_copy = x_copy.unsqueeze(1)
        if mode.lower() == "student":
            context_mask = self.conv_mask_s(x)
        else:
            context_mask = self.conv_mask_t(x)

        context_mask = paddle.reshape(context_mask, [batch, 1, height * width])
        context_mask = F.softmax(context_mask, axis=2)
        context_mask = context_mask.unsqueeze(-1)
        context = paddle.matmul(x_copy, context_mask)
        context = paddle.reshape(context, [batch, channel, 1, 1])

        return context

    def mask_loss(self, stu_channel_att, tea_channel_att, stu_spatial_att,
                  tea_spatial_att):
        def _func(a, b):
            return paddle.sum(paddle.abs(a - b)) / len(a)

        mask_loss = _func(stu_channel_att, tea_channel_att) + _func(
            stu_spatial_att, tea_spatial_att)

        return mask_loss

    def feature_loss(self, stu_feature, tea_feature, Mask_fg, Mask_bg,
                     tea_channel_att, tea_spatial_att):

        Mask_fg = Mask_fg.unsqueeze(axis=1)
        Mask_bg = Mask_bg.unsqueeze(axis=1)

        tea_channel_att = tea_channel_att.unsqueeze(axis=-1)
        tea_channel_att = tea_channel_att.unsqueeze(axis=-1)

        tea_spatial_att = tea_spatial_att.unsqueeze(axis=1)

        fea_t = paddle.multiply(tea_feature, paddle.sqrt(tea_spatial_att))
        fea_t = paddle.multiply(fea_t, paddle.sqrt(tea_channel_att))
        fg_fea_t = paddle.multiply(fea_t, paddle.sqrt(Mask_fg))
        bg_fea_t = paddle.multiply(fea_t, paddle.sqrt(Mask_bg))

        fea_s = paddle.multiply(stu_feature, paddle.sqrt(tea_spatial_att))
        fea_s = paddle.multiply(fea_s, paddle.sqrt(tea_channel_att))
        fg_fea_s = paddle.multiply(fea_s, paddle.sqrt(Mask_fg))
        bg_fea_s = paddle.multiply(fea_s, paddle.sqrt(Mask_bg))

        fg_loss = F.mse_loss(fg_fea_s, fg_fea_t, reduction="sum") / len(Mask_fg)
        bg_loss = F.mse_loss(bg_fea_s, bg_fea_t, reduction="sum") / len(Mask_bg)

        return fg_loss, bg_loss

    def relation_loss(self, stu_feature, tea_feature):
        context_s = self.spatial_pool(stu_feature, "student")
        context_t = self.spatial_pool(tea_feature, "teacher")

        out_s = stu_feature + self.stu_conv_block(context_s)
        out_t = tea_feature + self.tea_conv_block(context_t)

        rela_loss = F.mse_loss(out_s, out_t, reduction="sum") / len(out_s)

        return rela_loss

    def mask_value(self, mask, xl, xr, yl, yr, value):
        mask[xl:xr, yl:yr] = paddle.maximum(mask[xl:xr, yl:yr], value)
        return mask

    def forward(self, stu_feature, tea_feature, inputs):
        """Forward function.
        Args:
            stu_feature(Tensor): Bs*C*H*W, student's feature map
            tea_feature(Tensor): Bs*C*H*W, teacher's feature map
            inputs: The inputs with gt bbox and input shape info.
        """
        assert stu_feature.shape[-2:] == stu_feature.shape[-2:], \
            f'The shape of Student feature {stu_feature.shape} and Teacher feature {tea_feature.shape} should be the same.'
        assert "gt_bbox" in inputs.keys() and "im_shape" in inputs.keys(
        ), "ERROR! FGDFeatureLoss need gt_bbox and im_shape as inputs."
        gt_bboxes = inputs['gt_bbox']
        ins_shape = [
            inputs['im_shape'][i] for i in range(inputs['im_shape'].shape[0])
        ]

        index_gt = []
        for i in range(len(gt_bboxes)):
            if gt_bboxes[i].size > 2:
                index_gt.append(i)
        # only distill feature with labeled GTbox
        if len(index_gt) != len(gt_bboxes):
            index_gt_t = paddle.to_tensor(index_gt)
            preds_S = paddle.index_select(preds_S, index_gt_t)
            preds_T = paddle.index_select(preds_T, index_gt_t)

            ins_shape = [ins_shape[c] for c in index_gt]
            gt_bboxes = [gt_bboxes[c] for c in index_gt]
            assert len(gt_bboxes) == preds_T.shape[
                0], f"The number of selected GT box [{len(gt_bboxes)}] should be same with first dim of input tensor [{preds_T.shape[0]}]."

        if self.align is not None:
            stu_feature = self.align(stu_feature)

        N, C, H, W = stu_feature.shape

        tea_spatial_att, tea_channel_att = self.spatial_channel_attention(
            tea_feature, self.temp)
        stu_spatial_att, stu_channel_att = self.spatial_channel_attention(
            stu_feature, self.temp)

        Mask_fg = paddle.zeros(tea_spatial_att.shape)
        Mask_bg = paddle.ones_like(tea_spatial_att)
        one_tmp = paddle.ones([*tea_spatial_att.shape[1:]])
        zero_tmp = paddle.zeros([*tea_spatial_att.shape[1:]])
        Mask_fg.stop_gradient = True
        Mask_bg.stop_gradient = True
        one_tmp.stop_gradient = True
        zero_tmp.stop_gradient = True

        wmin, wmax, hmin, hmax, area = [], [], [], [], []

        for i in range(N):
            tmp_box = paddle.ones_like(gt_bboxes[i])
            tmp_box.stop_gradient = True
            tmp_box[:, 0] = gt_bboxes[i][:, 0] / ins_shape[i][1] * W
            tmp_box[:, 2] = gt_bboxes[i][:, 2] / ins_shape[i][1] * W
            tmp_box[:, 1] = gt_bboxes[i][:, 1] / ins_shape[i][0] * H
            tmp_box[:, 3] = gt_bboxes[i][:, 3] / ins_shape[i][0] * H

            zero = paddle.zeros_like(tmp_box[:, 0], dtype="int32")
            ones = paddle.ones_like(tmp_box[:, 2], dtype="int32")
            zero.stop_gradient = True
            ones.stop_gradient = True

            wmin.append(
                paddle.cast(paddle.floor(tmp_box[:, 0]), "int32").maximum(zero))
            wmax.append(paddle.cast(paddle.ceil(tmp_box[:, 2]), "int32"))
            hmin.append(
                paddle.cast(paddle.floor(tmp_box[:, 1]), "int32").maximum(zero))
            hmax.append(paddle.cast(paddle.ceil(tmp_box[:, 3]), "int32"))

            area_recip = 1.0 / (
                hmax[i].reshape([1, -1]) + 1 - hmin[i].reshape([1, -1])) / (
                    wmax[i].reshape([1, -1]) + 1 - wmin[i].reshape([1, -1]))

            for j in range(len(gt_bboxes[i])):
                Mask_fg[i] = self.mask_value(Mask_fg[i], hmin[i][j],
                                             hmax[i][j] + 1, wmin[i][j],
                                             wmax[i][j] + 1, area_recip[0][j])

            Mask_bg[i] = paddle.where(Mask_fg[i] > zero_tmp, zero_tmp, one_tmp)

            if paddle.sum(Mask_bg[i]):
                Mask_bg[i] /= paddle.sum(Mask_bg[i])

        fg_loss, bg_loss = self.feature_loss(stu_feature, tea_feature, Mask_fg,
                                             Mask_bg, tea_channel_att,
                                             tea_spatial_att)
        mask_loss = self.mask_loss(stu_channel_att, tea_channel_att,
                                   stu_spatial_att, tea_spatial_att)
        rela_loss = self.relation_loss(stu_feature, tea_feature)

        loss = self.alpha_fgd * fg_loss + self.beta_fgd * bg_loss \
               + self.gamma_fgd * mask_loss + self.lambda_fgd * rela_loss

        return loss


class LDDistillModel(nn.Layer):
    def __init__(self, cfg, slim_cfg):
        super(LDDistillModel, self).__init__()
        self.student_model = create(cfg.architecture)
        logger.debug('Load student model pretrain_weights:{}'.format(
            cfg.pretrain_weights))
        load_pretrain_weight(self.student_model, cfg.pretrain_weights)

        slim_cfg = load_config(slim_cfg)  #rewrite student cfg
        self.teacher_model = create(slim_cfg.architecture)
        logger.debug('Load teacher model pretrain_weights:{}'.format(
            slim_cfg.pretrain_weights))
        load_pretrain_weight(self.teacher_model, slim_cfg.pretrain_weights)

        for param in self.teacher_model.parameters():
            param.trainable = False

    def parameters(self):
        return self.student_model.parameters()

    def forward(self, inputs):
        if self.training:

            with paddle.no_grad():
                t_body_feats = self.teacher_model.backbone(inputs)
                t_neck_feats = self.teacher_model.neck(t_body_feats)
                t_head_outs = self.teacher_model.head(t_neck_feats)

            #student_loss = self.student_model(inputs)
            s_body_feats = self.student_model.backbone(inputs)
            s_neck_feats = self.student_model.neck(s_body_feats)
            s_head_outs = self.student_model.head(s_neck_feats)

            soft_label_list = t_head_outs[0]
            soft_targets_list = t_head_outs[1]
            student_loss = self.student_model.head.get_loss(
                s_head_outs, inputs, soft_label_list, soft_targets_list)
            total_loss = paddle.add_n(list(student_loss.values()))
            student_loss['loss'] = total_loss
            return student_loss
        else:
            return self.student_model(inputs)


@register
class KnowledgeDistillationKLDivLoss(nn.Layer):
    """Loss function for knowledge distilling using KL divergence.

    Args:
        reduction (str): Options are `'none'`, `'mean'` and `'sum'`.
        loss_weight (float): Loss weight of current loss.
        T (int): Temperature for distillation.
    """

    def __init__(self, reduction='mean', loss_weight=1.0, T=10):
        super(KnowledgeDistillationKLDivLoss, self).__init__()
        assert reduction in ('none', 'mean', 'sum')
        assert T >= 1
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.T = T

    def knowledge_distillation_kl_div_loss(self,
                                           pred,
                                           soft_label,
                                           T,
                                           detach_target=True):
        r"""Loss function for knowledge distilling using KL divergence.

        Args:
            pred (Tensor): Predicted logits with shape (N, n + 1).
            soft_label (Tensor): Target logits with shape (N, N + 1).
            T (int): Temperature for distillation.
            detach_target (bool): Remove soft_label from automatic differentiation

        Returns:
            torch.Tensor: Loss tensor with shape (N,).
        """

        assert pred.shape == soft_label.shape
        target = F.softmax(soft_label / T, axis=1)
        if detach_target:
            target = target.detach()

        kd_loss = F.kl_div(
            F.log_softmax(
                pred / T, axis=1), target, reduction='none').mean(1) * (T * T)

        return kd_loss

    def forward(self,
                pred,
                soft_label,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (Tensor): Predicted logits with shape (N, n + 1).
            soft_label (Tensor): Target logits with shape (N, N + 1).
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')

        reduction = (reduction_override
                     if reduction_override else self.reduction)

        loss_kd_out = self.knowledge_distillation_kl_div_loss(
            pred, soft_label, T=self.T)

        if weight is not None:
            loss_kd_out = weight * loss_kd_out

        if avg_factor is None:
            if reduction == 'none':
                loss = loss_kd_out
            elif reduction == 'mean':
                loss = loss_kd_out.mean()
            elif reduction == 'sum':
                loss = loss_kd_out.sum()
        else:
            # if reduction is mean, then average the loss by avg_factor
            if reduction == 'mean':
                loss = loss_kd_out.sum() / avg_factor
            # if reduction is 'none', then do nothing, otherwise raise an error
            elif reduction != 'none':
                raise ValueError(
                    'avg_factor can not be used with reduction="sum"')

        loss_kd = self.loss_weight * loss

        return loss_kd
