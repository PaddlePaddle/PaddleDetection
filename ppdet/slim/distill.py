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
                head_outs = self.student_model.head(s_neck_feats, self.student_model.export_post_process)
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
                head_outs = self.student_model.head(neck_feats, self.student_model.export_post_process)
                scale_factor = inputs['scale_factor']
                bboxes, bbox_num = self.student_model.head.post_process(
                               head_outs, scale_factor, export_nms=self.student_model.export_nms)
                return {'bbox': bboxes, 'bbox_num': bbox_num}
            else:
                raise ValueError(f"Unsupported model {self.arch}")


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
    The API is reference from https://github.com/yzd-v/FGD/blob/master/mmdet/distillation/losses/fgd.py
    Paddle version of `Focal and Global Knowledge Distillation for Detectors`
   
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map. 
        temp (float, optional): Temperature coefficient. Defaults to 0.5.
        name (str): the loss name of the layer
        alpha_fgd (float, optional): Weight of fg_loss. Defaults to 0.001
        beta_fgd (float, optional): Weight of bg_loss. Defaults to 0.0005
        gamma_fgd (float, optional): Weight of mask_loss. Defaults to 0.001
        lambda_fgd (float, optional): Weight of relation_loss. Defaults to 0.000005
    """


    def __init__(self,
                 student_channels,
                 teacher_channels,
                 name=None,
                 temp=0.5,
                 alpha_fgd=0.001,
                 beta_fgd=0.0005,
                 gamma_fgd=0.001,
                 lambda_fgd=0.000005,
                 ):
        super(FGDFeatureLoss, self).__init__()
        self.temp = temp
        self.alpha_fgd = alpha_fgd
        self.beta_fgd = beta_fgd
        self.gamma_fgd = gamma_fgd
        self.lambda_fgd = lambda_fgd

        kaiming_init = parameter_init("kaiming")
        zeros_init = parameter_init("constant", 0.0)

        if student_channels != teacher_channels:
            self.align = nn.Conv2d(
                student_channels, 
                teacher_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                weight_attr=kaiming_init)
        else:
            self.align = None
        
        self.conv_mask_s = nn.Conv2D(
            teacher_channels, 1, kernel_size=1, weight_attr=kaiming_init)
        self.conv_mask_t = nn.Conv2D(
            teacher_channels, 1, kernel_size=1, weight_attr=kaiming_init)
        self.channel_add_conv_s = nn.Sequential(
            nn.Conv2D(
                teacher_channels,
                teacher_channels//2,
                kernel_size=1,
                weight_attr=zeros_init),
            nn.LayerNorm([teacher_channels//2, 1, 1]),
            nn.ReLU(),  
            nn.Conv2D(
                    teacher_channels//2,
                    teacher_channels,
                    kernel_size=1,
                    weight_attr=zeros_init))
        self.channel_add_conv_t = nn.Sequential(
            nn.Conv2D(
                teacher_channels,
                teacher_channels//2,
                kernel_size=1,
                weight_attr=zeros_init),
            nn.LayerNorm([teacher_channels//2, 1, 1]),
            nn.ReLU(), 
            nn.Conv2D(
                teacher_channels//2,
                teacher_channels,
                kernel_size=1,
                weight_attr=zeros_init))

    def gc_block(self, feature, t=0.5):
        """
        """
        shape = paddle.shape(feature)
        N, C, H, W = shape

        _f = paddle.abs(feature)
        s_map = paddle.reshape(paddle.mean(_f, axis=1, keepdim=True)/t, [N, -1])
        s_map = F.softmax(s_map, axis=1, dtype="float32") * H * W
        s_attention = paddle.reshape(s_map, [N, H, W])

        c_map = paddle.mean(paddle.mean(_f, axis=2, keepdim=False), axis=2, keepdim=False)
        c_attention = F.softmax(c_map/t, axis=1, dtype="float32")  * C
        return s_attention, c_attention


    def spatial_pool(self, x, in_type):
        batch, channel, width, height = x.shape
        input_x = x
        # [N, C, H * W]
        input_x = paddle.reshape(input_x, [batch, channel, height * width])
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        if in_type == 0:
            context_mask = self.conv_mask_s(x)
        else:
            context_mask = self.conv_mask_t(x)
        # [N, 1, H * W]
        context_mask = paddle.reshape(context_mask, [batch, 1, height * width])
        # [N, 1, H * W]
        context_mask = F.softmax(context_mask, axis=2)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N, 1, C, 1]
        context = paddle.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = paddle.reshape(context, [batch, channel, 1, 1])

        return context

    def get_mask_loss(self, C_s, C_t, S_s, S_t):
        mask_loss = paddle.sum(paddle.abs((C_s-C_t)))/len(C_s) + paddle.sum(paddle.abs((S_s-S_t)))/len(S_s)
        return mask_loss


    def get_fea_loss(self, preds_S, preds_T, Mask_fg, Mask_bg, C_s, C_t, S_s, S_t):        
        Mask_fg = Mask_fg.unsqueeze(axis=1)
        Mask_bg = Mask_bg.unsqueeze(axis=1)

        C_t = C_t.unsqueeze(axis=-1)
        C_t = C_t.unsqueeze(axis=-1)

        S_t = S_t.unsqueeze(axis=1)

        fea_t= paddle.multiply(preds_T, paddle.sqrt(S_t))
        fea_t = paddle.multiply(fea_t, paddle.sqrt(C_t))
        fg_fea_t = paddle.multiply(fea_t, paddle.sqrt(Mask_fg))
        bg_fea_t = paddle.multiply(fea_t, paddle.sqrt(Mask_bg))

        fea_s = paddle.multiply(preds_S, paddle.sqrt(S_t))
        fea_s = paddle.multiply(fea_s, paddle.sqrt(C_t))
        fg_fea_s = paddle.multiply(fea_s, paddle.sqrt(Mask_fg))
        bg_fea_s = paddle.multiply(fea_s, paddle.sqrt(Mask_bg))

        fg_loss = F.mse_loss(fg_fea_s, fg_fea_t, reduction="sum")/len(Mask_fg)
        bg_loss = F.mse_loss(bg_fea_s, bg_fea_t, reduction="sum")/len(Mask_bg)

        return fg_loss, bg_loss

    def get_rela_loss(self, preds_S, preds_T):
        context_s = self.spatial_pool(preds_S, 0)
        context_t = self.spatial_pool(preds_T, 1)

        out_s = preds_S
        out_t = preds_T

        channel_add_s = self.channel_add_conv_s(context_s)
        out_s = out_s + channel_add_s

        channel_add_t = self.channel_add_conv_t(context_t)
        out_t = out_t + channel_add_t

        rela_loss = F.mse_loss(out_s, out_t, reduction="sum")/len(out_s)
        
        return rela_loss
    
    def mask_value(self, mask, xl, xr, yl, yr, value):
        mask[xl:xr, yl:yr] = paddle.maximum(mask[xl:xr, yl:yr], value)
        return mask


    def forward(self, 
                preds_S,
                preds_T,
                inputs):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
            inputs: The inputs with gt bbox and input shape info.
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:], \
            f'The shape of Student feature {preds_S.shape} and Teacher feature {preds_T.shape} should be the same.'
        gt_bboxes = inputs['gt_bbox']
        assert len(gt_bboxes) == preds_S.shape[0], "error"

        # select index 
        index_gt = []
        for i in range(len(gt_bboxes)):
            if gt_bboxes[i].size > 2:
                index_gt.append(i)
        index_gt_t = paddle.to_tensor(index_gt) # to tensor
        preds_S = paddle.index_select(preds_S, index_gt_t)
        preds_T = paddle.index_select(preds_T, index_gt_t)
        assert preds_S.shape == preds_T.shape, "error"

        img_metas_tmp = [{'img_shape': inputs['im_shape'][i]} for i in range(inputs['im_shape'].shape[0])]
        img_metas = [img_metas_tmp[c] for c in index_gt]
        gt_bboxes = [gt_bboxes[c] for c in index_gt]
        assert len(gt_bboxes) == len(img_metas), "error"
        
        assert len(gt_bboxes) == preds_T.shape[0], "error"

        if self.align is not None:
            preds_S = self.align(preds_S)
        
        N,C,H,W = preds_S.shape

        S_attention_t, C_attention_t = self.gc_block(preds_T, self.temp)
        S_attention_s, C_attention_s = self.gc_block(preds_S, self.temp)

        Mask_fg = paddle.zeros(S_attention_t.shape)
        Mask_bg = paddle.ones_like(S_attention_t)
        one_tmp = paddle.ones([*S_attention_t.shape[1:]])
        zero_tmp = paddle.zeros([*S_attention_t.shape[1:]])
        wmin,wmax,hmin,hmax, area = [],[],[],[], []
        for i in range(N):
            new_boxxes = paddle.ones_like(gt_bboxes[i])
            new_boxxes[:, 0] = gt_bboxes[i][:, 0]/img_metas[i]['img_shape'][1]*W
            new_boxxes[:, 2] = gt_bboxes[i][:, 2]/img_metas[i]['img_shape'][1]*W
            new_boxxes[:, 1] = gt_bboxes[i][:, 1]/img_metas[i]['img_shape'][0]*H
            new_boxxes[:, 3] = gt_bboxes[i][:, 3]/img_metas[i]['img_shape'][0]*H
            zero = paddle.zeros_like(new_boxxes[:, 0], dtype="int32")
            ones = paddle.ones_like(new_boxxes[:, 2], dtype="int32")
            wmin.append(paddle.cast(paddle.floor(new_boxxes[:, 0]), "int32").maximum(zero))
            wmax.append(paddle.cast(paddle.ceil(new_boxxes[:, 2]), "int32"))
            hmin.append(paddle.cast(paddle.floor(new_boxxes[:, 1]), "int32").maximum(zero))
            hmax.append(paddle.cast(paddle.ceil(new_boxxes[:, 3]), "int32"))
            
            area = 1.0/(hmax[i].reshape([1,-1]) + 1 - hmin[i].reshape([1,-1])) /(wmax[i].reshape([1,-1])+1-wmin[i].reshape([1,-1]))
            for j in range(len(gt_bboxes[i])):
                Mask_fg[i] = self.mask_value(Mask_fg[i], hmin[i][j], hmax[i][j]+1, wmin[i][j], wmax[i][j]+1, area[0][j])  
            Mask_bg[i] = paddle.where(Mask_fg[i] > zero_tmp, zero_tmp, one_tmp)

            if paddle.sum(Mask_bg[i]):
                Mask_bg[i] /= paddle.sum(Mask_bg[i])

        fg_loss, bg_loss = self.get_fea_loss(preds_S, preds_T, Mask_fg, Mask_bg, 
                           C_attention_s, C_attention_t, S_attention_s, S_attention_t)
        mask_loss = self.get_mask_loss(C_attention_s, C_attention_t, S_attention_s, S_attention_t)
        rela_loss = self.get_rela_loss(preds_S, preds_T)


        loss = self.alpha_fgd * fg_loss + self.beta_fgd * bg_loss \
               + self.gamma_fgd * mask_loss + self.lambda_fgd * rela_loss
        
        return loss