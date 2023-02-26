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
"""
This code is based on https://github.com/PeizeSun/SparseR-CNN/blob/main/projects/SparseRCNN/sparsercnn/head.py
Ths copyright of PeizeSun/SparseR-CNN is as follows:
MIT License [see LICENSE for details]
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random
import copy
from collections import namedtuple
import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.ops import nms as batched_nms

from ppdet.core.workspace import register
from ppdet.modeling.heads.roi_extractor import RoIAlign
from ppdet.modeling.bbox_utils import delta2bbox
from .. import initializer as init
from .sparsercnn_head import DynamicConv

_DEFAULT_SCALE_CLAMP = math.log(100000. / 16)
ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

class SinusoidalPositionEmbeddings(nn.Layer):
    def __init__(self, dim):
        super(SinusoidalPositionEmbeddings, self).__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = paddle.exp(paddle.arange(half_dim) * -embeddings)
        embeddings = time[:, None].cast("float32") * embeddings[None, :]
        embeddings = paddle.concat([embeddings.sin(), embeddings.cos()], axis=-1)
        return embeddings


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = paddle.linspace(0, timesteps, steps, dtype="float64")
    alphas_cumprod = paddle.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return paddle.clip(betas, 0, 0.999)


class RCNNHead_diffu(nn.Layer):
    def __init__(
            self,
            d_model,
            num_classes,
            dim_feedforward,
            nhead,
            dropout,
            head_cls,
            head_reg,
            head_dim_dynamic,
            head_num_dynamic,
            scale_clamp: float=_DEFAULT_SCALE_CLAMP,
            bbox_weights=(2.0, 2.0, 1.0, 1.0), ):
        super().__init__()

        self.d_model = d_model

        # dynamic.
        self.self_attn = nn.MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.inst_interact = DynamicConv(d_model, head_dim_dynamic,
                                         head_num_dynamic)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        
        # block time mlp
        self.block_time_mlp = nn.Sequential(nn.Silu(), nn.Linear(d_model * 4, d_model * 2))

        # cls.
        num_cls = head_cls
        cls_module = list()
        for _ in range(num_cls):
            cls_module.append(nn.Linear(d_model, d_model, bias_attr=False))
            cls_module.append(nn.LayerNorm(d_model))
            cls_module.append(nn.ReLU())
        self.cls_module = nn.LayerList(cls_module)

        # reg.
        num_reg = head_reg
        reg_module = list()
        for _ in range(num_reg):
            reg_module.append(nn.Linear(d_model, d_model, bias_attr=False))
            reg_module.append(nn.LayerNorm(d_model))
            reg_module.append(nn.ReLU())
        self.reg_module = nn.LayerList(reg_module)

        # pred.
        self.class_logits = nn.Linear(d_model, num_classes)
        self.bboxes_delta = nn.Linear(d_model, 4)
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights

    def forward(self, features, bboxes, pro_features, pooler, time_emb):
        """
        :param bboxes: (N, nr_boxes, 4)
        :param pro_features: (N, nr_boxes, d_model)
        """

        N, nr_boxes = bboxes.shape[:2]

        # roi_feature.
        proposal_boxes = list()
        for b in range(N):
            proposal_boxes.append(bboxes[b].astype("float32"))
        roi_num = paddle.full([N], nr_boxes).astype("int32")

        roi_features = pooler(features, proposal_boxes, roi_num)
        roi_features = roi_features.reshape(
            [N * nr_boxes, self.d_model, -1]).transpose(perm=[2, 0, 1])
        
        if pro_features is None:
            pro_features = roi_features.reshape([N, nr_boxes, self.d_model, -1]).mean(-1)
            
        # self_att.
        pro_features = pro_features.reshape([N, nr_boxes, self.d_model])
        pro_features2 = self.self_attn(
            pro_features, pro_features, value=pro_features)
        pro_features = pro_features.transpose(perm=[1, 0, 2]) + self.dropout1(
            pro_features2.transpose(perm=[1, 0, 2]))
        pro_features = self.norm1(pro_features)

        # inst_interact.
        pro_features = pro_features.reshape(
            [nr_boxes, N, self.d_model]).transpose(perm=[1, 0, 2]).reshape(
                [1, N * nr_boxes, self.d_model])
        pro_features2 = self.inst_interact(pro_features, roi_features)
        pro_features = pro_features + self.dropout2(pro_features2)
        obj_features = self.norm2(pro_features)

        # obj_feature.
        obj_features2 = self.linear2(
            self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)

        fc_feature = obj_features.transpose(perm=[1, 0, 2]).reshape(
            [N * nr_boxes, -1])
        
        # time
        scale_shift = self.block_time_mlp(time_emb)
        scale_shift = paddle.repeat_interleave(scale_shift, nr_boxes, axis=0)
        scale, shift = scale_shift.chunk(2, axis=1)
        fc_feature = fc_feature * (scale + 1) + shift
        
        
        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)
        class_logits = self.class_logits(cls_feature)
        bboxes_deltas = self.bboxes_delta(reg_feature)
        pred_bboxes = delta2bbox(bboxes_deltas,
                                 bboxes.reshape([-1, 4]), self.bbox_weights)

        return class_logits.reshape([N, nr_boxes, -1]), pred_bboxes.reshape(
            [N, nr_boxes, -1]), obj_features


@register
class DiffusionDetHead(nn.Layer):
    '''
    DiffusionDetHead
    Args:
        roi_input_shape (list[ShapeSpec]): The output shape of fpn
        num_classes (int): Number of classes,
        head_hidden_dim (int): The param of MultiHeadAttention,
        head_dim_feedforward (int): The param of MultiHeadAttention,
        nhead (int): The param of MultiHeadAttention,
        head_dropout (float): The p of dropout,
        head_cls (int): The number of class head,
        head_reg (int): The number of regressionhead,
        head_num_dynamic (int): The number of DynamicConv's param,
        head_num_heads (int): The number of RCNNHead_diffu,
        deep_supervision (int): wheather supervise the intermediate results,
        num_proposals (int): the number of proposals boxes and features
    '''
    __inject__ = ['loss_func']
    __shared__ = ['num_classes', 'use_focal', 'use_fed_loss']

    def __init__(
            self,
            head_hidden_dim,
            head_dim_feedforward,
            nhead,
            head_dropout,
            head_cls,
            head_reg,
            head_dim_dynamic,
            head_num_dynamic,
            head_num_heads,
            deep_supervision,
            num_proposals,
            num_classes=80,
            timesteps=1000,
            snr_scale=2.,
            sampling_timesteps=1,
            use_nms=True,
            use_focal=True,
            use_fed_loss=False,
            loss_func="SparseRCNNLoss",
            roi_input_shape=None, ):
        super().__init__()
        assert head_num_heads > 0, \
            f'At least one RoI Head is required, but {head_num_heads}.'

        # Build RoI.
        box_pooler = self._init_box_pooler(roi_input_shape)
        self.box_pooler = box_pooler
        
        # Gaussian random feature embedding layer for time
        d_model = head_hidden_dim
        self.d_model = d_model
        time_dim = d_model * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Build heads.
        rcnn_head = RCNNHead_diffu(
            head_hidden_dim,
            num_classes,
            head_dim_feedforward,
            nhead,
            head_dropout,
            head_cls,
            head_reg,
            head_dim_dynamic,
            head_num_dynamic, 
            )
        self.head_series = nn.LayerList(
            [copy.deepcopy(rcnn_head) for i in range(head_num_heads)])
        self.return_intermediate = deep_supervision

        self.objective = 'pred_x0'
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = paddle.cumprod(alphas, 0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        
        self.num_classes = num_classes
        self.num_timesteps = timesteps
        self.num_proposals = num_proposals
        self.scale = snr_scale
        self.sampling_timesteps = sampling_timesteps
        
        self.ddim_sampling_eta = 1.
        self.self_condition = False
        self.box_renewal = True
        self.use_ensemble = True
        
        self.use_focal = use_focal
        self.use_fed_loss = use_fed_loss
        self.use_nms = use_nms
        
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps

        self.register_buffer('betas', betas, persistable=False)
        self.register_buffer('alphas_cumprod', alphas_cumprod, persistable=False)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev, persistable=False)


        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', paddle.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', paddle.sqrt(1. - alphas_cumprod))
        # self.register_buffer('log_one_minus_alphas_cumprod', paddle.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', paddle.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', paddle.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', paddle.log(posterior_variance.clip(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * paddle.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * paddle.sqrt(alphas) / (1. - alphas_cumprod))



        # build init proposal
        self.init_proposal_features = nn.Embedding(num_proposals,
                                                   head_hidden_dim)
        self.init_proposal_boxes = nn.Embedding(num_proposals, 4)

        self.lossfunc = loss_func

        # Init parameters.
        init.reset_initialized_parameter(self)
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        for m in self.sublayers():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight, reverse=True)
            elif not isinstance(m, nn.Embedding) and hasattr(
                    m, "weight") and m.weight.dim() > 1:
                init.xavier_normal_(m.weight, reverse=False)

            if hasattr(m, "bias") and m.bias is not None and m.bias.shape[
                    -1] == self.num_classes:
                init.constant_(m.bias, bias_value)

        init_bboxes = paddle.empty_like(self.init_proposal_boxes.weight)
        init_bboxes[:, :2] = 0.5
        init_bboxes[:, 2:] = 1.0
        self.init_proposal_boxes.weight.set_value(init_bboxes)

    @staticmethod
    def _init_box_pooler(input_shape):

        pooler_resolution = 7
        sampling_ratio = 2

        if input_shape is not None:
            pooler_scales = tuple(1.0 / input_shape[k].stride
                                  for k in range(len(input_shape)))
            in_channels = [
                input_shape[f].channels for f in range(len(input_shape))
            ]
            end_level = len(input_shape) - 1
            # Check all channel counts are equal
            assert len(set(in_channels)) == 1, in_channels
        else:
            pooler_scales = [1.0 / 4.0, 1.0 / 8.0, 1.0 / 16.0, 1.0 / 32.0]
            end_level = 3

        box_pooler = RoIAlign(
            resolution=pooler_resolution,
            spatial_scale=pooler_scales,
            sampling_ratio=sampling_ratio,
            end_level=end_level,
            aligned=True)
        return box_pooler
    
    
    def prepare_targets(self, targets):
        new_targets = []
        diffused_boxes = []
        noises = []
        ts = []
        
        bs = targets['im_id'].shape[0]
        
        for im_id in range(bs):
            
            target = {}
            
            image_size_xyxy = targets["img_whwh"][im_id]
            gt_classes = targets["gt_class"][im_id]
            gt_boxes = targets["gt_bbox"][im_id] / image_size_xyxy if targets["gt_bbox"][im_id].size != 0 else targets["gt_bbox"][im_id]
            gt_boxes = bbox_xyxy_to_cxcywh(gt_boxes)
            d_boxes, d_noise, d_t = self.prepare_diffusion_concat(gt_boxes)

            diffused_boxes.append(d_boxes)
            noises.append(d_noise)
            ts.append(d_t)
            
            target["labels"] = gt_classes.flatten()
            target["boxes"] = gt_boxes
            target["boxes_xyxy"] = targets["gt_bbox"][im_id]
            target["image_size_xyxy"] = image_size_xyxy
            image_size_xyxy_tgt = image_size_xyxy[None].tile([len(gt_boxes), 1]) if 0 != gt_boxes.size else image_size_xyxy[None]
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt
            # target["area"] = targets_per_image.gt_boxes.area().to(self.device)
            new_targets.append(target)

        return new_targets, paddle.stack(diffused_boxes), paddle.stack(noises), paddle.stack(ts)

    
    def prepare_diffusion_concat(self, gt_boxes):
        """
        :param gt_boxes: (cx, cy, w, h), normalized
        :param num_proposals:
        """
        t = paddle.randint(low=0, high=self.num_timesteps, shape=(1,)).astype("int64")
        noise = paddle.randn([self.num_proposals, 4])

        num_gt = gt_boxes.shape[0]
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            gt_boxes = paddle.to_tensor([[0.5, 0.5, 1., 1.]], dtype="float32")
            num_gt = 1

        if num_gt < self.num_proposals:
            box_placeholder = paddle.randn([self.num_proposals - num_gt, 4]) / 6. + 0.5  # 3sigma = 1/2 --> sigma: 1/6
            box_placeholder[:, 2:] = paddle.clip(box_placeholder[:, 2:], min=1e-4)
            x_start = paddle.concat([gt_boxes, box_placeholder], axis=0)
        elif num_gt > self.num_proposals:
            select_mask = [True] * self.num_proposals + [False] * (num_gt - self.num_proposals)
            random.shuffle(select_mask)
            x_start = gt_boxes[select_mask]
        else:
            x_start = gt_boxes

        x_start = (x_start * 2. - 1.) * self.scale

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = paddle.clip(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.

        diff_boxes = bbox_cxcywh_to_xyxy(x)

        return diff_boxes, noise, t


    def extract(self, a, t, x_shape):
        """extract the appropriate  t  index for a batch of indices"""
        batch_size = t.shape[0]
        out = a.gather(axis=a.ndim-1, index=t)
        return out.reshape([batch_size, *((1,) * (len(x_shape) - 1))])

    
    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = paddle.randn(shape=x_start.shape, dtype=x_start.dtype)

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                 self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, backbone_feats, images_whwh, x, t, x_self_cond=None, clip_x_start=False):
        x_boxes = paddle.clip(x, min=-1 * self.scale, max=self.scale)
        x_boxes = ((x_boxes / self.scale) + 1) / 2
        x_boxes = bbox_cxcywh_to_xyxy(x_boxes)
        x_boxes = x_boxes * images_whwh[:, None, :]
        outputs_class, outputs_coord = self.inter_forward(backbone_feats, x_boxes, t, None)

        x_start = outputs_coord[-1]  # (batch, num_proposals, 4) predict boxes: absolute coordinates (x1, y1, x2, y2)
        x_start = x_start / images_whwh[:, None, :]
        x_start = bbox_xyxy_to_cxcywh(x_start)
        x_start = (x_start * 2 - 1.) * self.scale
        x_start = paddle.clip(x_start, min=-1 * self.scale, max=self.scale)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start), outputs_class, outputs_coord
    
    

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        if self.use_focal or self.use_fed_loss:
            scores = F.sigmoid(box_cls)
            labels = paddle.arange(self.num_classes). \
                unsqueeze(0).tile([self.num_proposals, 1]).flatten(0, 1)

            for i, (scores_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, box_pred, image_sizes
            )):
                # result = Instances(image_size)
                scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(self.num_proposals, sorted=False)
                labels_per_image = labels[topk_indices]
                box_pred_per_image = box_pred_per_image.reshape([-1, 1, 4]).tile([1, self.num_classes, 1]).reshape([-1, 4])
                box_pred_per_image = box_pred_per_image[topk_indices]

                if self.use_ensemble and self.sampling_timesteps > 1:
                    return box_pred_per_image, scores_per_image, labels_per_image

                if self.use_nms:
                    # keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                    cls_arange = paddle.arange(self.num_classes)
                    keep = batched_nms(box_pred_per_image, 0.5, scores_per_image, labels_per_image, cls_arange)
                    box_pred_per_image = box_pred_per_image[keep]
                    scores_per_image = scores_per_image[keep]
                    labels_per_image = labels_per_image[keep]

                result = {}
                result['pred_boxes'] = box_pred_per_image
                result['scores'] = scores_per_image
                result['pred_classes'] = labels_per_image
                results.append(result)

        else:
            # For each box we assign the best class or the second best if the best on is `no_object`.
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

            for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, labels, box_pred, image_sizes
            )):
                if self.use_ensemble and self.sampling_timesteps > 1:
                    return box_pred_per_image, scores_per_image, labels_per_image

                if self.use_nms:
                    # keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                    keep = batched_nms(box_pred_per_image, 0.5, scores_per_image, labels_per_image)
                    box_pred_per_image = box_pred_per_image[keep]
                    scores_per_image = scores_per_image[keep]
                    labels_per_image = labels_per_image[keep]
                # result = Instances(image_size)
                result = {}
                result["pred_boxes"] = box_pred_per_image
                result["scores"] = scores_per_image
                result["pred_classes"] = labels_per_image
                results.append(result)

        return results
    
    
    
    

    @paddle.no_grad()
    def ddim_sample(self, batched_inputs, backbone_feats, images=None, clip_denoised=True, do_postprocess=True):
        
        
        images_real_whwh = batched_inputs["img_whwh"]
        batch = images_real_whwh.shape[0]
        shape = (batch, self.num_proposals, 4)
        total_timesteps, sampling_timesteps, eta, objective = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = paddle.linspace(-1, total_timesteps - 1, num=sampling_timesteps + 1)
        times = list(reversed(times.cast("int32").tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = paddle.randn(shape)

        ensemble_score, ensemble_label, ensemble_coord = [], [], []
        x_start = None
        for time, time_next in time_pairs:
            time_cond = paddle.full(shape=(batch,), fill_value=time, dtype="int64")
            self_cond = x_start if self.self_condition else None

            preds, outputs_class, outputs_coord = self.model_predictions(backbone_feats, images_real_whwh, 
                                                                         img, time_cond,
                                                                         self_cond, clip_x_start=clip_denoised)
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start

            if self.box_renewal:  # filter
                score_per_image, box_per_image = outputs_class[-1][0], outputs_coord[-1][0]
                threshold = 0.5
                score_per_image = F.sigmoid(score_per_image)
                value = paddle.max(score_per_image, axis=-1, keepdim=False)
                keep_idx = value > threshold
                num_remain = keep_idx.sum()
                
                
                bs_zero_4_tensor = paddle.to_tensor(np.array([]).reshape([batch, 0, 4]))
                pred_noise = pred_noise[:, keep_idx] if keep_idx.all().item() else bs_zero_4_tensor.astype(pred_noise.dtype)
                x_start = x_start[:, keep_idx] if keep_idx.all().item() else bs_zero_4_tensor.astype(x_start.dtype)
                img = img[:, keep_idx] if keep_idx.all().item() else bs_zero_4_tensor.astype(img.dtype)
            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            if self.box_renewal:  # filter
                # replenish with randn boxes
                img = paddle.concat((img, torch.randn(1, self.num_proposals - num_remain, 4, device=img.device)), dim=1)
            if self.use_ensemble and self.sampling_timesteps > 1:
                box_pred_per_image, scores_per_image, labels_per_image = self.inference(outputs_class[-1],
                                                                                        outputs_coord[-1],
                                                                                        images.image_sizes)
                ensemble_score.append(scores_per_image)
                ensemble_label.append(labels_per_image)
                ensemble_coord.append(box_pred_per_image)

        if self.use_ensemble and self.sampling_timesteps > 1: # False
            box_pred_per_image = paddle.concat(ensemble_coord, axis=0)
            scores_per_image = paddle.concat(ensemble_score, axis=0)
            labels_per_image = paddle.concat(ensemble_label, axis=0)
            if self.use_nms:
                # keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                keep = batched_nms(box_pred_per_image, 0.5, scores_per_image, labels_per_image)
                box_pred_per_image = box_pred_per_image[keep]
                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]

            # result = Instances(images.image_sizes[0])
            result = {}
            result["pred_boxes"] = box_pred_per_image
            result["scores"] = scores_per_image
            result["pred_classes"] = labels_per_image
            results = [result]
        else:
            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred, images_real_whwh)
            
        # if do_postprocess:
        #     processed_results = []
        #     for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
        #         height = input_per_image.get("height", image_size[0])
        #         width = input_per_image.get("width", image_size[1])
        #         r = detector_postprocess(results_per_image, height, width)
        #         processed_results.append({"instances": r})
        #     return processed_results
        
        return results
    
    def forward(self, features, batched_inputs, init_features=None):
        

        # Prepare Proposals.
        if not self.training:
            results = self.ddim_sample(batched_inputs, features)
            return results, None
        
        
        targets, x_boxes, noises, t = self.prepare_targets(batched_inputs)
        t = t.squeeze(-1)
        init_bboxes = x_boxes * batched_inputs['img_whwh'][:, None, :]
        
        time = self.time_mlp(t)

        inter_class_logits = []
        inter_pred_bboxes = []

        bs = len(features[0])
        bboxes = init_bboxes
        # num_boxes = bboxes.shape[1]

        if init_features is not None:
            init_features = init_features[None].repeat(1, bs, 1)
            proposal_features = init_features.clone()
        else:
            proposal_features = None
        
        
        for stage, rcnn_head in enumerate(self.head_series):
            class_logits, pred_bboxes, proposal_features = rcnn_head(features, bboxes, proposal_features, self.box_pooler, time)
            if self.return_intermediate or stage == len(self.head_series) - 1:
                inter_class_logits.append(class_logits)
                inter_pred_bboxes.append(pred_bboxes)
            bboxes = pred_bboxes.detach()


        output = {
            'pred_logits': inter_class_logits[-1],
            'pred_boxes': inter_pred_bboxes[-1]
        }
        if self.return_intermediate:
            output['aux_outputs'] = [{
                'pred_logits': a,
                'pred_boxes': b
            } for a, b in zip(inter_class_logits[:-1], inter_pred_bboxes[:-1])]

        return output, targets


    def inter_forward(self, features, init_bboxes, t, init_features):
        # assert t shape (batch_size)
        time = self.time_mlp(t)

        inter_class_logits = []
        inter_pred_bboxes = []

        bs = len(features[0])
        bboxes = init_bboxes
        num_boxes = bboxes.shape[1]

        if init_features is not None:
            init_features = init_features[None].tile([1, bs, 1])
            proposal_features = init_features.clone()
        else:
            proposal_features = None
        
        for head_idx, rcnn_head in enumerate(self.head_series):
            class_logits, pred_bboxes, proposal_features = rcnn_head(features, bboxes, proposal_features, self.box_pooler, time)
            if self.return_intermediate:
                inter_class_logits.append(class_logits)
                inter_pred_bboxes.append(pred_bboxes)
            bboxes = pred_bboxes.detach()

        if self.return_intermediate:
            return paddle.stack(inter_class_logits), paddle.stack(inter_pred_bboxes)

        return class_logits[None], pred_bboxes[None]



        # if self.return_intermediate:
        #     return paddle.stack(inter_class_logits), paddle.stack(inter_pred_bboxes)
        
        # return class_logits[None], pred_bboxes[None]
        

        # init_features = self.init_proposal_features.weight.unsqueeze(0).tile([bs, 1, 1])
        # proposal_features = init_features.clone()

        # inter_class_logits = []
        # inter_pred_bboxes = []

        # for stage, rcnn_head in enumerate(self.head_series):
        #     class_logits, pred_bboxes, proposal_features = rcnn_head(
        #         features, bboxes, proposal_features, self.box_pooler)

        #     if self.return_intermediate or stage == len(self.head_series) - 1:
        #         inter_class_logits.append(class_logits)
        #         inter_pred_bboxes.append(pred_bboxes)
        #     bboxes = pred_bboxes.detach()

        # output = {
        #     'pred_logits': inter_class_logits[-1],
        #     'pred_boxes': inter_pred_bboxes[-1]
        # }
        # if self.return_intermediate:
        #     output['aux_outputs'] = [{
        #         'pred_logits': a,
        #         'pred_boxes': b
        #     } for a, b in zip(inter_class_logits[:-1], inter_pred_bboxes[:-1])]

        # return output
    
    

    def get_loss(self, outputs, targets):
        losses = self.lossfunc(outputs, targets)
        weight_dict = self.lossfunc.weight_dict

        for k in losses.keys():
            if k in weight_dict:
                losses[k] *= weight_dict[k]

        return losses


def bbox_cxcywh_to_xyxy(x):
    
    if x.size == 0:
        return x
    
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return paddle.stack(b, axis=-1)


def bbox_xyxy_to_cxcywh(x):
    
    if x.size == 0:
        return x
    
    x1, y1, x2, y2 = x.split(4, axis=-1)
    return paddle.concat(
        [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)], axis=-1)