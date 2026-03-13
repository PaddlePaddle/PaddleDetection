# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.regularizer import L2Decay

from ppdet.core.workspace import register
from .rtdetr_transformer import TransformerDecoderLayer
from .utils import (_get_clones, inverse_sigmoid, get_denoising_training_group,
                    mask_to_box_coordinate)
from ..heads.detr_head import MLP
from ..initializer import (linear_init_, constant_, xavier_uniform_, bias_init_with_prob)

__all__ = ['MaskRTDETR', 'DocLayoutV3Transformer']


def _get_pred_class_and_mask(query_embed,
                             mask_feat,
                             dec_norm,
                             score_head,
                             mask_query_head):
    out_query = dec_norm(query_embed)
    out_logits = score_head(out_query)
    mask_query_embed = mask_query_head(out_query)
    batch_size, mask_dim, _ = mask_query_embed.shape
    _, _, mask_h, mask_w = mask_feat.shape
    out_mask = paddle.bmm(
        mask_query_embed, mask_feat.flatten(2)).reshape(
        [batch_size, mask_dim, mask_h, mask_w])
    return out_logits, out_mask


class MaskTransformerDecoder(nn.Layer):
    """
    Mask RT-DETR Transformer Decoder.

    This decoder processes queries through multiple transformer layers and
    produces bounding box, classification, and mask predictions.
    """

    def __init__(self,
                 hidden_dim,
                 decoder_layer,
                 num_layers,
                 eval_idx=-1,
                 eval_topk=100):
        super(MaskTransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 \
            else num_layers + eval_idx
        self.eval_topk = eval_topk

    def forward(self,
                tgt,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                mask_feat,
                bbox_head,
                score_head,
                query_pos_head,
                mask_query_head,
                dec_norm,
                attn_mask=None,
                memory_mask=None,
                query_pos_head_inv_sig=False):
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_masks = []
        ref_points_detach = F.sigmoid(ref_points_unact)
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            if not query_pos_head_inv_sig:
                query_pos_embed = query_pos_head(ref_points_detach)
            else:
                query_pos_embed = query_pos_head(
                    inverse_sigmoid(ref_points_detach))

            output = layer(output, ref_points_input, memory,
                           memory_spatial_shapes, memory_level_start_index,
                           attn_mask, memory_mask, query_pos_embed)

            inter_ref_bbox = F.sigmoid(bbox_head(output) +
                                       inverse_sigmoid(ref_points_detach))

            if self.training:
                logits_, masks_ = _get_pred_class_and_mask(
                    output, mask_feat, dec_norm,
                    score_head, mask_query_head)
                dec_out_logits.append(logits_)
                dec_out_masks.append(masks_)
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(
                        F.sigmoid(bbox_head(output) +
                                  inverse_sigmoid(ref_points)))
            elif i == self.eval_idx:
                logits_, masks_ = _get_pred_class_and_mask(
                    output, mask_feat, dec_norm,
                    score_head, mask_query_head)
                dec_out_logits.append(logits_)
                dec_out_masks.append(masks_)
                dec_out_bboxes.append(inter_ref_bbox)
                return (paddle.stack(dec_out_bboxes),
                        paddle.stack(dec_out_logits),
                        paddle.stack(dec_out_masks))

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach(
            ) if self.training else inter_ref_bbox

        return (paddle.stack(dec_out_bboxes),
                paddle.stack(dec_out_logits),
                paddle.stack(dec_out_masks))


@register
class MaskRTDETR(nn.Layer):
    """
    Mask RT-DETR model for instance segmentation.

    This model uses RT-DETR architecture with mask prediction capability
    for instance segmentation tasks.

    Args:
        num_classes (int): Number of object classes.
        hidden_dim (int): Hidden dimension of transformer.
        num_queries (int): Number of object queries.
        position_embed_type (str): Type of position embedding ('sine' or 'learned').
        backbone_feat_channels (list): Channels of backbone features.
        feat_strides (list): Strides of backbone features.
        num_prototypes (int): Number of mask prototypes.
        num_levels (int): Number of feature levels.
        num_decoder_points (int): Number of decoder points.
        nhead (int): Number of attention heads.
        num_decoder_layers (int): Number of decoder layers.
        dim_feedforward (int): Dimension of feedforward network.
        dropout (float): Dropout rate.
        activation (str): Activation function.
        num_denoising (int): Number of denoising queries.
        label_noise_ratio (float): Label noise ratio.
        box_noise_scale (float): Box noise scale.
        learnt_init_query (bool): Whether to use learnt query initialization.
        query_pos_head_inv_sig (bool): Whether to use inverse sigmoid for query position.
        mask_enhanced (bool): Whether to use mask-enhanced anchor initialization.
        eval_size (list|None): Evaluation size for anchor generation.
        eval_idx (int): Index of decoder layer for evaluation.
        eps (float): Small value for numerical stability.
    """
    __shared__ = ['num_classes', 'hidden_dim', 'eval_size', 'num_prototypes']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 backbone_feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_prototypes=32,
                 num_levels=3,
                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.4,
                 box_noise_scale=0.4,
                 learnt_init_query=False,
                 query_pos_head_inv_sig=False,
                 mask_enhanced=True,
                 eval_size=None,
                 eval_idx=-1,
                 eps=1e-2):
        super(MaskRTDETR, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(backbone_feat_channels) <= num_levels
        assert len(feat_strides) == len(backbone_feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.mask_enhanced = mask_enhanced
        self.eval_size = eval_size

        # backbone feature projection
        self._build_input_proj_layer(backbone_feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels,
            num_decoder_points)
        self.decoder = MaskTransformerDecoder(hidden_dim, decoder_layer,
                                              num_decoder_layers, eval_idx)

        # denoising part
        self.denoising_class_embed = nn.Embedding(
            num_classes,
            hidden_dim,
            weight_attr=ParamAttr(initializer=nn.initializer.Normal()))
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim,
                                  hidden_dim, num_layers=2)
        self.query_pos_head_inv_sig = query_pos_head_inv_sig

        # mask embedding
        self.mask_query_head = MLP(hidden_dim, hidden_dim,
                                   num_prototypes, num_layers=3)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(
                hidden_dim,
                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0))))

        # decoder norm layer
        self.dec_norm = nn.LayerNorm(
            hidden_dim,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        # shared prediction head
        self.score_head = nn.Linear(hidden_dim, num_classes)
        self.bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize model parameters."""
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01)
        linear_init_(self.score_head)
        constant_(self.score_head.bias, bias_cls)
        constant_(self.bbox_head.layers[-1].weight)
        constant_(self.bbox_head.layers[-1].bias)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for l in self.input_proj:
            xavier_uniform_(l[0].weight)

        # init encoder output anchors and valid_mask
        if self.eval_size:
            self.anchors, self.valid_mask = self._generate_anchors()

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'backbone_feat_channels': [i.channels for i in input_shape],
                'feat_strides': [i.stride for i in input_shape]}

    def _build_input_proj_layer(self, backbone_feat_channels):
        """Build input projection layers for backbone features."""
        self.input_proj = nn.LayerList()
        for in_channels in backbone_feat_channels:
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=1,
                        bias_attr=False)),
                    ('norm', nn.BatchNorm2D(
                        self.hidden_dim,
                        weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                        bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
        in_channels = backbone_feat_channels[-1]
        for _ in range(self.num_levels - len(backbone_feat_channels)):
            self.input_proj.append(
                nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels,
                        self.hidden_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias_attr=False)),
                    ('norm', nn.BatchNorm2D(
                        self.hidden_dim,
                        weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                        bias_attr=ParamAttr(regularizer=L2Decay(0.0))))))
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        """Get encoder input from backbone features."""
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).transpose([0, 2, 1]))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = paddle.concat(feat_flatten, 1)
        level_start_index.pop()
        return feat_flatten, spatial_shapes, level_start_index

    def forward(self, feats, pad_mask=None, gt_meta=None, is_teacher=False):
        """
        Forward pass of MaskRTDETR.

        Args:
            feats (tuple): (encoder features, mask features)
            pad_mask (Tensor|None): Padding mask.
            gt_meta (dict|None): Ground truth metadata for denoising training.
            is_teacher (bool): Whether this is a teacher model.

        Returns:
            tuple: (out_logits, out_bboxes, out_masks, enc_out, init_out, dn_meta)
        """
        enc_feats, mask_feat = feats
        # input projection and embedding
        (memory, spatial_shapes,
         level_start_index) = self._get_encoder_input(enc_feats)

        # prepare denoising training
        if self.training:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_denoising_training_group(gt_meta,
                                             self.num_classes,
                                             self.num_queries,
                                             self.denoising_class_embed.weight,
                                             self.num_denoising,
                                             self.label_noise_ratio,
                                             self.box_noise_scale)
        else:
            denoising_class, denoising_bbox_unact,\
                attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_out, init_out = \
            self._get_decoder_input(
                memory, mask_feat, spatial_shapes,
                denoising_class, denoising_bbox_unact, is_teacher)

        # decoder
        out_bboxes, out_logits, out_masks = self.decoder(
            target,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            level_start_index,
            mask_feat,
            self.bbox_head,
            self.score_head,
            self.query_pos_head,
            self.mask_query_head,
            self.dec_norm,
            attn_mask=attn_mask,
            memory_mask=None,
            query_pos_head_inv_sig=self.query_pos_head_inv_sig)

        return out_logits, out_bboxes, out_masks, enc_out, init_out, dn_meta

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype=paddle.float32):
        """Generate anchor boxes for encoder output."""
        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.eval_size[0] / s), int(self.eval_size[1] / s)]
                for s in self.feat_strides
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = paddle.meshgrid(
                paddle.arange(
                    end=h, dtype=dtype),
                paddle.arange(
                    end=w, dtype=dtype))
            grid_xy = paddle.stack([grid_x, grid_y], -1)

            valid_WH = paddle.to_tensor([h, w]).astype(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = paddle.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            anchors.append(
                paddle.concat([grid_xy, wh], -1).reshape([-1, h * w, 4]))

        anchors = paddle.concat(anchors, 1)
        valid_mask = ((anchors > self.eps) *
                      (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = paddle.log(anchors / (1 - anchors))
        anchors = paddle.where(valid_mask, anchors,
                               paddle.to_tensor(float("inf")))
        return anchors, valid_mask

    def _get_decoder_input(self,
                           memory,
                           mask_feat,
                           spatial_shapes,
                           denoising_class=None,
                           denoising_bbox_unact=None,
                           is_teacher=False):
        """
        Get decoder input from encoder output.

        Args:
            memory (Tensor): Encoder output memory.
            mask_feat (Tensor): Mask features.
            spatial_shapes (list): Spatial shapes of each level.
            denoising_class (Tensor|None): Denoising class embeddings.
            denoising_bbox_unact (Tensor|None): Denoising bounding boxes.
            is_teacher (bool): Whether this is a teacher model.

        Returns:
            tuple: (target, reference_points_unact, enc_out, init_out)
        """
        bs, _, _ = memory.shape
        # prepare input for decoder
        if self.training or self.eval_size is None or is_teacher:
            anchors, valid_mask = self._generate_anchors(spatial_shapes)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
        memory = paddle.where(valid_mask, memory, paddle.to_tensor(0.))
        output_memory = self.enc_output(memory)

        enc_logits_unact = self.score_head(output_memory)
        enc_bboxes_unact = self.bbox_head(output_memory) + anchors

        # get topk index
        _, topk_ind = paddle.topk(
            enc_logits_unact.max(-1), self.num_queries, axis=1)
        batch_ind = paddle.arange(end=bs).astype(topk_ind.dtype)
        batch_ind = batch_ind.unsqueeze(-1).tile([1, self.num_queries])
        topk_ind = paddle.stack([batch_ind, topk_ind], axis=-1)

        # extract content and position query embedding
        target = paddle.gather_nd(output_memory, topk_ind)
        reference_points_unact = paddle.gather_nd(enc_bboxes_unact,
                                                  topk_ind)  # unsigmoided.
        # get encoder output: {logits, bboxes, masks}
        enc_out_logits, enc_out_masks = _get_pred_class_and_mask(
            target, mask_feat, self.dec_norm,
            self.score_head, self.mask_query_head)
        enc_out_bboxes = F.sigmoid(reference_points_unact)
        enc_out = (enc_out_logits, enc_out_bboxes, enc_out_masks)

        # concat denoising query
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = target.detach()
        if denoising_class is not None:
            target = paddle.concat([denoising_class, target], 1)
        if self.mask_enhanced:
            # use mask-enhanced anchor box initialization
            reference_points = mask_to_box_coordinate(
                enc_out_masks > 0, normalize=True, format="xywh")
            reference_points_unact = inverse_sigmoid(reference_points)
        if denoising_bbox_unact is not None:
            reference_points_unact = paddle.concat(
                [denoising_bbox_unact, reference_points_unact], 1)

        # direct prediction from the matching and denoising part in the beginning
        if self.training and denoising_class is not None:
            init_out_logits, init_out_masks = _get_pred_class_and_mask(
                target, mask_feat, self.dec_norm,
                self.score_head, self.mask_query_head)
            init_out_bboxes = F.sigmoid(reference_points_unact)
            init_out = (init_out_logits, init_out_bboxes, init_out_masks)
        else:
            init_out = None

        return target, reference_points_unact.detach(), enc_out, init_out


class DocLayoutV3TransformerDecoder(MaskTransformerDecoder):
    """
    PP-DocLayoutV3 Transformer Decoder with reading order prediction support.

    This decoder extends MaskTransformerDecoder by adding reading order prediction
    capability for document layout analysis. It predicts pairwise reading order
    relationships between detected elements using a global pointer network.

    The key enhancement over the base class is the addition of order_logits output,
    which represents the relative reading order between all pairs of queries.
    Each decoder layer produces its own order predictions for auxiliary supervision.

    Args:
        hidden_dim (int): Hidden dimension of transformer. Default: 256.
        decoder_layer (nn.Layer): Transformer decoder layer module.
        num_layers (int): Number of stacked decoder layers. Default: 6.
        num_queries (int|None): Number of object queries (excluding denoising queries).
            If not None, only the last num_queries are used for order prediction to
            exclude denoising queries. Default: None.
        eval_idx (int): Index of decoder layer to use for evaluation output.
            Negative values count from the end. Default: -1 (last layer).
        eval_topk (int): Number of top-scoring predictions to keep during evaluation.
            Default: 100.

    Note:
        The order prediction is only applied to matching queries (the last num_queries),
        not to denoising queries, as denoising queries do not have ground truth order labels.
    """

    def __init__(self,
                 hidden_dim,
                 decoder_layer,
                 num_layers,
                 num_queries=None,
                 eval_idx=-1,
                 eval_topk=100):
        super(DocLayoutV3TransformerDecoder, self).__init__(
            hidden_dim, decoder_layer, num_layers, eval_idx, eval_topk)
        self.num_queries = num_queries

    def forward(self,
                tgt,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                mask_feat,
                bbox_head,
                score_head,
                order_head,
                global_pointer,
                query_pos_head,
                mask_query_head,
                dec_norm,
                attn_mask=None,
                memory_mask=None,
                query_pos_head_inv_sig=False):
        """
        Forward pass with reading order prediction.

        This method extends the base decoder forward pass by computing reading order
        predictions at each decoder layer using the global pointer mechanism.

        Args:
            tgt (Tensor): Target query embeddings.
                Shape: [batch_size, num_queries, hidden_dim]
            ref_points_unact (Tensor): Reference points before sigmoid activation.
                Shape: [batch_size, num_queries, 4]
            memory (Tensor): Encoder output memory.
                Shape: [batch_size, num_memory, hidden_dim]
            memory_spatial_shapes (list): Spatial shapes [H, W] for each feature level.
            memory_level_start_index (list): Start indices for each feature level in memory.
            mask_feat (Tensor): Mask features from encoder.
                Shape: [batch_size, num_prototypes, H, W]
            bbox_head (nn.Layer): Bounding box prediction head (shared across layers).
            score_head (nn.Layer): Classification score head (shared across layers).
            order_head (nn.LayerList): Order prediction heads (one per decoder layer).
            global_pointer (nn.Layer): Global pointer module for pairwise order prediction.
            query_pos_head (nn.Layer): Query position embedding head.
            mask_query_head (nn.Layer): Mask query embedding head.
            dec_norm (nn.Layer): Decoder output normalization layer.
            attn_mask (Tensor|None): Attention mask for denoising queries. Default: None.
            memory_mask (Tensor|None): Memory mask for padding. Default: None.
            query_pos_head_inv_sig (bool): Whether to apply inverse sigmoid to reference
                points before feeding to query_pos_head. Default: False.

        Returns:
            tuple: (dec_out_bboxes, dec_out_logits, dec_out_masks, dec_out_order_logits)
                - dec_out_bboxes (Tensor): Predicted bounding boxes from all layers.
                    Shape: [num_layers, batch_size, num_queries, 4]
                - dec_out_logits (Tensor): Classification logits from all layers.
                    Shape: [num_layers, batch_size, num_queries, num_classes]
                - dec_out_masks (Tensor): Predicted masks from all layers.
                    Shape: [num_layers, batch_size, num_queries, mask_h, mask_w]
                - dec_out_order_logits (Tensor): Pairwise order logits from all layers.
                    Shape: [num_layers, batch_size, num_queries, num_queries]
        """
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_masks = []
        dec_out_order_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact)
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            if not query_pos_head_inv_sig:
                query_pos_embed = query_pos_head(ref_points_detach)
            else:
                query_pos_embed = query_pos_head(
                    inverse_sigmoid(ref_points_detach))

            output = layer(output, ref_points_input, memory,
                           memory_spatial_shapes, memory_level_start_index,
                           attn_mask, memory_mask, query_pos_embed)

            inter_ref_bbox = F.sigmoid(bbox_head(output) +
                                       inverse_sigmoid(ref_points_detach))

            if self.training:
                logits_, masks_ = _get_pred_class_and_mask(
                    output, mask_feat, dec_norm,
                    score_head, mask_query_head)

                dec_out_logits.append(logits_)
                dec_out_masks.append(masks_)

                # Extract valid matching queries for order prediction (exclude denoising queries)
                # Denoising queries are prepended to the query sequence during training,
                # so we only use the last num_queries for order prediction
                valid_output = output[:, -self.num_queries:] if self.num_queries is not None else output
                dec_out_order_logits.append(global_pointer(order_head[i](valid_output)))

                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(
                        F.sigmoid(bbox_head(output) +
                                  inverse_sigmoid(ref_points)))
            elif i == self.eval_idx:
                logits_, masks_ = _get_pred_class_and_mask(
                    output, mask_feat, dec_norm,
                    score_head, mask_query_head)

                dec_out_logits.append(logits_)
                dec_out_masks.append(masks_)
                dec_out_bboxes.append(inter_ref_bbox)

                # Extract valid matching queries for order prediction
                valid_output = output[:, -self.num_queries:] if self.num_queries is not None else output
                dec_out_order_logits.append(global_pointer(order_head[i](valid_output)))

                return (paddle.stack(dec_out_bboxes),
                        paddle.stack(dec_out_logits),
                        paddle.stack(dec_out_masks),
                        paddle.stack(dec_out_order_logits))

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach(
            ) if self.training else inter_ref_bbox

        return (paddle.stack(dec_out_bboxes),
                paddle.stack(dec_out_logits),
                paddle.stack(dec_out_masks),
                paddle.stack(dec_out_order_logits))


@register
class DocLayoutV3Transformer(MaskRTDETR):
    """
    PP-DocLayoutV3 Transformer for document layout analysis with reading order prediction.

    This model extends MaskRTDETR to predict reading order of detected elements in
    document images. It uses a global pointer mechanism to model pairwise reading
    order relationships between elements, which are then decoded into a sequential order.

    Key enhancements over MaskRTDETR:
        1. Independent order prediction head for each decoder layer, enabling deep
           supervision for better training convergence.
        2. Global pointer module that predicts pairwise order relationships using
           query-key interactions with antisymmetric constraints.
        3. Extended output to include order_logits for both training and inference.

    Architecture:
        Input Image -> Backbone -> Neck -> Encoder -> Decoder -> Predictions
                                                          |
                                                          v
                                    Order Heads -> Global Pointer -> Order Logits

    The reading order prediction branch operates on decoder query features and
    produces an NxN matrix of pairwise order relationships, where order_logits[i,j]
    indicates whether element i comes before element j in reading order.

    Args:
        num_classes (int): Number of object classes (e.g., 25 for DocLayout). Default: 80.
        hidden_dim (int): Hidden dimension of transformer features. Default: 256.
        num_queries (int): Number of object queries for detection. Default: 300.
        position_embed_type (str): Type of position embedding ('sine' or 'learned').
            Default: 'sine'.
        backbone_feat_channels (list[int]): Output channels of backbone feature pyramid.
            Default: [512, 1024, 2048].
        feat_strides (list[int]): Feature strides corresponding to backbone features.
            Default: [8, 16, 32].
        num_prototypes (int): Number of mask prototype channels. Default: 32.
        num_levels (int): Number of feature pyramid levels. Default: 3.
        num_decoder_points (int): Number of sampling points in deformable attention.
            Default: 4.
        nhead (int): Number of attention heads in transformer. Default: 8.
        num_decoder_layers (int): Number of decoder layers. Default: 6.
        dim_feedforward (int): Dimension of feedforward network. Default: 1024.
        dropout (float): Dropout rate. Default: 0.0.
        activation (str): Activation function type ('relu', 'gelu', etc.). Default: 'relu'.
        num_denoising (int): Number of denoising queries for training stability. Default: 100.
        label_noise_ratio (float): Noise ratio for label denoising. Default: 0.4.
        box_noise_scale (float): Noise scale for box denoising. Default: 0.4.
        learnt_init_query (bool): Whether to use learnable query initialization. Default: False.
        query_pos_head_inv_sig (bool): Whether to apply inverse sigmoid to query positions.
            Default: False.
        mask_enhanced (bool): Whether to use mask-enhanced anchor box initialization.
            This refines anchor boxes using predicted mask shapes. Default: True.
        eval_size (tuple[int]|None): Fixed evaluation size (H, W) for anchor generation.
            If None, anchors are generated dynamically. Default: None.
        eval_idx (int): Decoder layer index for evaluation output. Negative values count
            from the end (-1 = last layer). Default: -1.
        eps (float): Small epsilon for numerical stability in anchor generation. Default: 1e-2.

    Note:
        The order prediction branch only operates on matching queries (excluding denoising
        queries) since denoising queries do not have ground truth reading order labels.

    Examples:
        .. code-block:: python

            model = DocLayoutV3Transformer(
                num_classes=25,
                hidden_dim=256,
                num_queries=300,
                num_decoder_layers=6
            )
            # Input: encoder features and mask features from neck
            feats = (enc_feats, mask_feat)
            # Output: logits, bboxes, masks, order_logits, and auxiliary outputs
            out = model(feats)
    """
    __shared__ = ['num_classes', 'hidden_dim', 'eval_size', 'num_prototypes']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 backbone_feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_prototypes=32,
                 num_levels=3,
                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.4,
                 box_noise_scale=0.4,
                 learnt_init_query=False,
                 query_pos_head_inv_sig=False,
                 mask_enhanced=True,
                 eval_size=None,
                 eval_idx=-1,
                 eps=1e-2):
        # Initialize parent class
        super(DocLayoutV3Transformer, self).__init__(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            position_embed_type=position_embed_type,
            backbone_feat_channels=backbone_feat_channels,
            feat_strides=feat_strides,
            num_prototypes=num_prototypes,
            num_levels=num_levels,
            num_decoder_points=num_decoder_points,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            num_denoising=num_denoising,
            label_noise_ratio=label_noise_ratio,
            box_noise_scale=box_noise_scale,
            learnt_init_query=learnt_init_query,
            query_pos_head_inv_sig=query_pos_head_inv_sig,
            mask_enhanced=mask_enhanced,
            eval_size=eval_size,
            eval_idx=eval_idx,
            eps=eps)

        # Override decoder with order-enabled version.
        # Note: parent's MaskTransformerDecoder (created by super().__init__) is
        # intentionally replaced here. This causes minor initialization overhead
        # but avoids modifying the parent class to add a factory method.
        decoder_layer = TransformerDecoderLayer(
            hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels,
            num_decoder_points)
        self.decoder = DocLayoutV3TransformerDecoder(
            hidden_dim, decoder_layer,
            num_decoder_layers, num_queries, eval_idx)

        from .utils import AntisymmetricPairwiseScorer

        # Create independent order prediction heads for each decoder layer
        # Each layer has its own Linear projection to allow different order
        # representations at different decoding stages (deep supervision)
        self.dec_order_head = nn.LayerList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_decoder_layers)
        ])
        # Global pointer converts query features into pairwise order logits
        # with antisymmetric constraint: logits[i,j] = -logits[j,i]
        self.dec_global_pointer = AntisymmetricPairwiseScorer(
            hidden_size=hidden_dim, head_size=64)

        # Initialize order head parameters with proper bias
        bias_cls = bias_init_with_prob(0.01)
        for order_head in self.dec_order_head:
            linear_init_(order_head)
            constant_(order_head.bias, bias_cls)

    def forward(self, feats, pad_mask=None, gt_meta=None, is_teacher=False):
        """
        Forward pass with reading order prediction.

        This method extends MaskRTDETR's forward pass to include reading order logits
        in the output. The order prediction branch operates in parallel with bbox,
        classification, and mask predictions.

        Args:
            feats (tuple): A 2-tuple of (encoder_features, mask_features).
                - encoder_features (list[Tensor]): Multi-scale features from backbone.
                    Each tensor has shape [batch_size, channels, H, W].
                - mask_features (Tensor): High-resolution features for mask prediction.
                    Shape: [batch_size, num_prototypes, H, W].
            pad_mask (Tensor|None): Padding mask for variable-size images. Default: None.
                Shape: [batch_size, H, W] with 1 for valid pixels, 0 for padding.
            gt_meta (dict|None): Ground truth metadata for denoising training, containing:
                - gt_class: Ground truth class labels
                - gt_bbox: Ground truth bounding boxes
                - gt_read_order: Ground truth reading order (for DocLayoutV3)
                Default: None (not used during inference).
            is_teacher (bool): Whether this forward pass is for a teacher model in
                knowledge distillation. Affects anchor generation. Default: False.

        Returns:
            tuple: An 8-tuple containing:
                - out_logits (Tensor): Classification logits from all decoder layers.
                    Shape: [num_layers, batch_size, num_queries, num_classes]
                - out_bboxes (Tensor): Predicted bounding boxes (sigmoid-activated).
                    Shape: [num_layers, batch_size, num_queries, 4] in (cx, cy, w, h) format.
                - out_masks (Tensor): Predicted instance masks.
                    Shape: [num_layers, batch_size, num_queries, mask_h, mask_w]
                - out_order_logits (Tensor): Pairwise reading order logits from all layers.
                    Shape: [num_layers, batch_size, num_queries, num_queries]
                    where order_logits[l, b, i, j] > 0 indicates element i comes before j.
                - enc_out (tuple): Encoder-level predictions for auxiliary loss.
                    (enc_logits, enc_bboxes, enc_masks) each with shape:
                    [batch_size, num_queries, ...].
                - init_out (tuple|None): Initial predictions before decoder refinement.
                    Same format as enc_out, only present during training with denoising.
                - enc_topk_order (None): Reserved for encoder-level order prediction
                    (currently not implemented, always returns None).
                - dn_meta (dict|None): Metadata for denoising training, including
                    attention masks and query counts. None during inference.

        Note:
            The out_order_logits tensor uses antisymmetric property: logits[i,j] = -logits[j,i].
            This ensures consistency in pairwise order relationships.
        """
        enc_feats, mask_feat = feats
        # input projection and embedding
        (memory, spatial_shapes,
         level_start_index) = self._get_encoder_input(enc_feats)

        # prepare denoising training
        if self.training:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_denoising_training_group(gt_meta,
                                             self.num_classes,
                                             self.num_queries,
                                             self.denoising_class_embed.weight,
                                             self.num_denoising,
                                             self.label_noise_ratio,
                                             self.box_noise_scale)
        else:
            denoising_class, denoising_bbox_unact,\
                attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_out, init_out, enc_topk_order = \
            self._get_decoder_input(
                memory, mask_feat, spatial_shapes,
                denoising_class, denoising_bbox_unact, is_teacher)

        # Decoder forward pass with order prediction
        # Order heads and global pointer are passed to decoder for layer-wise prediction
        out_bboxes, out_logits, out_masks, out_order_logits = self.decoder(
            target,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            level_start_index,
            mask_feat,
            self.bbox_head,
            self.score_head,
            self.dec_order_head,
            self.dec_global_pointer,
            self.query_pos_head,
            self.mask_query_head,
            self.dec_norm,
            attn_mask=attn_mask,
            memory_mask=None,
            query_pos_head_inv_sig=self.query_pos_head_inv_sig)

        return (out_logits, out_bboxes, out_masks, out_order_logits,
                enc_out, init_out, enc_topk_order, dn_meta)

    def _get_decoder_input(self,
                           memory,
                           mask_feat,
                           spatial_shapes,
                           denoising_class=None,
                           denoising_bbox_unact=None,
                           is_teacher=False):
        """Get decoder input, extending parent with enc_topk_order placeholder."""
        target, reference_points_unact, enc_out, init_out = \
            super()._get_decoder_input(
                memory, mask_feat, spatial_shapes,
                denoising_class, denoising_bbox_unact, is_teacher)
        # Order prediction happens at decoder level only; no encoder-level order.
        enc_topk_order = None
        return target, reference_points_unact, enc_out, init_out, enc_topk_order

