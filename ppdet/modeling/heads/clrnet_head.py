import math
import paddle
import numpy as np
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register

from ppdet.modeling.initializer import normal_
from ppdet.modeling.lane_utils import Lane
from ppdet.modeling.losses import line_iou
from ppdet.modeling.clrnet_utils import ROIGather, LinearModule, SegDecoder

__all__ = ['CLRHead']


@register
class CLRHead(nn.Layer):
    __inject__ = ['loss']
    __shared__ = [
        'img_w', 'img_h', 'ori_img_h', 'num_classes', 'cut_height',
        'num_points', "max_lanes"
    ]

    def __init__(self,
                 num_points=72,
                 prior_feat_channels=64,
                 fc_hidden_dim=64,
                 num_priors=192,
                 img_w=800,
                 img_h=320,
                 ori_img_h=590,
                 cut_height=270,
                 num_classes=5,
                 num_fc=2,
                 refine_layers=3,
                 sample_points=36,
                 conf_threshold=0.4,
                 nms_thres=0.5,
                 max_lanes=4,
                 loss='CLRNetLoss'):
        super(CLRHead, self).__init__()
        self.img_w = img_w
        self.img_h = img_h
        self.n_strips = num_points - 1
        self.n_offsets = num_points
        self.num_priors = num_priors
        self.sample_points = sample_points
        self.refine_layers = refine_layers
        self.num_classes = num_classes
        self.fc_hidden_dim = fc_hidden_dim
        self.ori_img_h = ori_img_h
        self.cut_height = cut_height
        self.conf_threshold = conf_threshold
        self.nms_thres = nms_thres
        self.max_lanes = max_lanes
        self.prior_feat_channels = prior_feat_channels
        self.loss = loss
        self.register_buffer(
            name='sample_x_indexs',
            tensor=(paddle.linspace(
                start=0, stop=1, num=self.sample_points,
                dtype=paddle.float32) * self.n_strips).astype(dtype='int64'))
        self.register_buffer(
            name='prior_feat_ys',
            tensor=paddle.flip(
                x=(1 - self.sample_x_indexs.astype('float32') / self.n_strips),
                axis=[-1]))
        self.register_buffer(
            name='prior_ys',
            tensor=paddle.linspace(
                start=1, stop=0, num=self.n_offsets).astype('float32'))
        self.prior_feat_channels = prior_feat_channels
        self._init_prior_embeddings()
        init_priors, priors_on_featmap = self.generate_priors_from_embeddings()
        self.register_buffer(name='priors', tensor=init_priors)
        self.register_buffer(name='priors_on_featmap', tensor=priors_on_featmap)
        self.seg_decoder = SegDecoder(self.img_h, self.img_w, self.num_classes,
                                      self.prior_feat_channels,
                                      self.refine_layers)
        reg_modules = list()
        cls_modules = list()
        for _ in range(num_fc):
            reg_modules += [*LinearModule(self.fc_hidden_dim)]
            cls_modules += [*LinearModule(self.fc_hidden_dim)]
        self.reg_modules = nn.LayerList(sublayers=reg_modules)
        self.cls_modules = nn.LayerList(sublayers=cls_modules)
        self.roi_gather = ROIGather(self.prior_feat_channels, self.num_priors,
                                    self.sample_points, self.fc_hidden_dim,
                                    self.refine_layers)
        self.reg_layers = nn.Linear(
            in_features=self.fc_hidden_dim,
            out_features=self.n_offsets + 1 + 2 + 1,
            bias_attr=True)
        self.cls_layers = nn.Linear(
            in_features=self.fc_hidden_dim, out_features=2, bias_attr=True)
        self.init_weights()

    def init_weights(self):
        for m in self.cls_layers.parameters():
            normal_(m, mean=0.0, std=0.001)
        for m in self.reg_layers.parameters():
            normal_(m, mean=0.0, std=0.001)

    def pool_prior_features(self, batch_features, num_priors, prior_xs):
        """
        pool prior feature from feature map.
        Args:
            batch_features (Tensor): Input feature maps, shape: (B, C, H, W) 
        """
        batch_size = batch_features.shape[0]
        prior_xs = prior_xs.reshape([batch_size, num_priors, -1, 1])

        prior_ys = self.prior_feat_ys.tile(repeat_times=[
            batch_size * num_priors
        ]).reshape([batch_size, num_priors, -1, 1])
        prior_xs = prior_xs * 2.0 - 1.0
        prior_ys = prior_ys * 2.0 - 1.0
        grid = paddle.concat(x=(prior_xs, prior_ys), axis=-1)
        feature = F.grid_sample(
            x=batch_features, grid=grid,
            align_corners=True).transpose(perm=[0, 2, 1, 3])
        feature = feature.reshape([
            batch_size * num_priors, self.prior_feat_channels,
            self.sample_points, 1
        ])
        return feature

    def generate_priors_from_embeddings(self):
        predictions = self.prior_embeddings.weight
        # 2 scores, 1 start_y, 1 start_x, 1 theta, 1 length, 72 coordinates, score[0] = negative prob, score[1] = positive prob       
        priors = paddle.zeros(
            (self.num_priors, 2 + 2 + 2 + self.n_offsets),
            dtype=predictions.dtype)
        priors[:, 2:5] = predictions.clone()
        priors[:, 6:] = (
            priors[:, 3].unsqueeze(1).clone().tile([1, self.n_offsets]) *
            (self.img_w - 1) +
            ((1 - self.prior_ys.tile([self.num_priors, 1]) -
              priors[:, 2].unsqueeze(1).clone().tile([1, self.n_offsets])) *
             self.img_h / paddle.tan(x=priors[:, 4].unsqueeze(1).clone().tile(
                 [1, self.n_offsets]) * math.pi + 1e-05))) / (self.img_w - 1)
        priors_on_featmap = paddle.index_select(
            priors, 6 + self.sample_x_indexs, axis=-1)
        return priors, priors_on_featmap

    def _init_prior_embeddings(self):
        self.prior_embeddings = nn.Embedding(self.num_priors, 3)
        bottom_priors_nums = self.num_priors * 3 // 4
        left_priors_nums, _ = self.num_priors // 8, self.num_priors // 8
        strip_size = 0.5 / (left_priors_nums // 2 - 1)
        bottom_strip_size = 1 / (bottom_priors_nums // 4 + 1)

        with paddle.no_grad():
            for i in range(left_priors_nums):
                self.prior_embeddings.weight[i, 0] = i // 2 * strip_size
                self.prior_embeddings.weight[i, 1] = 0.0
                self.prior_embeddings.weight[i,
                                             2] = 0.16 if i % 2 == 0 else 0.32

            for i in range(left_priors_nums,
                           left_priors_nums + bottom_priors_nums):
                self.prior_embeddings.weight[i, 0] = 0.0
                self.prior_embeddings.weight[i, 1] = (
                    (i - left_priors_nums) // 4 + 1) * bottom_strip_size
                self.prior_embeddings.weight[i, 2] = 0.2 * (i % 4 + 1)

            for i in range(left_priors_nums + bottom_priors_nums,
                           self.num_priors):
                self.prior_embeddings.weight[i, 0] = (
                    i - left_priors_nums - bottom_priors_nums) // 2 * strip_size
                self.prior_embeddings.weight[i, 1] = 1.0
                self.prior_embeddings.weight[i,
                                             2] = 0.68 if i % 2 == 0 else 0.84

    def forward(self, x, inputs=None):
        """
        Take pyramid features as input to perform Cross Layer Refinement and finally output the prediction lanes.
        Each feature is a 4D tensor.
        Args:
            x: input features (list[Tensor])
        Return:
            prediction_list: each layer's prediction result
            seg: segmentation result for auxiliary loss
        """
        batch_features = list(x[len(x) - self.refine_layers:])
        batch_features.reverse()
        batch_size = batch_features[-1].shape[0]

        if self.training:
            self.priors, self.priors_on_featmap = self.generate_priors_from_embeddings(
            )
        priors, priors_on_featmap = self.priors.tile(
            [batch_size, 1,
             1]), self.priors_on_featmap.tile([batch_size, 1, 1])
        predictions_lists = []
        prior_features_stages = []

        for stage in range(self.refine_layers):
            num_priors = priors_on_featmap.shape[1]
            prior_xs = paddle.flip(x=priors_on_featmap, axis=[2])
            batch_prior_features = self.pool_prior_features(
                batch_features[stage], num_priors, prior_xs)
            prior_features_stages.append(batch_prior_features)

            fc_features = self.roi_gather(prior_features_stages,
                                          batch_features[stage], stage)
            # return fc_features
            fc_features = fc_features.reshape(
                [num_priors, batch_size, -1]).reshape(
                    [batch_size * num_priors, self.fc_hidden_dim])
            cls_features = fc_features.clone()
            reg_features = fc_features.clone()

            for cls_layer in self.cls_modules:
                cls_features = cls_layer(cls_features)

            # return cls_features
            for reg_layer in self.reg_modules:
                reg_features = reg_layer(reg_features)
            cls_logits = self.cls_layers(cls_features)
            reg = self.reg_layers(reg_features)

            cls_logits = cls_logits.reshape(
                [batch_size, -1, cls_logits.shape[1]])
            reg = reg.reshape([batch_size, -1, reg.shape[1]])
            predictions = priors.clone()
            predictions[:, :, :2] = cls_logits
            predictions[:, :, 2:5] += reg[:, :, :3]
            predictions[:, :, 5] = reg[:, :, 3]

            def tran_tensor(t):
                return t.unsqueeze(axis=2).clone().tile([1, 1, self.n_offsets])

            predictions[..., 6:] = (
                tran_tensor(predictions[..., 3]) * (self.img_w - 1) +
                ((1 - self.prior_ys.tile([batch_size, num_priors, 1]) -
                  tran_tensor(predictions[..., 2])) * self.img_h / paddle.tan(
                      tran_tensor(predictions[..., 4]) * math.pi + 1e-05))) / (
                          self.img_w - 1)

            prediction_lines = predictions.clone()
            predictions[..., 6:] += reg[..., 4:]
            predictions_lists.append(predictions)

            if stage != self.refine_layers - 1:
                priors = prediction_lines.detach().clone()
                priors_on_featmap = priors.index_select(
                    6 + self.sample_x_indexs, axis=-1)

        if self.training:
            seg = None
            seg_features = paddle.concat(
                [
                    F.interpolate(
                        feature,
                        size=[
                            batch_features[-1].shape[2],
                            batch_features[-1].shape[3]
                        ],
                        mode='bilinear',
                        align_corners=False) for feature in batch_features
                ],
                axis=1)

            seg = self.seg_decoder(seg_features)

            output = {'predictions_lists': predictions_lists, 'seg': seg}
            return self.loss(output, inputs)
        return predictions_lists[-1]

    def predictions_to_pred(self, predictions):
        """
        Convert predictions to internal Lane structure for evaluation.
        """
        self.prior_ys = paddle.to_tensor(self.prior_ys)
        self.prior_ys = self.prior_ys.astype('float64')
        lanes = []
        for lane in predictions:
            lane_xs = lane[6:].clone()
            start = min(
                max(0, int(round(lane[2].item() * self.n_strips))),
                self.n_strips)
            length = int(round(lane[5].item()))
            end = start + length - 1
            end = min(end, len(self.prior_ys) - 1)
            if start > 0:
                mask = ((lane_xs[:start] >= 0.) &
                        (lane_xs[:start] <= 1.)).cpu().detach().numpy()[::-1]
                mask = ~((mask.cumprod()[::-1]).astype(np.bool_))
                lane_xs[:start][mask] = -2
            if end < len(self.prior_ys) - 1:
                lane_xs[end + 1:] = -2

            lane_ys = self.prior_ys[lane_xs >= 0].clone()
            lane_xs = lane_xs[lane_xs >= 0]
            lane_xs = lane_xs.flip(axis=0).astype('float64')
            lane_ys = lane_ys.flip(axis=0)

            lane_ys = (lane_ys *
                       (self.ori_img_h - self.cut_height) + self.cut_height
                       ) / self.ori_img_h
            if len(lane_xs) <= 1:
                continue
            points = paddle.stack(
                x=(lane_xs.reshape([-1, 1]), lane_ys.reshape([-1, 1])),
                axis=1).squeeze(axis=2)
            lane = Lane(
                points=points.cpu().numpy(),
                metadata={
                    'start_x': lane[3],
                    'start_y': lane[2],
                    'conf': lane[1]
                })
            lanes.append(lane)
        return lanes

    def lane_nms(self, predictions, scores, nms_overlap_thresh, top_k):
        """
        NMS for lane detection.
        predictions: paddle.Tensor [num_lanes,conf,y,x,lenght,72offsets] [12,77]
        scores: paddle.Tensor [num_lanes]
        nms_overlap_thresh: float
        top_k: int
        """
        # sort by scores to get idx
        idx = scores.argsort(descending=True)
        keep = []

        condidates = predictions.clone()
        condidates = condidates.index_select(idx)

        while len(condidates) > 0:
            keep.append(idx[0])
            if len(keep) >= top_k or len(condidates) == 1:
                break

            ious = []
            for i in range(1, len(condidates)):
                ious.append(1 - line_iou(
                    condidates[i].unsqueeze(0),
                    condidates[0].unsqueeze(0),
                    img_w=self.img_w,
                    length=15))
            ious = paddle.to_tensor(ious)

            mask = ious <= nms_overlap_thresh
            id = paddle.where(mask == False)[0]

            if id.shape[0] == 0:
                break
            condidates = condidates[1:].index_select(id)
            idx = idx[1:].index_select(id)
        keep = paddle.stack(keep)

        return keep

    def get_lanes(self, output, as_lanes=True):
        """
        Convert model output to lanes.
        """
        softmax = nn.Softmax(axis=1)
        decoded = []

        for predictions in output:
            threshold = self.conf_threshold
            scores = softmax(predictions[:, :2])[:, 1]
            keep_inds = scores >= threshold
            predictions = predictions[keep_inds]
            scores = scores[keep_inds]

            if predictions.shape[0] == 0:
                decoded.append([])
                continue
            nms_predictions = predictions.detach().clone()
            nms_predictions = paddle.concat(
                x=[nms_predictions[..., :4], nms_predictions[..., 5:]], axis=-1)

            nms_predictions[..., 4] = nms_predictions[..., 4] * self.n_strips
            nms_predictions[..., 5:] = nms_predictions[..., 5:] * (
                self.img_w - 1)

            keep = self.lane_nms(
                nms_predictions[..., 5:],
                scores,
                nms_overlap_thresh=self.nms_thres,
                top_k=self.max_lanes)

            predictions = predictions.index_select(keep)

            if predictions.shape[0] == 0:
                decoded.append([])
                continue
            predictions[:, 5] = paddle.round(predictions[:, 5] * self.n_strips)
            if as_lanes:
                pred = self.predictions_to_pred(predictions)
            else:
                pred = predictions
            decoded.append(pred)
        return decoded
