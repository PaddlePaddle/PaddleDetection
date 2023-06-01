import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register
from ppdet.modeling.clrnet_utils import accuracy
from ppdet.modeling.assigners.clrnet_assigner import assign
from ppdet.modeling.losses.clrnet_line_iou_loss import liou_loss

__all__ = ['CLRNetLoss']


class SoftmaxFocalLoss(nn.Layer):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = paddle.pow(1. - scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss


def focal_loss(input: paddle.Tensor,
               target: paddle.Tensor,
               alpha: float,
               gamma: float=2.0,
               reduction: str='none',
               eps: float=1e-8) -> paddle.Tensor:
    r"""Function that computes Focal loss.

    See :class:`~kornia.losses.FocalLoss` for details.
    """
    if not paddle.is_tensor(input):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(input)))

    if not len(input.shape) >= 2:
        raise ValueError("Invalid input shape, we expect BxCx*. Got: {}".format(
            input.shape))

    if input.shape[0] != target.shape[0]:
        raise ValueError(
            'Expected input batch_size ({}) to match target batch_size ({}).'.
            format(input.shape[0], target.shape[0]))

    n = input.shape[0]
    out_size = (n, ) + tuple(input.shape[2:])
    if target.shape[1:] != input.shape[2:]:
        raise ValueError('Expected target size {}, got {}'.format(out_size,
                                                                  target.shape))
    if (isinstance(input.place, paddle.CUDAPlace) and
            isinstance(target.place, paddle.CPUPlace)) | (isinstance(
                input.place, paddle.CPUPlace) and isinstance(target.place,
                                                             paddle.CUDAPlace)):
        raise ValueError(
            "input and target must be in the same device. Got: {} and {}".
            format(input.place, target.place))

    # compute softmax over the classes axis
    input_soft: paddle.Tensor = F.softmax(input, axis=1) + eps

    # create the labels one hot tensor
    target_one_hot: paddle.Tensor = paddle.to_tensor(
        F.one_hot(
            target, num_classes=input.shape[1]).cast(input.dtype),
        place=input.place)

    # compute the actual focal loss
    weight = paddle.pow(-input_soft + 1., gamma)

    focal = -alpha * weight * paddle.log(input_soft)
    loss_tmp = paddle.sum(target_one_hot * focal, axis=1)

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = paddle.mean(loss_tmp)
    elif reduction == 'sum':
        loss = paddle.sum(loss_tmp)
    else:
        raise NotImplementedError("Invalid reduction mode: {}".format(
            reduction))
    return loss


class FocalLoss(nn.Layer):
    r"""Criterion that computes Focal loss.

    According to [1], the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    where:
       - :math:`p_t` is the model's estimated probability for each class.


    Arguments:
        alpha (float): Weighting factor :math:`\alpha \in [0, 1]`.
        gamma (float): Focusing parameter :math:`\gamma >= 0`.
        reduction (str, optional): Specifies the reduction to apply to the
         output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
         in the output, ‘sum’: the output will be summed. Default: ‘none’.

    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> loss = kornia.losses.FocalLoss(**kwargs)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()

    References:
        [1] https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha: float, gamma: float=2.0,
                 reduction: str='none') -> None:
        super(FocalLoss, self).__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.eps: float = 1e-6

    def forward(  # type: ignore
            self, input: paddle.Tensor, target: paddle.Tensor) -> paddle.Tensor:
        return focal_loss(input, target, self.alpha, self.gamma, self.reduction,
                          self.eps)


@register
class CLRNetLoss(nn.Layer):
    __shared__ = ['img_w', 'img_h', 'num_classes', 'num_points']

    def __init__(self,
                 cls_loss_weight=2.0,
                 xyt_loss_weight=0.2,
                 iou_loss_weight=2.0,
                 seg_loss_weight=1.0,
                 refine_layers=3,
                 num_points=72,
                 img_w=800,
                 img_h=320,
                 num_classes=5,
                 ignore_label=255,
                 bg_weight=0.4):
        super(CLRNetLoss, self).__init__()
        self.cls_loss_weight = cls_loss_weight
        self.xyt_loss_weight = xyt_loss_weight
        self.iou_loss_weight = iou_loss_weight
        self.seg_loss_weight = seg_loss_weight
        self.refine_layers = refine_layers
        self.img_w = img_w
        self.img_h = img_h
        self.n_strips = num_points - 1
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        weights = paddle.ones(shape=[self.num_classes])
        weights[0] = bg_weight
        self.criterion = nn.NLLLoss(
            ignore_index=self.ignore_label, weight=weights)

    def forward(self, output, batch):
        predictions_lists = output['predictions_lists']
        targets = batch['lane_line'].clone()
        cls_criterion = FocalLoss(alpha=0.25, gamma=2.0)
        cls_loss = paddle.to_tensor(0.0)
        reg_xytl_loss = paddle.to_tensor(0.0)
        iou_loss = paddle.to_tensor(0.0)
        cls_acc = []
        cls_acc_stage = []
        for stage in range(self.refine_layers):
            predictions_list = predictions_lists[stage]
            for predictions, target in zip(predictions_list, targets):
                target = target[target[:, 1] == 1]

                if len(target) == 0:
                    # If there are no targets, all predictions have to be negatives (i.e., 0 confidence)
                    cls_target = paddle.zeros(
                        [predictions.shape[0]], dtype='int64')
                    cls_pred = predictions[:, :2]
                    cls_loss = cls_loss + cls_criterion(cls_pred,
                                                        cls_target).sum()
                    continue

                with paddle.no_grad():
                    matched_row_inds, matched_col_inds = assign(
                        predictions, target, self.img_w, self.img_h)

                # classification targets
                cls_target = paddle.zeros([predictions.shape[0]], dtype='int64')
                cls_target[matched_row_inds] = 1
                cls_pred = predictions[:, :2]

                # regression targets -> [start_y, start_x, theta] (all transformed to absolute values), only on matched pairs
                reg_yxtl = predictions.index_select(matched_row_inds)[..., 2:6]

                reg_yxtl[:, 0] *= self.n_strips
                reg_yxtl[:, 1] *= (self.img_w - 1)
                reg_yxtl[:, 2] *= 180
                reg_yxtl[:, 3] *= self.n_strips

                target_yxtl = target.index_select(matched_col_inds)[..., 2:
                                                                    6].clone()

                # regression targets -> S coordinates (all transformed to absolute values)
                reg_pred = predictions.index_select(matched_row_inds)[..., 6:]
                reg_pred *= (self.img_w - 1)
                reg_targets = target.index_select(matched_col_inds)[...,
                                                                    6:].clone()

                with paddle.no_grad():
                    predictions_starts = paddle.clip(
                        (predictions.index_select(matched_row_inds)[..., 2] *
                         self.n_strips).round().cast("int64"),
                        min=0,
                        max=self.
                        n_strips)  # ensure the predictions starts is valid

                    target_starts = (
                        target.index_select(matched_col_inds)[..., 2] *
                        self.n_strips).round().cast("int64")
                    target_yxtl[:, -1] -= (
                        predictions_starts - target_starts)  # reg length

                # Loss calculation
                cls_loss = cls_loss + cls_criterion(
                    cls_pred, cls_target).sum() / target.shape[0]

                target_yxtl[:, 0] *= self.n_strips
                target_yxtl[:, 2] *= 180

                reg_xytl_loss = reg_xytl_loss + F.smooth_l1_loss(
                    input=reg_yxtl, label=target_yxtl, reduction='none').mean()

                iou_loss = iou_loss + liou_loss(
                    reg_pred, reg_targets, self.img_w, length=15)

                cls_accuracy = accuracy(cls_pred, cls_target)
                cls_acc_stage.append(cls_accuracy)

            cls_acc.append(sum(cls_acc_stage) / (len(cls_acc_stage) + 1e-5))

        # extra segmentation loss
        seg_loss = self.criterion(
            F.log_softmax(
                output['seg'], axis=1), batch['seg'].cast('int64'))

        cls_loss /= (len(targets) * self.refine_layers)
        reg_xytl_loss /= (len(targets) * self.refine_layers)
        iou_loss /= (len(targets) * self.refine_layers)

        loss = cls_loss * self.cls_loss_weight \
            + reg_xytl_loss * self.xyt_loss_weight \
            + seg_loss * self.seg_loss_weight \
            + iou_loss * self.iou_loss_weight

        return_value = {
            'loss': loss,
            'cls_loss': cls_loss * self.cls_loss_weight,
            'reg_xytl_loss': reg_xytl_loss * self.xyt_loss_weight,
            'seg_loss': seg_loss * self.seg_loss_weight,
            'iou_loss': iou_loss * self.iou_loss_weight
        }

        for i in range(self.refine_layers):
            if not isinstance(cls_acc[i], paddle.Tensor):
                cls_acc[i] = paddle.to_tensor(cls_acc[i])
            return_value['stage_{}_acc'.format(i)] = cls_acc[i]

        return return_value
