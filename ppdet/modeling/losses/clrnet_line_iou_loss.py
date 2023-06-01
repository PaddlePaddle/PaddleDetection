import paddle


def line_iou(pred, target, img_w, length=15, aligned=True):
    '''
    Calculate the line iou value between predictions and targets
    Args:
        pred: lane predictions, shape: (num_pred, 72)
        target: ground truth, shape: (num_target, 72)
        img_w: image width
        length: extended radius
        aligned: True for iou loss calculation, False for pair-wise ious in assign
    '''
    px1 = pred - length
    px2 = pred + length
    tx1 = target - length
    tx2 = target + length

    if aligned:
        invalid_mask = target
        ovr = paddle.minimum(px2, tx2) - paddle.maximum(px1, tx1)
        union = paddle.maximum(px2, tx2) - paddle.minimum(px1, tx1)
    else:
        num_pred = pred.shape[0]
        invalid_mask = target.tile([num_pred, 1, 1])

        ovr = (paddle.minimum(px2[:, None, :], tx2[None, ...]) - paddle.maximum(
            px1[:, None, :], tx1[None, ...]))
        union = (paddle.maximum(px2[:, None, :], tx2[None, ...]) -
                 paddle.minimum(px1[:, None, :], tx1[None, ...]))

    invalid_masks = (invalid_mask < 0) | (invalid_mask >= img_w)

    ovr[invalid_masks] = 0.
    union[invalid_masks] = 0.
    iou = ovr.sum(axis=-1) / (union.sum(axis=-1) + 1e-9)
    return iou


def liou_loss(pred, target, img_w, length=15):
    return (1 - line_iou(pred, target, img_w, length)).mean()
