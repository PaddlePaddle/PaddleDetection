import numpy as np
import os
import sys
import cv2
import time
import shapely
from shapely.geometry import Polygon

import paddle
from paddle.utils.cpp_extension import load

paddle.set_device('gpu:0')
paddle.disable_static()

custom_ops = load(
    name="custom_jit_ops", sources=["rbox_iou_op.cc", "rbox_iou_op.cu"])

# generate random data
rbox1 = np.random.rand(13000, 5)
rbox2 = np.random.rand(7, 5)

# x1 y1 w h [0, 0.5]
rbox1[:, 0:4] = rbox1[:, 0:4] * 0.45 + 0.001
rbox2[:, 0:4] = rbox2[:, 0:4] * 0.45 + 0.001

# generate rbox
rbox1[:, 4] = rbox1[:, 4] - 0.5
rbox2[:, 4] = rbox2[:, 4] - 0.5

print('rbox1', rbox1.shape, 'rbox2', rbox2.shape)

# to paddle tensor
pd_rbox1 = paddle.to_tensor(rbox1)
pd_rbox2 = paddle.to_tensor(rbox2)

iou = custom_ops.rbox_iou(pd_rbox1, pd_rbox2)
print('paddle time:', time.time() - start_time)
print('iou is', iou.cpu().shape)


# get gt
def rbox2poly_single(rrect, get_best_begin_point=False):
    """
    rrect:[x_ctr,y_ctr,w,h,angle]
    to
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    """
    x_ctr, y_ctr, width, height, angle = rrect[:5]
    tl_x, tl_y, br_x, br_y = -width / 2, -height / 2, width / 2, height / 2
    # rect 2x4
    rect = np.array([[tl_x, br_x, br_x, tl_x], [tl_y, tl_y, br_y, br_y]])
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    # poly
    poly = R.dot(rect)
    x0, x1, x2, x3 = poly[0, :4] + x_ctr
    y0, y1, y2, y3 = poly[1, :4] + y_ctr
    poly = np.array([x0, y0, x1, y1, x2, y2, x3, y3], dtype=np.float32)
    return poly


def intersection(g, p):
    """
    Intersection.
    """

    g = g[:8].reshape((4, 2))
    p = p[:8].reshape((4, 2))

    a = g
    b = p

    use_filter = True
    if use_filter:
        # step1:
        inter_x1 = np.maximum(np.min(a[:, 0]), np.min(b[:, 0]))
        inter_x2 = np.minimum(np.max(a[:, 0]), np.max(b[:, 0]))
        inter_y1 = np.maximum(np.min(a[:, 1]), np.min(b[:, 1]))
        inter_y2 = np.minimum(np.max(a[:, 1]), np.max(b[:, 1]))
        if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
            return 0.
        x1 = np.minimum(np.min(a[:, 0]), np.min(b[:, 0]))
        x2 = np.maximum(np.max(a[:, 0]), np.max(b[:, 0]))
        y1 = np.minimum(np.min(a[:, 1]), np.min(b[:, 1]))
        y2 = np.maximum(np.max(a[:, 1]), np.max(b[:, 1]))
        if x1 >= x2 or y1 >= y2 or (x2 - x1) < 2 or (y2 - y1) < 2:
            return 0.

    g = Polygon(g)
    p = Polygon(p)
    #g = g.buffer(0)
    #p = p.buffer(0)
    if not g.is_valid or not p.is_valid:
        return 0

    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter / union


# rbox_iou by python
def rbox_overlaps(anchors, gt_bboxes, use_cv2=False):
    """

    Args:
        anchors: [NA, 5]  x1,y1,x2,y2,angle
        gt_bboxes: [M, 5]  x1,y1,x2,y2,angle

    Returns:

    """
    assert anchors.shape[1] == 5
    assert gt_bboxes.shape[1] == 5

    gt_bboxes_ploy = [rbox2poly_single(e) for e in gt_bboxes]
    anchors_ploy = [rbox2poly_single(e) for e in anchors]

    num_gt, num_anchors = len(gt_bboxes_ploy), len(anchors_ploy)
    iou = np.zeros((num_gt, num_anchors), dtype=np.float32)

    start_time = time.time()
    for i in range(num_gt):
        for j in range(num_anchors):
            try:
                iou[i, j] = intersection(gt_bboxes_ploy[i], anchors_ploy[j])
            except Exception as e:
                print('cur gt_bboxes_ploy[i]', gt_bboxes_ploy[i],
                      'anchors_ploy[j]', anchors_ploy[j], e)
    iou = iou.T
    print('intersection  all sp_time', time.time() - start_time)
    return iou


# make coor as int
ploy_rbox1[:, 0:4] = rbox1[:, 0:4] * 1024
ploy_rbox2[:, 0:4] = rbox2[:, 0:4] * 1024

start_time = time.time()
iou_py = rbox_overlaps(ploy_rbox1, ploy_rbox2, use_cv2=False)
print('rbox time', time.time() - start_time)
print(iou_py.shape)

iou_pd = iou.cpu().numpy()
print('diff sum', np.sum(np.abs(iou_pd - iou_py)))
diff1 = (iou_pd - iou_py) / (iou_py + 1e-8)
print('diff1:', np.sum(np.abs(diff1)))
