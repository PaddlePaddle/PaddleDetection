import numpy as np
import sys
import time
from shapely.geometry import Polygon
import paddle
import unittest

from ext_op import rbox_iou


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
    poly = np.array([x0, y0, x1, y1, x2, y2, x3, y3], dtype=np.float64)
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
    if not g.is_valid or not p.is_valid:
        return 0

    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter / union


def rbox_overlaps(anchors, gt_bboxes, use_cv2=False):
    """

    Args:
        anchors: [NA, 5]  x1,y1,x2,y2,angle
        gt_bboxes: [M, 5]  x1,y1,x2,y2,angle

    Returns:
        iou: [NA, M]
    """
    assert anchors.shape[1] == 5
    assert gt_bboxes.shape[1] == 5

    gt_bboxes_ploy = [rbox2poly_single(e) for e in gt_bboxes]
    anchors_ploy = [rbox2poly_single(e) for e in anchors]

    num_gt, num_anchors = len(gt_bboxes_ploy), len(anchors_ploy)
    iou = np.zeros((num_anchors, num_gt), dtype=np.float64)

    start_time = time.time()
    for i in range(num_anchors):
        for j in range(num_gt):
            try:
                iou[i, j] = intersection(anchors_ploy[i], gt_bboxes_ploy[j])
            except Exception as e:
                print('cur anchors_ploy[i]', anchors_ploy[i],
                      'gt_bboxes_ploy[j]', gt_bboxes_ploy[j], e)
    return iou


def gen_sample(n):
    rbox = np.random.rand(n, 5)
    rbox[:, 0:4] = rbox[:, 0:4] * 0.45 + 0.001
    rbox[:, 4] = rbox[:, 4] - 0.5
    return rbox


class RBoxIoUTest(unittest.TestCase):
    def setUp(self):
        self.initTestCase()
        self.rbox1 = gen_sample(self.n)
        self.rbox2 = gen_sample(self.m)

    def initTestCase(self):
        self.n = 13000
        self.m = 7

    def assertAllClose(self, x, y, msg, atol=5e-1, rtol=1e-2):
        self.assertTrue(np.allclose(x, y, atol=atol, rtol=rtol), msg=msg)

    def get_places(self):
        places = [paddle.CPUPlace()]
        if paddle.device.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))

        return places

    def check_output(self, place):
        paddle.disable_static()
        pd_rbox1 = paddle.to_tensor(self.rbox1, place=place)
        pd_rbox2 = paddle.to_tensor(self.rbox2, place=place)
        actual_t = rbox_iou(pd_rbox1, pd_rbox2).numpy()
        poly_rbox1 = self.rbox1
        poly_rbox2 = self.rbox2
        poly_rbox1[:, 0:4] = self.rbox1[:, 0:4] * 1024
        poly_rbox2[:, 0:4] = self.rbox2[:, 0:4] * 1024
        expect_t = rbox_overlaps(poly_rbox1, poly_rbox2, use_cv2=False)
        self.assertAllClose(
            actual_t,
            expect_t,
            msg="rbox_iou has diff at {} \nExpect {}\nBut got {}".format(
                str(place), str(expect_t), str(actual_t)))

    def test_output(self):
        places = self.get_places()
        for place in places:
            self.check_output(place)


if __name__ == "__main__":
    unittest.main()
