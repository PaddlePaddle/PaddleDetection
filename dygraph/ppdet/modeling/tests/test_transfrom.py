from __future__ import print_function
import random
import unittest
import numpy as np
import copy
# add python path of PadleDetection to sys.path
import os
import sys
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 4)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

from ppdet.data.transform import *


def gen_sample(h, w, nt, nc, random_score=True, channel_first=False):
    im = np.random.randint(0, 256, size=(h, w, 3)).astype('float32')
    if channel_first:
        im = im.transpose((2, 0, 1))
    gt_bbox = np.random.random(size=(nt, 4)).astype('float32')
    gt_class = np.random.randint(0, nc, size=(nt, 1)).astype('int32')
    if random_score:
        gt_score = np.random.random(size=(nt, 1))
    else:
        gt_score = np.ones(shape=(nt, 1)).astype('float32')
    is_crowd = np.zeros_like(gt_class)
    sample = {
        'image': im,
        'gt_bbox': gt_bbox,
        'gt_class': gt_class,
        'gt_score': gt_score,
        'is_crowd': is_crowd,
        'h': h,
        'w': w
    }
    return sample


class TestTransformOp(unittest.TestCase):
    def setUp(self):
        self.h, self.w = np.random.randint(1, 1024, size=2)
        self.nt = np.random.randint(1, 50)
        self.nc = 80

    def assertAllClose(self, x, y, msg, atol=1e-5, rtol=1e-3):
        self.assertTrue(np.allclose(x, y, atol=atol, rtol=rtol), msg=msg)


class TestResizeOp(TestTransformOp):
    def test_resize(self):
        sample = gen_sample(self.h, self.w, self.nt, self.nc)
        orig_op = Resize(target_dim=608, interp=2)
        curr_op = ResizeOp(target_size=608, keep_ratio=False, interp=2)
        orig_res = orig_op(copy.deepcopy(sample))
        curr_res = curr_op(copy.deepcopy(sample))
        fields = ['image', 'gt_bbox', 'gt_class', 'gt_score']
        for k in fields:
            self.assertAllClose(orig_res[k], curr_res[k], msg=k)


# only for specified random seed
# class TestMixupOp(TestTransformOp):
#     def setUp(self):
#         self.h, self.w = np.random.randint(1024, size=2)
#         self.nt = np.random.randint(50)
#         self.nc = 80

#     def test_mixup(self):
#         curr_sample = [gen_sample(self.h, self.w, self.nt, self.nc) for _ in range(2)]
#         orig_sample = copy.deepcopy(curr_sample[0])
#         orig_sample['mixup'] = copy.deepcopy(curr_sample[1])
#         orig_op = MixupImage(alpha=1.5, beta=1.5)
#         curr_op = MixupOp(alpha=1.5, beta=1.5)
#         orig_res = orig_op(orig_sample)
#         curr_res = curr_op(curr_sample)
#         fields = ['image', 'gt_bbox', 'gt_class', 'gt_score']
#         for k in fields:
#             self.assertAllClose(orig_res[k], curr_res[k], msg=k)

# only for specified random seed
# class TestRandomDistortOp(TestTransformOp):

#     def test_random_distort(self):
#         sample = gen_sample(self.h, self.w, self.nt, self.nc)
#         orig_op = ColorDistort(hsv_format=True, random_apply=False)
#         curr_op = RandomDistortOp(random_apply=False)
#         orig_res = orig_op(copy.deepcopy(sample))
#         curr_res = curr_op(copy.deepcopy(sample))
#         fields = ['image', 'gt_bbox', 'gt_class', 'gt_score']
#         for k in fields:
#             self.assertAllClose(orig_res[k], curr_res[k], msg=k)

# only for specified random seed
# class TestRandomExpandOp(TestTransformOp):

#     def test_random_expand(self):
#         sample = gen_sample(self.h, self.w, self.nt, self.nc)
#         orig_op = RandomExpand(fill_value=[123.675, 116.28, 103.53])
#         curr_op = RandomExpandOp(fill_value=[123.675, 116.28, 103.53])
#         orig_res = orig_op(copy.deepcopy(sample))
#         curr_res = curr_op(copy.deepcopy(sample))
#         fields = ['image', 'gt_bbox', 'gt_class', 'gt_score']
#         for k in fields:
#             self.assertAllClose(orig_res[k], curr_res[k], msg=k)

# only for specified random seed
# class TestRandomCropOp(TestTransformOp):

#     def test_random_crop(self):
#         sample = gen_sample(self.h, self.w, self.nt, self.nc)
#         orig_op = RandomCrop()
#         curr_op = RandomCropOp()
#         orig_res = orig_op(copy.deepcopy(sample))
#         curr_res = curr_op(copy.deepcopy(sample))
#         fields = ['image', 'gt_bbox', 'gt_class', 'gt_score']
#         for k in fields:
#             self.assertAllClose(orig_res[k], curr_res[k], msg=k)

# only for specified random seed
# class TestRandomFlipOp(TestTransformOp):

#     def test_random_flip(self):
#         sample = gen_sample(self.h, self.w, self.nt, self.nc)
#         orig_op = RandomFlipImage(is_normalized=False)
#         curr_op = RandomFlipOp()
#         orig_res = orig_op(copy.deepcopy(sample))
#         curr_res = curr_op(copy.deepcopy(sample))
#         fields = ['image', 'gt_bbox', 'gt_class', 'gt_score']
#         for k in fields:
#             self.assertAllClose(orig_res[k], curr_res[k], msg=k)

# only for specified random seed
# class TestBatchRandomResizeOp(TestTransformOp):

#     def test_batch_random_resize(self):
#         sample = [gen_sample(self.h, self.w, self.nt, self.nc) for _ in range(10)]
#         orig_op = RandomShape(sizes=[320, 352, 384, 416, 448, 480, 512, 544, 576, 608], random_inter=True, resize_box=True)
#         curr_op = BatchRandomResizeOp(target_size=[320, 352, 384, 416, 448, 480, 512, 544, 576, 608], random_size=True, random_interp=True, keep_ratio=False)
#         orig_ress = orig_op(copy.deepcopy(sample))
#         curr_ress = curr_op(copy.deepcopy(sample))
#         fields = ['image', 'gt_bbox', 'gt_class', 'gt_score']
#         for orig_res, curr_res in zip(orig_ress, curr_ress):
#             for k in fields:
#                 self.assertAllClose(orig_res[k], curr_res[k], msg=k)


class TestNormalizeBoxOp(TestTransformOp):
    def test_normalize_box(self):
        sample = gen_sample(self.h, self.w, self.nt, self.nc)
        orig_op = NormalizeBox()
        curr_op = NormalizeBoxOp()
        orig_res = orig_op(copy.deepcopy(sample))
        curr_res = curr_op(copy.deepcopy(sample))
        fields = ['image', 'gt_bbox', 'gt_class', 'gt_score']
        for k in fields:
            self.assertAllClose(orig_res[k], curr_res[k], msg=k)


class TestPadBoxOp(TestTransformOp):
    def test_pad_box(self):
        sample = gen_sample(self.h, self.w, self.nt, self.nc)
        orig_op = PadBox(num_max_boxes=50)
        curr_op = PadBoxOp(num_max_boxes=50)
        orig_res = orig_op(copy.deepcopy(sample))
        curr_res = curr_op(copy.deepcopy(sample))
        fields = ['image', 'gt_bbox', 'gt_class', 'gt_score']
        for k in fields:
            self.assertAllClose(orig_res[k], curr_res[k], msg=k)


class TestBboxXYXY2XYWHOp(TestTransformOp):
    def test_bbox_xyxy2xywh(self):
        sample = gen_sample(self.h, self.w, self.nt, self.nc)
        orig_op = BboxXYXY2XYWH()
        curr_op = BboxXYXY2XYWHOp()
        orig_res = orig_op(copy.deepcopy(sample))
        curr_res = curr_op(copy.deepcopy(sample))
        fields = ['image', 'gt_bbox', 'gt_class', 'gt_score']
        for k in fields:
            self.assertAllClose(orig_res[k], curr_res[k], msg=k)


class TestNormalizeImageOp(TestTransformOp):
    def test_normalize_image(self):
        sample = gen_sample(self.h, self.w, self.nt, self.nc)
        orig_op = NormalizeImage(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            is_scale=True,
            is_channel_first=False)
        curr_op = NormalizeImageOp(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            is_scale=True)
        orig_res = orig_op(copy.deepcopy(sample))
        curr_res = curr_op(copy.deepcopy(sample))
        fields = ['image', 'gt_bbox', 'gt_class', 'gt_score']
        for k in fields:
            self.assertAllClose(orig_res[k], curr_res[k], msg=k)


class TestPermuteOp(TestTransformOp):
    def test_permute(self):
        sample = gen_sample(self.h, self.w, self.nt, self.nc)
        orig_op = Permute(to_bgr=False, channel_first=True)
        curr_op = PermuteOp()
        orig_res = orig_op(copy.deepcopy(sample))
        curr_res = curr_op(copy.deepcopy(sample))
        fields = ['image', 'gt_bbox', 'gt_class', 'gt_score']
        for k in fields:
            self.assertAllClose(orig_res[k], curr_res[k], msg=k)


class TestGt2YoloTargetOp(TestTransformOp):
    def test_gt2yolotarget(self):
        sample = [
            gen_sample(
                self.h, self.w, self.nt, self.nc, channel_first=True)
            for _ in range(10)
        ]
        orig_op = Gt2YoloTarget(
            anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
            anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                     [59, 119], [116, 90], [156, 198], [373, 326]],
            downsample_ratios=[32, 16, 8])
        curr_op = Gt2YoloTargetOp(
            anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
            anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                     [59, 119], [116, 90], [156, 198], [373, 326]],
            downsample_ratios=[32, 16, 8])
        orig_ress = orig_op(copy.deepcopy(sample))
        curr_ress = curr_op(copy.deepcopy(sample))
        fields = ['image', 'gt_bbox', 'gt_class', 'gt_score']
        for orig_res, curr_res in zip(orig_ress, curr_ress):
            for k in fields:
                self.assertAllClose(orig_res[k], curr_res[k], msg=k)


if __name__ == "__main__":
    unittest.main()
