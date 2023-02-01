#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
import os, sys
# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 4)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

import unittest
import numpy as np

import paddle

import ppdet.modeling.ops as ops
from ppdet.modeling.tests.test_base import LayerTest


def make_rois(h, w, rois_num, output_size):
    rois = np.zeros((0, 4)).astype('float32')
    for roi_num in rois_num:
        roi = np.zeros((roi_num, 4)).astype('float32')
        roi[:, 0] = np.random.randint(0, h - output_size[0], size=roi_num)
        roi[:, 1] = np.random.randint(0, w - output_size[1], size=roi_num)
        roi[:, 2] = np.random.randint(roi[:, 0] + output_size[0], h)
        roi[:, 3] = np.random.randint(roi[:, 1] + output_size[1], w)
        rois = np.vstack((rois, roi))
    return rois


def softmax(x):
    # clip to shiftx, otherwise, when calc loss with
    # log(exp(shiftx)), may get log(0)=INF
    shiftx = (x - np.max(x)).clip(-64.)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


class TestROIAlign(LayerTest):
    def test_roi_align(self):
        b, c, h, w = 2, 12, 20, 20
        inputs_np = np.random.rand(b, c, h, w).astype('float32')
        rois_num = [4, 6]
        output_size = (7, 7)
        rois_np = make_rois(h, w, rois_num, output_size)
        rois_num_np = np.array(rois_num).astype('int32')
        with self.static_graph():
            inputs = paddle.static.data(
                name='inputs', shape=[b, c, h, w], dtype='float32')
            rois = paddle.static.data(
                name='rois', shape=[10, 4], dtype='float32')
            rois_num = paddle.static.data(
                name='rois_num', shape=[None], dtype='int32')

            output = paddle.vision.ops.roi_align(
                x=inputs,
                boxes=rois,
                boxes_num=rois_num,
                output_size=output_size)
            output_np, = self.get_static_graph_result(
                feed={
                    'inputs': inputs_np,
                    'rois': rois_np,
                    'rois_num': rois_num_np
                },
                fetch_list=output,
                with_lod=False)

        with self.dynamic_graph():
            inputs_dy = paddle.to_tensor(inputs_np)
            rois_dy = paddle.to_tensor(rois_np)
            rois_num_dy = paddle.to_tensor(rois_num_np)

            output_dy = paddle.vision.ops.roi_align(
                x=inputs_dy,
                boxes=rois_dy,
                boxes_num=rois_num_dy,
                output_size=output_size)
            output_dy_np = output_dy.numpy()

        self.assertTrue(np.array_equal(output_np, output_dy_np))

    def test_roi_align_error(self):
        with self.static_graph():
            inputs = paddle.static.data(
                name='inputs', shape=[2, 12, 20, 20], dtype='float32')
            rois = paddle.static.data(
                name='data_error', shape=[10, 4], dtype='int32', lod_level=1)
            self.assertRaises(
                TypeError,
                paddle.vision.ops.roi_align,
                input=inputs,
                rois=rois,
                output_size=(7, 7))

        paddle.disable_static()


class TestROIPool(LayerTest):
    def test_roi_pool(self):
        b, c, h, w = 2, 12, 20, 20
        inputs_np = np.random.rand(b, c, h, w).astype('float32')
        rois_num = [4, 6]
        output_size = (7, 7)
        rois_np = make_rois(h, w, rois_num, output_size)
        rois_num_np = np.array(rois_num).astype('int32')
        with self.static_graph():
            inputs = paddle.static.data(
                name='inputs', shape=[b, c, h, w], dtype='float32')
            rois = paddle.static.data(
                name='rois', shape=[10, 4], dtype='float32')
            rois_num = paddle.static.data(
                name='rois_num', shape=[None], dtype='int32')

            output = paddle.vision.ops.roi_pool(
                x=inputs,
                boxes=rois,
                boxes_num=rois_num,
                output_size=output_size)
            output_np, = self.get_static_graph_result(
                feed={
                    'inputs': inputs_np,
                    'rois': rois_np,
                    'rois_num': rois_num_np
                },
                fetch_list=[output],
                with_lod=False)

        with self.dynamic_graph():
            inputs_dy = paddle.to_tensor(inputs_np)
            rois_dy = paddle.to_tensor(rois_np)
            rois_num_dy = paddle.to_tensor(rois_num_np)

            output_dy = paddle.vision.ops.roi_pool(
                x=inputs_dy,
                boxes=rois_dy,
                boxes_num=rois_num_dy,
                output_size=output_size)
            output_dy_np = output_dy.numpy()

        self.assertTrue(np.array_equal(output_np, output_dy_np))

    def test_roi_pool_error(self):
        with self.static_graph():
            inputs = paddle.static.data(
                name='inputs', shape=[2, 12, 20, 20], dtype='float32')
            rois = paddle.static.data(
                name='data_error', shape=[10, 4], dtype='int32', lod_level=1)
            self.assertRaises(
                TypeError,
                paddle.vision.ops.roi_pool,
                input=inputs,
                rois=rois,
                output_size=(7, 7))

        paddle.disable_static()


class TestPriorBox(LayerTest):
    def test_prior_box(self):
        input_np = np.random.rand(2, 10, 32, 32).astype('float32')
        image_np = np.random.rand(2, 10, 40, 40).astype('float32')
        min_sizes = [2, 4]
        with self.static_graph():
            input = paddle.static.data(
                name='input', shape=[2, 10, 32, 32], dtype='float32')
            image = paddle.static.data(
                name='image', shape=[2, 10, 40, 40], dtype='float32')

            box, var = ops.prior_box(
                input=input,
                image=image,
                min_sizes=min_sizes,
                clip=True,
                flip=True)
            box_np, var_np = self.get_static_graph_result(
                feed={
                    'input': input_np,
                    'image': image_np,
                },
                fetch_list=[box, var],
                with_lod=False)

        with self.dynamic_graph():
            inputs_dy = paddle.to_tensor(input_np)
            image_dy = paddle.to_tensor(image_np)

            box_dy, var_dy = ops.prior_box(
                input=inputs_dy,
                image=image_dy,
                min_sizes=min_sizes,
                clip=True,
                flip=True)
            box_dy_np = box_dy.numpy()
            var_dy_np = var_dy.numpy()

        self.assertTrue(np.array_equal(box_np, box_dy_np))
        self.assertTrue(np.array_equal(var_np, var_dy_np))

    def test_prior_box_error(self):
        with self.static_graph():
            input = paddle.static.data(
                name='input', shape=[2, 10, 32, 32], dtype='int32')
            image = paddle.static.data(
                name='image', shape=[2, 10, 40, 40], dtype='int32')
            self.assertRaises(
                TypeError,
                ops.prior_box,
                input=input,
                image=image,
                min_sizes=[2, 4],
                clip=True,
                flip=True)

        paddle.disable_static()


class TestMulticlassNms(LayerTest):
    def test_multiclass_nms(self):
        boxes_np = np.random.rand(10, 81, 4).astype('float32')
        scores_np = np.random.rand(10, 81).astype('float32')
        rois_num_np = np.array([2, 8]).astype('int32')
        with self.static_graph():
            boxes = paddle.static.data(
                name='bboxes',
                shape=[None, 81, 4],
                dtype='float32',
                lod_level=1)
            scores = paddle.static.data(
                name='scores', shape=[None, 81], dtype='float32', lod_level=1)
            rois_num = paddle.static.data(
                name='rois_num', shape=[None], dtype='int32')

            output = ops.multiclass_nms(
                bboxes=boxes,
                scores=scores,
                background_label=0,
                score_threshold=0.5,
                nms_top_k=400,
                nms_threshold=0.3,
                keep_top_k=200,
                normalized=False,
                return_index=True,
                rois_num=rois_num)
            out_np, index_np, nms_rois_num_np = self.get_static_graph_result(
                feed={
                    'bboxes': boxes_np,
                    'scores': scores_np,
                    'rois_num': rois_num_np
                },
                fetch_list=output,
                with_lod=True)
            out_np = np.array(out_np)
            index_np = np.array(index_np)
            nms_rois_num_np = np.array(nms_rois_num_np)

        with self.dynamic_graph():
            boxes_dy = paddle.to_tensor(boxes_np)
            scores_dy = paddle.to_tensor(scores_np)
            rois_num_dy = paddle.to_tensor(rois_num_np)

            out_dy, index_dy, nms_rois_num_dy = ops.multiclass_nms(
                bboxes=boxes_dy,
                scores=scores_dy,
                background_label=0,
                score_threshold=0.5,
                nms_top_k=400,
                nms_threshold=0.3,
                keep_top_k=200,
                normalized=False,
                return_index=True,
                rois_num=rois_num_dy)
            out_dy_np = out_dy.numpy()
            index_dy_np = index_dy.numpy()
            nms_rois_num_dy_np = nms_rois_num_dy.numpy()

        self.assertTrue(np.array_equal(out_np, out_dy_np))
        self.assertTrue(np.array_equal(index_np, index_dy_np))
        self.assertTrue(np.array_equal(nms_rois_num_np, nms_rois_num_dy_np))

    def test_multiclass_nms_error(self):
        with self.static_graph():
            boxes = paddle.static.data(
                name='bboxes', shape=[81, 4], dtype='float32', lod_level=1)
            scores = paddle.static.data(
                name='scores', shape=[81], dtype='float32', lod_level=1)
            rois_num = paddle.static.data(
                name='rois_num', shape=[40, 41], dtype='int32')
            self.assertRaises(
                TypeError,
                ops.multiclass_nms,
                boxes=boxes,
                scores=scores,
                background_label=0,
                score_threshold=0.5,
                nms_top_k=400,
                nms_threshold=0.3,
                keep_top_k=200,
                normalized=False,
                return_index=True,
                rois_num=rois_num)


class TestMatrixNMS(LayerTest):
    def test_matrix_nms(self):
        N, M, C = 7, 1200, 21
        BOX_SIZE = 4
        nms_top_k = 400
        keep_top_k = 200
        score_threshold = 0.01
        post_threshold = 0.

        scores_np = np.random.random((N * M, C)).astype('float32')
        scores_np = np.apply_along_axis(softmax, 1, scores_np)
        scores_np = np.reshape(scores_np, (N, M, C))
        scores_np = np.transpose(scores_np, (0, 2, 1))

        boxes_np = np.random.random((N, M, BOX_SIZE)).astype('float32')
        boxes_np[:, :, 0:2] = boxes_np[:, :, 0:2] * 0.5
        boxes_np[:, :, 2:4] = boxes_np[:, :, 2:4] * 0.5 + 0.5

        with self.static_graph():
            boxes = paddle.static.data(
                name='boxes', shape=[N, M, BOX_SIZE], dtype='float32')
            scores = paddle.static.data(
                name='scores', shape=[N, C, M], dtype='float32')
            out, index, _ = ops.matrix_nms(
                bboxes=boxes,
                scores=scores,
                score_threshold=score_threshold,
                post_threshold=post_threshold,
                nms_top_k=nms_top_k,
                keep_top_k=keep_top_k,
                return_index=True)
            out_np, index_np = self.get_static_graph_result(
                feed={'boxes': boxes_np,
                      'scores': scores_np},
                fetch_list=[out, index],
                with_lod=True)

        with self.dynamic_graph():
            boxes_dy = paddle.to_tensor(boxes_np)
            scores_dy = paddle.to_tensor(scores_np)

            out_dy, index_dy, _ = ops.matrix_nms(
                bboxes=boxes_dy,
                scores=scores_dy,
                score_threshold=score_threshold,
                post_threshold=post_threshold,
                nms_top_k=nms_top_k,
                keep_top_k=keep_top_k,
                return_index=True)
            out_dy_np = out_dy.numpy()
            index_dy_np = index_dy.numpy()

        self.assertTrue(np.array_equal(out_np, out_dy_np))
        self.assertTrue(np.array_equal(index_np, index_dy_np))

    def test_matrix_nms_error(self):
        with self.static_graph():
            bboxes = paddle.static.data(
                name='bboxes', shape=[7, 1200, 4], dtype='float32')
            scores = paddle.static.data(
                name='data_error', shape=[7, 21, 1200], dtype='int32')
            self.assertRaises(
                TypeError,
                ops.matrix_nms,
                bboxes=bboxes,
                scores=scores,
                score_threshold=0.01,
                post_threshold=0.,
                nms_top_k=400,
                keep_top_k=200,
                return_index=True)

        paddle.disable_static()


class TestBoxCoder(LayerTest):
    def test_box_coder(self):

        prior_box_np = np.random.random((81, 4)).astype('float32')
        prior_box_var_np = np.random.random((81, 4)).astype('float32')
        target_box_np = np.random.random((20, 81, 4)).astype('float32')

        # static
        with self.static_graph():
            prior_box = paddle.static.data(
                name='prior_box', shape=[81, 4], dtype='float32')
            prior_box_var = paddle.static.data(
                name='prior_box_var', shape=[81, 4], dtype='float32')
            target_box = paddle.static.data(
                name='target_box', shape=[20, 81, 4], dtype='float32')

            boxes = ops.box_coder(
                prior_box=prior_box,
                prior_box_var=prior_box_var,
                target_box=target_box,
                code_type="decode_center_size",
                box_normalized=False)

            boxes_np, = self.get_static_graph_result(
                feed={
                    'prior_box': prior_box_np,
                    'prior_box_var': prior_box_var_np,
                    'target_box': target_box_np,
                },
                fetch_list=[boxes],
                with_lod=False)

        # dygraph
        with self.dynamic_graph():
            prior_box_dy = paddle.to_tensor(prior_box_np)
            prior_box_var_dy = paddle.to_tensor(prior_box_var_np)
            target_box_dy = paddle.to_tensor(target_box_np)

            boxes_dy = ops.box_coder(
                prior_box=prior_box_dy,
                prior_box_var=prior_box_var_dy,
                target_box=target_box_dy,
                code_type="decode_center_size",
                box_normalized=False)

            boxes_dy_np = boxes_dy.numpy()

            self.assertTrue(np.array_equal(boxes_np, boxes_dy_np))

    def test_box_coder_error(self):
        with self.static_graph():
            prior_box = paddle.static.data(
                name='prior_box', shape=[81, 4], dtype='int32')
            prior_box_var = paddle.static.data(
                name='prior_box_var', shape=[81, 4], dtype='float32')
            target_box = paddle.static.data(
                name='target_box', shape=[20, 81, 4], dtype='float32')

            self.assertRaises(TypeError, ops.box_coder, prior_box,
                              prior_box_var, target_box)

        paddle.disable_static()


if __name__ == '__main__':
    unittest.main()
