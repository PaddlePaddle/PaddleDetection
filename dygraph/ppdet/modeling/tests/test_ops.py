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
# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 4)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

import unittest
import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid.framework import Program, program_guard
from paddle.fluid.dygraph import base

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


class TestCollectFpnProposals(LayerTest):
    def test_collect_fpn_proposals(self):
        multi_bboxes_np = []
        multi_scores_np = []
        rois_num_per_level_np = []
        for i in range(4):
            bboxes_np = np.random.rand(5, 4).astype('float32')
            scores_np = np.random.rand(5, 1).astype('float32')
            rois_num = np.array([2, 3]).astype('int32')
            multi_bboxes_np.append(bboxes_np)
            multi_scores_np.append(scores_np)
            rois_num_per_level_np.append(rois_num)

        with self.static_graph():
            multi_bboxes = []
            multi_scores = []
            rois_num_per_level = []
            for i in range(4):
                bboxes = paddle.static.data(
                    name='rois' + str(i),
                    shape=[5, 4],
                    dtype='float32',
                    lod_level=1)
                scores = paddle.static.data(
                    name='scores' + str(i),
                    shape=[5, 1],
                    dtype='float32',
                    lod_level=1)
                rois_num = paddle.static.data(
                    name='rois_num' + str(i), shape=[None], dtype='int32')

                multi_bboxes.append(bboxes)
                multi_scores.append(scores)
                rois_num_per_level.append(rois_num)

            fpn_rois, rois_num = ops.collect_fpn_proposals(
                multi_bboxes,
                multi_scores,
                2,
                5,
                10,
                rois_num_per_level=rois_num_per_level)
            feed = {}
            for i in range(4):
                feed['rois' + str(i)] = multi_bboxes_np[i]
                feed['scores' + str(i)] = multi_scores_np[i]
                feed['rois_num' + str(i)] = rois_num_per_level_np[i]
            fpn_rois_stat, rois_num_stat = self.get_static_graph_result(
                feed=feed, fetch_list=[fpn_rois, rois_num], with_lod=True)
            fpn_rois_stat = np.array(fpn_rois_stat)
            rois_num_stat = np.array(rois_num_stat)

        with self.dynamic_graph():
            multi_bboxes_dy = []
            multi_scores_dy = []
            rois_num_per_level_dy = []
            for i in range(4):
                bboxes_dy = base.to_variable(multi_bboxes_np[i])
                scores_dy = base.to_variable(multi_scores_np[i])
                rois_num_dy = base.to_variable(rois_num_per_level_np[i])
                multi_bboxes_dy.append(bboxes_dy)
                multi_scores_dy.append(scores_dy)
                rois_num_per_level_dy.append(rois_num_dy)
            fpn_rois_dy, rois_num_dy = ops.collect_fpn_proposals(
                multi_bboxes_dy,
                multi_scores_dy,
                2,
                5,
                10,
                rois_num_per_level=rois_num_per_level_dy)
            fpn_rois_dy = fpn_rois_dy.numpy()
            rois_num_dy = rois_num_dy.numpy()

        self.assertTrue(np.array_equal(fpn_rois_stat, fpn_rois_dy))
        self.assertTrue(np.array_equal(rois_num_stat, rois_num_dy))

    def test_collect_fpn_proposals_error(self):
        def generate_input(bbox_type, score_type, name):
            multi_bboxes = []
            multi_scores = []
            for i in range(4):
                bboxes = paddle.static.data(
                    name='rois' + name + str(i),
                    shape=[10, 4],
                    dtype=bbox_type,
                    lod_level=1)
                scores = paddle.static.data(
                    name='scores' + name + str(i),
                    shape=[10, 1],
                    dtype=score_type,
                    lod_level=1)
                multi_bboxes.append(bboxes)
                multi_scores.append(scores)
            return multi_bboxes, multi_scores

        with self.static_graph():
            bbox1 = paddle.static.data(
                name='rois', shape=[5, 10, 4], dtype='float32', lod_level=1)
            score1 = paddle.static.data(
                name='scores', shape=[5, 10, 1], dtype='float32', lod_level=1)
            bbox2, score2 = generate_input('int32', 'float32', '2')
            self.assertRaises(
                TypeError,
                ops.collect_fpn_proposals,
                multi_rois=bbox1,
                multi_scores=score1,
                min_level=2,
                max_level=5,
                post_nms_top_n=2000)
            self.assertRaises(
                TypeError,
                ops.collect_fpn_proposals,
                multi_rois=bbox2,
                multi_scores=score2,
                min_level=2,
                max_level=5,
                post_nms_top_n=2000)


class TestDistributeFpnProposals(LayerTest):
    def test_distribute_fpn_proposals(self):
        rois_np = np.random.rand(10, 4).astype('float32')
        rois_num_np = np.array([4, 6]).astype('int32')
        with self.static_graph():
            rois = paddle.static.data(
                name='rois', shape=[10, 4], dtype='float32')
            rois_num = paddle.static.data(
                name='rois_num', shape=[None], dtype='int32')
            multi_rois, restore_ind, rois_num_per_level = ops.distribute_fpn_proposals(
                fpn_rois=rois,
                min_level=2,
                max_level=5,
                refer_level=4,
                refer_scale=224,
                rois_num=rois_num)
            fetch_list = multi_rois + [restore_ind] + rois_num_per_level
            output_stat = self.get_static_graph_result(
                feed={'rois': rois_np,
                      'rois_num': rois_num_np},
                fetch_list=fetch_list,
                with_lod=True)
            output_stat_np = []
            for output in output_stat:
                output_np = np.array(output)
                if len(output_np) > 0:
                    output_stat_np.append(output_np)

        with self.dynamic_graph():
            rois_dy = base.to_variable(rois_np)
            rois_num_dy = base.to_variable(rois_num_np)
            multi_rois_dy, restore_ind_dy, rois_num_per_level_dy = ops.distribute_fpn_proposals(
                fpn_rois=rois_dy,
                min_level=2,
                max_level=5,
                refer_level=4,
                refer_scale=224,
                rois_num=rois_num_dy)
            output_dy = multi_rois_dy + [restore_ind_dy] + rois_num_per_level_dy
            output_dy_np = []
            for output in output_dy:
                output_np = output.numpy()
                if len(output_np) > 0:
                    output_dy_np.append(output_np)

        for res_stat, res_dy in zip(output_stat_np, output_dy_np):
            self.assertTrue(np.array_equal(res_stat, res_dy))

    def test_distribute_fpn_proposals_error(self):
        with self.static_graph():
            fpn_rois = paddle.static.data(
                name='data_error', shape=[10, 4], dtype='int32', lod_level=1)
            self.assertRaises(
                TypeError,
                ops.distribute_fpn_proposals,
                fpn_rois=fpn_rois,
                min_level=2,
                max_level=5,
                refer_level=4,
                refer_scale=224)


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

            output = ops.roi_align(
                input=inputs,
                rois=rois,
                output_size=output_size,
                rois_num=rois_num)
            output_np, = self.get_static_graph_result(
                feed={
                    'inputs': inputs_np,
                    'rois': rois_np,
                    'rois_num': rois_num_np
                },
                fetch_list=output,
                with_lod=False)

        with self.dynamic_graph():
            inputs_dy = base.to_variable(inputs_np)
            rois_dy = base.to_variable(rois_np)
            rois_num_dy = base.to_variable(rois_num_np)

            output_dy = ops.roi_align(
                input=inputs_dy,
                rois=rois_dy,
                output_size=output_size,
                rois_num=rois_num_dy)
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
                ops.roi_align,
                input=inputs,
                rois=rois,
                output_size=(7, 7))


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

            output, _ = ops.roi_pool(
                input=inputs,
                rois=rois,
                output_size=output_size,
                rois_num=rois_num)
            output_np, = self.get_static_graph_result(
                feed={
                    'inputs': inputs_np,
                    'rois': rois_np,
                    'rois_num': rois_num_np
                },
                fetch_list=[output],
                with_lod=False)

        with self.dynamic_graph():
            inputs_dy = base.to_variable(inputs_np)
            rois_dy = base.to_variable(rois_np)
            rois_num_dy = base.to_variable(rois_num_np)

            output_dy, _ = ops.roi_pool(
                input=inputs_dy,
                rois=rois_dy,
                output_size=output_size,
                rois_num=rois_num_dy)
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
                ops.roi_pool,
                input=inputs,
                rois=rois,
                output_size=(7, 7))


class TestIoUSimilarity(LayerTest):
    def test_iou_similarity(self):
        b, c, h, w = 2, 12, 20, 20
        inputs_np = np.random.rand(b, c, h, w).astype('float32')
        output_size = (7, 7)
        x_np = make_rois(h, w, [20], output_size)
        y_np = make_rois(h, w, [10], output_size)
        with self.static_graph():
            x = paddle.static.data(name='x', shape=[20, 4], dtype='float32')
            y = paddle.static.data(name='y', shape=[10, 4], dtype='float32')

            iou = ops.iou_similarity(x=x, y=y)
            iou_np, = self.get_static_graph_result(
                feed={
                    'x': x_np,
                    'y': y_np,
                }, fetch_list=[iou], with_lod=False)

        with self.dynamic_graph():
            x_dy = base.to_variable(x_np)
            y_dy = base.to_variable(y_np)

            iou_dy = ops.iou_similarity(x=x_dy, y=y_dy)
            iou_dy_np = iou_dy.numpy()

        self.assertTrue(np.array_equal(iou_np, iou_dy_np))


class TestBipartiteMatch(LayerTest):
    def test_bipartite_match(self):
        distance = np.random.random((20, 10)).astype('float32')
        with self.static_graph():
            x = paddle.static.data(name='x', shape=[20, 10], dtype='float32')

            match_indices, match_dist = ops.bipartite_match(
                x, match_type='per_prediction', dist_threshold=0.5)
            match_indices_np, match_dist_np = self.get_static_graph_result(
                feed={'x': distance, },
                fetch_list=[match_indices, match_dist],
                with_lod=False)

        with self.dynamic_graph():
            x_dy = base.to_variable(distance)

            match_indices_dy, match_dist_dy = ops.bipartite_match(
                x_dy, match_type='per_prediction', dist_threshold=0.5)
            match_indices_dy_np = match_indices_dy.numpy()
            match_dist_dy_np = match_dist_dy.numpy()

        self.assertTrue(np.array_equal(match_indices_np, match_indices_dy_np))
        self.assertTrue(np.array_equal(match_dist_np, match_dist_dy_np))


class TestYoloBox(LayerTest):
    def test_yolo_box(self):

        # x shape [N C H W], C=K * (5 + class_num), class_num=10, K=2
        np_x = np.random.random([1, 30, 7, 7]).astype('float32')
        np_origin_shape = np.array([[608, 608]], dtype='int32')
        class_num = 10
        conf_thresh = 0.01
        downsample_ratio = 32
        scale_x_y = 1.2

        # static
        with self.static_graph():
            # x shape [N C H W], C=K * (5 + class_num), class_num=10, K=2
            x = paddle.static.data(
                name='x', shape=[1, 30, 7, 7], dtype='float32')
            origin_shape = paddle.static.data(
                name='origin_shape', shape=[1, 2], dtype='int32')

            boxes, scores = ops.yolo_box(
                x,
                origin_shape, [10, 13, 30, 13],
                class_num,
                conf_thresh,
                downsample_ratio,
                scale_x_y=scale_x_y)

            boxes_np, scores_np = self.get_static_graph_result(
                feed={
                    'x': np_x,
                    'origin_shape': np_origin_shape,
                },
                fetch_list=[boxes, scores],
                with_lod=False)

        # dygraph
        with self.dynamic_graph():
            x_dy = fluid.layers.assign(np_x)
            origin_shape_dy = fluid.layers.assign(np_origin_shape)

            boxes_dy, scores_dy = ops.yolo_box(
                x_dy,
                origin_shape_dy, [10, 13, 30, 13],
                10,
                0.01,
                32,
                scale_x_y=scale_x_y)

            boxes_dy_np = boxes_dy.numpy()
            scores_dy_np = scores_dy.numpy()

        self.assertTrue(np.array_equal(boxes_np, boxes_dy_np))
        self.assertTrue(np.array_equal(scores_np, scores_dy_np))

    def test_yolo_box_error(self):
        with self.static_graph():
            # x shape [N C H W], C=K * (5 + class_num), class_num=10, K=2
            x = paddle.static.data(
                name='x', shape=[1, 30, 7, 7], dtype='float32')
            origin_shape = paddle.static.data(
                name='origin_shape', shape=[1, 2], dtype='int32')

            self.assertRaises(
                TypeError,
                ops.yolo_box,
                x,
                origin_shape, [10, 13, 30, 13],
                10.123,
                0.01,
                32,
                scale_x_y=1.2)


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
            inputs_dy = base.to_variable(input_np)
            image_dy = base.to_variable(image_np)

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


class TestAnchorGenerator(LayerTest):
    def test_anchor_generator(self):
        b, c, h, w = 2, 48, 16, 16
        input_np = np.random.rand(2, 48, 16, 16).astype('float32')
        with self.static_graph():
            input = paddle.static.data(
                name='input', shape=[b, c, h, w], dtype='float32')

            anchor, var = ops.anchor_generator(
                input=input,
                anchor_sizes=[64, 128, 256, 512],
                aspect_ratios=[0.5, 1.0, 2.0],
                variance=[0.1, 0.1, 0.2, 0.2],
                stride=[16.0, 16.0],
                offset=0.5)
            anchor_np, var_np = self.get_static_graph_result(
                feed={'input': input_np, },
                fetch_list=[anchor, var],
                with_lod=False)

        with self.dynamic_graph():
            inputs_dy = base.to_variable(input_np)

            anchor_dy, var_dy = ops.anchor_generator(
                input=inputs_dy,
                anchor_sizes=[64, 128, 256, 512],
                aspect_ratios=[0.5, 1.0, 2.0],
                variance=[0.1, 0.1, 0.2, 0.2],
                stride=[16.0, 16.0],
                offset=0.5)
            anchor_dy_np = anchor_dy.numpy()
            var_dy_np = var_dy.numpy()

        self.assertTrue(np.array_equal(anchor_np, anchor_dy_np))
        self.assertTrue(np.array_equal(var_np, var_dy_np))


class TestMulticlassNms(LayerTest):
    def test_multiclass_nms(self):
        boxes_np = np.random.rand(81, 4).astype('float32')
        scores_np = np.random.rand(81).astype('float32')
        rois_num_np = np.array([40, 41]).astype('int32')
        with self.static_graph():
            boxes = paddle.static.data(
                name='bboxes', shape=[81, 4], dtype='float32', lod_level=1)
            scores = paddle.static.data(
                name='scores', shape=[81], dtype='float32', lod_level=1)
            rois_num = paddle.static.data(
                name='rois_num', shape=[40, 41], dtype='int32')

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
                with_lod=False)

        with self.dynamic_graph():
            boxes_dy = base.to_variable(boxes_np)
            scores_dy = base.to_variable(scores_np)
            rois_num_dy = base.to_variable(rois_num_np)

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
            boxes_dy = base.to_variable(boxes_np)
            scores_dy = base.to_variable(scores_np)

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
            prior_box_dy = base.to_variable(prior_box_np)
            prior_box_var_dy = base.to_variable(prior_box_var_np)
            target_box_dy = base.to_variable(target_box_np)

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


class TestGenerateProposals(LayerTest):
    def test_generate_proposals(self):
        scores_np = np.random.rand(2, 3, 4, 4).astype('float32')
        bbox_deltas_np = np.random.rand(2, 12, 4, 4).astype('float32')
        im_shape_np = np.array([[8, 8], [6, 6]]).astype('float32')
        anchors_np = np.reshape(np.arange(4 * 4 * 3 * 4),
                                [4, 4, 3, 4]).astype('float32')
        variances_np = np.ones((4, 4, 3, 4)).astype('float32')

        with self.static_graph():
            scores = paddle.static.data(
                name='scores', shape=[2, 3, 4, 4], dtype='float32')
            bbox_deltas = paddle.static.data(
                name='bbox_deltas', shape=[2, 12, 4, 4], dtype='float32')
            im_shape = paddle.static.data(
                name='im_shape', shape=[2, 2], dtype='float32')
            anchors = paddle.static.data(
                name='anchors', shape=[4, 4, 3, 4], dtype='float32')
            variances = paddle.static.data(
                name='var', shape=[4, 4, 3, 4], dtype='float32')
            rois, roi_probs, rois_num = ops.generate_proposals(
                scores,
                bbox_deltas,
                im_shape,
                anchors,
                variances,
                pre_nms_top_n=10,
                post_nms_top_n=5,
                return_rois_num=True)
            rois_stat, roi_probs_stat, rois_num_stat = self.get_static_graph_result(
                feed={
                    'scores': scores_np,
                    'bbox_deltas': bbox_deltas_np,
                    'im_shape': im_shape_np,
                    'anchors': anchors_np,
                    'var': variances_np
                },
                fetch_list=[rois, roi_probs, rois_num],
                with_lod=True)

        with self.dynamic_graph():
            scores_dy = base.to_variable(scores_np)
            bbox_deltas_dy = base.to_variable(bbox_deltas_np)
            im_shape_dy = base.to_variable(im_shape_np)
            anchors_dy = base.to_variable(anchors_np)
            variances_dy = base.to_variable(variances_np)
            rois, roi_probs, rois_num = ops.generate_proposals(
                scores_dy,
                bbox_deltas_dy,
                im_shape_dy,
                anchors_dy,
                variances_dy,
                pre_nms_top_n=10,
                post_nms_top_n=5,
                return_rois_num=True)
            rois_dy = rois.numpy()
            roi_probs_dy = roi_probs.numpy()
            rois_num_dy = rois_num.numpy()

        self.assertTrue(np.array_equal(np.array(rois_stat), rois_dy))
        self.assertTrue(np.array_equal(np.array(roi_probs_stat), roi_probs_dy))
        self.assertTrue(np.array_equal(np.array(rois_num_stat), rois_num_dy))


if __name__ == '__main__':
    unittest.main()
