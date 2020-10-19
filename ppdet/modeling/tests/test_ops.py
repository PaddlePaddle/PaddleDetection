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

        paddle.enable_static()
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

        paddle.disable_static()
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

        paddle.enable_static()
        program = Program()
        with program_guard(program):
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


if __name__ == '__main__':
    unittest.main()
