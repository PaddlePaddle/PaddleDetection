# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os
import onnx
import onnx_graphsurgeon
import numpy as np
from collections import OrderedDict
from paddle2onnx.command import program2onnx

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '--onnx_file', required=True, type=str, help='onnx model path')
parser.add_argument(
    '--model_dir',
    type=str,
    default=None,
    help=("Directory include:'model.pdiparams', 'model.pdmodel', "
          "'infer_cfg.yml', created by tools/export_model.py."))
parser.add_argument(
    "--opset_version",
    type=int,
    default=11,
    help="set onnx opset version to export")
parser.add_argument(
    '--topk_all', type=int, default=300, help='topk objects for every images')
parser.add_argument(
    '--iou_thres', type=float, default=0.7, help='iou threshold for NMS')
parser.add_argument(
    '--conf_thres', type=float, default=0.01, help='conf threshold for NMS')


def main(FLAGS):
    assert os.path.exists(FLAGS.onnx_file)
    onnx_model = onnx.load(FLAGS.onnx_file)
    graph = onnx_graphsurgeon.import_onnx(onnx_model)
    graph.toposort()
    graph.fold_constants()
    graph.cleanup()

    num_anchors = graph.outputs[1].shape[2]
    num_classes = graph.outputs[1].shape[1]
    scores = onnx_graphsurgeon.Variable(
        name='scores', shape=[-1, num_anchors, num_classes], dtype=np.float32)
    graph.layer(
        op='Transpose',
        name='lastTranspose',
        inputs=[graph.outputs[1]],
        outputs=[scores],
        attrs=OrderedDict(perm=[0, 2, 1]))

    attrs = OrderedDict(
        plugin_version="1",
        background_class=-1,
        max_output_boxes=FLAGS.topk_all,
        score_threshold=FLAGS.conf_thres,
        iou_threshold=FLAGS.iou_thres,
        score_activation=False,
        box_coding=0, )
    outputs = [
        onnx_graphsurgeon.Variable("num_dets", np.int32, [-1, 1]),
        onnx_graphsurgeon.Variable("det_boxes", np.float32,
                                   [-1, FLAGS.topk_all, 4]),
        onnx_graphsurgeon.Variable("det_scores", np.float32,
                                   [-1, FLAGS.topk_all]),
        onnx_graphsurgeon.Variable("det_classes", np.int32,
                                   [-1, FLAGS.topk_all])
    ]
    graph.layer(
        op='EfficientNMS_TRT',
        name="batched_nms",
        inputs=[graph.outputs[0], scores],
        outputs=outputs,
        attrs=attrs)
    graph.outputs = outputs
    graph.cleanup().toposort()
    onnx.save(onnx_graphsurgeon.export_onnx(graph), FLAGS.onnx_file)
    print(f"The modified onnx model is saved in {FLAGS.onnx_file}")


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    if FLAGS.model_dir is not None:
        assert os.path.exists(FLAGS.model_dir)
        program2onnx(
            model_dir=FLAGS.model_dir,
            save_file=FLAGS.onnx_file,
            model_filename="model.pdmodel",
            params_filename="model.pdiparams",
            opset_version=FLAGS.opset_version,
            enable_onnx_checker=True)
    main(FLAGS)
