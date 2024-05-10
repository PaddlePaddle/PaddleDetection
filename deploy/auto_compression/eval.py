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

import os
import sys
import numpy as np
import argparse
import paddle
from ppdet.core.workspace import load_config, merge_config
from ppdet.core.workspace import create
from ppdet.metrics import COCOMetric, VOCMetric, KeyPointTopDownCOCOEval
from paddleslim.auto_compression.config_helpers import load_config as load_slim_config
from post_process import PPYOLOEPostProcess


def argsparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--act_config_path',
        type=str,
        default=None,
        help="path of compression strategy config.",
        required=True)
    parser.add_argument(
        '--devices',
        type=str,
        default='gpu',
        help="which device used to compress.")

    return parser


def reader_wrapper(reader, input_list):
    def gen():
        for data in reader:
            in_dict = {}
            if isinstance(input_list, list):
                for input_name in input_list:
                    in_dict[input_name] = data[input_name]
            elif isinstance(input_list, dict):
                for input_name in input_list.keys():
                    in_dict[input_list[input_name]] = data[input_name]
            yield in_dict

    return gen


def convert_numpy_data(data, metric):
    data_all = {}
    data_all = {k: np.array(v) for k, v in data.items()}
    if isinstance(metric, VOCMetric):
        for k, v in data_all.items():
            if not isinstance(v[0], np.ndarray):
                tmp_list = []
                for t in v:
                    tmp_list.append(np.array(t))
                data_all[k] = np.array(tmp_list)
    else:
        data_all = {k: np.array(v) for k, v in data.items()}
    return data_all


def eval():

    place = paddle.CUDAPlace(0) if FLAGS.devices == 'gpu' else paddle.CPUPlace()
    exe = paddle.static.Executor(place)

    val_program, feed_target_names, fetch_targets = paddle.static.load_inference_model(
        global_config["model_dir"].rstrip('/'),
        exe,
        model_filename=global_config["model_filename"],
        params_filename=global_config["params_filename"])
    print('Loaded model from: {}'.format(global_config["model_dir"]))

    metric = global_config['metric']
    for batch_id, data in enumerate(val_loader):
        data_all = convert_numpy_data(data, metric)
        data_input = {}
        for k, v in data.items():
            if isinstance(global_config['input_list'], list):
                if k in global_config['input_list']:
                    data_input[k] = np.array(v)
            elif isinstance(global_config['input_list'], dict):
                if k in global_config['input_list'].keys():
                    data_input[global_config['input_list'][k]] = np.array(v)

        outs = exe.run(val_program,
                       feed=data_input,
                       fetch_list=fetch_targets,
                       return_numpy=False)
        res = {}
        if 'arch' in global_config and global_config['arch'] == 'PPYOLOE':
            postprocess = PPYOLOEPostProcess(
                score_threshold=0.01, nms_threshold=0.6)
            res = postprocess(np.array(outs[0]), data_all['scale_factor'])
        else:
            for out in outs:
                v = np.array(out)
                if len(v.shape) > 1:
                    res['bbox'] = v
                else:
                    res['bbox_num'] = v
        metric.update(data_all, res)
        if batch_id % 100 == 0:
            print('Eval iter:', batch_id)
    metric.accumulate()
    metric.log()
    metric.reset()


def main():
    global global_config
    all_config = load_slim_config(FLAGS.act_config_path)
    assert "Global" in all_config, "Key 'Global' not found in config file."
    global_config = all_config["Global"]
    reader_cfg = load_config(global_config['reader_config'])

    dataset = reader_cfg['EvalDataset']
    global val_loader
    val_loader = create('EvalReader')(reader_cfg['EvalDataset'],
                                      reader_cfg['worker_num'],
                                      return_list=True)
    metric = None
    if reader_cfg['metric'] == 'COCO':
        clsid2catid = {v: k for k, v in dataset.catid2clsid.items()}
        anno_file = dataset.get_anno()
        metric = COCOMetric(
            anno_file=anno_file, clsid2catid=clsid2catid, IouType='bbox')
    elif reader_cfg['metric'] == 'VOC':
        metric = VOCMetric(
            label_list=dataset.get_label_list(),
            class_num=reader_cfg['num_classes'],
            map_type=reader_cfg['map_type'])
    elif reader_cfg['metric'] == 'KeyPointTopDownCOCOEval':
        anno_file = dataset.get_anno()
        metric = KeyPointTopDownCOCOEval(anno_file,
                                         len(dataset), 17, 'output_eval')
    else:
        raise ValueError("metric currently only supports COCO and VOC.")
    global_config['metric'] = metric

    eval()


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()
    assert FLAGS.devices in ['cpu', 'gpu', 'xpu', 'npu']
    paddle.set_device(FLAGS.devices)

    main()
