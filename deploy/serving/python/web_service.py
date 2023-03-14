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
import copy

from paddle_serving_server.web_service import WebService, Op
from paddle_serving_server.proto import general_model_config_pb2 as m_config
import google.protobuf.text_format

import os
import numpy as np
import base64
from PIL import Image
import io
from preprocess_ops import Compose
from postprocess_ops import HRNetPostProcess

from argparse import ArgumentParser, RawDescriptionHelpFormatter
import yaml

# Global dictionary
SUPPORT_MODELS = {
    'YOLO', 'PPYOLOE', 'RCNN', 'SSD', 'Face', 'FCOS', 'SOLOv2', 'TTFNet',
    'S2ANet', 'JDE', 'FairMOT', 'DeepSORT', 'GFL', 'PicoDet', 'CenterNet',
    'TOOD', 'RetinaNet', 'StrongBaseline', 'STGCN', 'YOLOX', 'HRNet'
}

GLOBAL_VAR = {}


class ArgsParser(ArgumentParser):
    def __init__(self):
        super(ArgsParser, self).__init__(
            formatter_class=RawDescriptionHelpFormatter)
        self.add_argument(
            "-c",
            "--config",
            default="deploy/serving/python/config.yml",
            help="configuration file to use")
        self.add_argument(
            "--model_dir",
            type=str,
            default=None,
            help=("Directory include:'model.pdiparams', 'model.pdmodel', "
                  "'infer_cfg.yml', created by tools/export_model.py."),
            required=True)
        self.add_argument(
            "-o", "--opt", nargs='+', help="set configuration options")

    def parse_args(self, argv=None):
        args = super(ArgsParser, self).parse_args(argv)
        assert args.config is not None, \
            "Please specify --config=configure_file_path."
        args.service_config = self._parse_opt(args.opt, args.config)
        args.model_config = PredictConfig(args.model_dir)
        return args

    def _parse_helper(self, v):
        if v.isnumeric():
            if "." in v:
                v = float(v)
            else:
                v = int(v)
        elif v == "True" or v == "False":
            v = (v == "True")
        return v

    def _parse_opt(self, opts, conf_path):
        f = open(conf_path)
        config = yaml.load(f, Loader=yaml.Loader)
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            k, v = s.split('=')
            v = self._parse_helper(v)
            if "devices" in k:
                v = str(v)
            print(k, v, type(v))
            cur = config
            parent = cur
            for kk in k.split("."):
                if kk not in cur:
                    cur[kk] = {}
                    parent = cur
                    cur = cur[kk]
                else:
                    parent = cur
                    cur = cur[kk]
            parent[k.split(".")[-1]] = v
        return config


class PredictConfig(object):
    """set config of preprocess, postprocess and visualize
    Args:
        model_dir (str): root path of infer_cfg.yml
    """

    def __init__(self, model_dir):
        # parsing Yaml config for Preprocess
        deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        self.check_model(yml_conf)
        self.arch = yml_conf['arch']
        self.preprocess_infos = yml_conf['Preprocess']
        self.min_subgraph_size = yml_conf['min_subgraph_size']
        self.label_list = yml_conf['label_list']
        self.use_dynamic_shape = yml_conf['use_dynamic_shape']
        self.draw_threshold = yml_conf.get("draw_threshold", 0.5)
        self.mask = yml_conf.get("mask", False)
        self.tracker = yml_conf.get("tracker", None)
        self.nms = yml_conf.get("NMS", None)
        self.fpn_stride = yml_conf.get("fpn_stride", None)
        if self.arch == 'RCNN' and yml_conf.get('export_onnx', False):
            print(
                'The RCNN export model is used for ONNX and it only supports batch_size = 1'
            )
        self.print_config()

    def check_model(self, yml_conf):
        """
        Raises:
            ValueError: loaded model not in supported model type
        """
        for support_model in SUPPORT_MODELS:
            if support_model in yml_conf['arch']:
                return True
        raise ValueError("Unsupported arch: {}, expect {}".format(yml_conf[
            'arch'], SUPPORT_MODELS))

    def print_config(self):
        print('-----------  Model Configuration -----------')
        print('%s: %s' % ('Model Arch', self.arch))
        print('%s: ' % ('Transform Order'))
        for op_info in self.preprocess_infos:
            print('--%s: %s' % ('transform op', op_info['type']))
        print('--------------------------------------------')


class DetectorOp(Op):
    def init_op(self):
        self.preprocess_pipeline = Compose(GLOBAL_VAR['preprocess_ops'])

    def preprocess(self, input_dicts, data_id, log_id):
        (_, input_dict), = input_dicts.items()
        inputs = []
        for key, data in input_dict.items():
            data = base64.b64decode(data.encode('utf8'))
            byte_stream = io.BytesIO(data)
            img = Image.open(byte_stream).convert("RGB")
            inputs.append(self.preprocess_pipeline(img))
        inputs = self.collate_inputs(inputs)
        return inputs, False, None, ""

    def postprocess(self, input_dicts, fetch_dict, data_id, log_id):
        (_, input_dict), = input_dicts.items()
        if GLOBAL_VAR['model_config'].arch in ["HRNet"]:
            result = self.parse_keypoint_result(input_dict, fetch_dict)
        else:
            result = self.parse_detection_result(input_dict, fetch_dict)
        return result, None, ""

    def collate_inputs(self, inputs):
        collate_inputs = {k: [] for k in inputs[0].keys()}
        for info in inputs:
            for k in collate_inputs.keys():
                collate_inputs[k].append(info[k])
        return {
            k: np.stack(v)
            for k, v in collate_inputs.items() if k in GLOBAL_VAR['feed_vars']
        }

    def parse_detection_result(self, input_dict, fetch_dict):
        bboxes = fetch_dict[GLOBAL_VAR['fetch_vars'][0]]
        bboxes_num = fetch_dict[GLOBAL_VAR['fetch_vars'][1]]
        if GLOBAL_VAR['model_config'].mask:
            masks = fetch_dict[GLOBAL_VAR['fetch_vars'][2]]
        idx = 0
        results = {}
        for img_name, num in zip(input_dict.keys(), bboxes_num):
            if num == 0:
                results[img_name] = 'No object detected!'
            else:
                result = []
                bbox = bboxes[idx:idx + num]
                for line in bbox:
                    if line[0] > -1 and line[1] > GLOBAL_VAR[
                            'model_config'].draw_threshold:
                        result.append(
                            f"{int(line[0])} {line[1]} "
                            f"{line[2]} {line[3]} {line[4]} {line[5]}")
                if len(result) == 0:
                    result = 'No object detected!'
                results[img_name] = result
            idx += num
        return results

    def parse_keypoint_result(self, input_dict, fetch_dict):
        heatmap = fetch_dict["conv2d_441.tmp_1"]
        keypoint_postprocess = HRNetPostProcess()
        im_shape = []
        for key, data in input_dict.items():
            data = base64.b64decode(data.encode('utf8'))
            byte_stream = io.BytesIO(data)
            img = Image.open(byte_stream).convert("RGB")
            im_shape.append([img.width, img.height])
        im_shape = np.array(im_shape)
        center = np.round(im_shape / 2.)
        scale = im_shape / 200.
        kpts, scores = keypoint_postprocess(heatmap, center, scale)
        results = {"keypoint": kpts, "scores": scores}
        return results


class DetectorService(WebService):
    def get_pipeline_response(self, read_op):
        return DetectorOp(name="ppdet", input_ops=[read_op])


def get_model_vars(model_dir, service_config):
    serving_server_dir = os.path.join(model_dir, "serving_server")
    # rewrite model_config
    service_config['op']['ppdet']['local_service_conf'][
        'model_config'] = serving_server_dir
    serving_server_conf = os.path.join(serving_server_dir,
                                       "serving_server_conf.prototxt")
    with open(serving_server_conf, 'r') as f:
        model_var = google.protobuf.text_format.Merge(
            str(f.read()), m_config.GeneralModelConfig())
    feed_vars = [var.name for var in model_var.feed_var]
    fetch_vars = [var.name for var in model_var.fetch_var]
    return feed_vars, fetch_vars


if __name__ == '__main__':
    # load config and prepare the service
    FLAGS = ArgsParser().parse_args()
    feed_vars, fetch_vars = get_model_vars(FLAGS.model_dir,
                                           FLAGS.service_config)
    GLOBAL_VAR['feed_vars'] = feed_vars
    GLOBAL_VAR['fetch_vars'] = fetch_vars
    GLOBAL_VAR['preprocess_ops'] = FLAGS.model_config.preprocess_infos
    GLOBAL_VAR['model_config'] = FLAGS.model_config
    print(FLAGS)
    # define the service
    uci_service = DetectorService(name="ppdet")
    uci_service.prepare_pipeline_config(yml_dict=FLAGS.service_config)
    # start the service
    uci_service.run_service()
