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
import yaml
import argparse
import numpy as np
import glob
from onnxruntime import InferenceSession

from preprocess import Compose

# Global dictionary
SUPPORT_MODELS = {
    'YOLO', 'PPYOLOE', 'RCNN', 'SSD', 'Face', 'FCOS', 'SOLOv2', 'TTFNet',
    'S2ANet', 'JDE', 'FairMOT', 'DeepSORT', 'GFL', 'PicoDet', 'CenterNet',
    'TOOD', 'RetinaNet', 'StrongBaseline', 'STGCN', 'YOLOX', 'HRNet'
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--infer_cfg", type=str, help="infer_cfg.yml")
parser.add_argument(
    '--onnx_file', type=str, default="model.onnx", help="onnx model file path")
parser.add_argument("--image_dir", type=str)
parser.add_argument("--image_file", type=str)


def get_test_images(infer_dir, infer_img):
    """
    Get image path list in TEST mode
    """
    assert infer_img is not None or infer_dir is not None, \
        "--image_file or --image_dir should be set"
    assert infer_img is None or os.path.isfile(infer_img), \
            "{} is not a file".format(infer_img)
    assert infer_dir is None or os.path.isdir(infer_dir), \
            "{} is not a directory".format(infer_dir)

    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        return [infer_img]

    images = set()
    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), \
        "infer_dir {} is not a directory".format(infer_dir)
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
    images = list(images)

    assert len(images) > 0, "no image found in {}".format(infer_dir)
    print("Found {} inference images in total.".format(len(images)))

    return images


class PredictConfig(object):
    """set config of preprocess, postprocess and visualize
    Args:
        infer_config (str): path of infer_cfg.yml
    """

    def __init__(self, infer_config):
        # parsing Yaml config for Preprocess
        with open(infer_config) as f:
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


def predict_image(infer_config, predictor, img_list):
    # load preprocess transforms
    transforms = Compose(infer_config.preprocess_infos)
    # predict image
    for img_path in img_list:
        inputs = transforms(img_path)
        inputs_name = [var.name for var in predictor.get_inputs()]
        inputs = {k: inputs[k][None, ] for k in inputs_name}

        outputs = predictor.run(output_names=None, input_feed=inputs)

        print("ONNXRuntime predict: ")
        if infer_config.arch in ["HRNet"]:
            print(np.array(outputs[0]))
        else:
            bboxes = np.array(outputs[0])
            for bbox in bboxes:
                if bbox[0] > -1 and bbox[1] > infer_config.draw_threshold:
                    print(f"{int(bbox[0])} {bbox[1]} "
                          f"{bbox[2]} {bbox[3]} {bbox[4]} {bbox[5]}")


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    # load image list
    img_list = get_test_images(FLAGS.image_dir, FLAGS.image_file)
    # load predictor
    predictor = InferenceSession(FLAGS.onnx_file)
    # load infer config
    infer_config = PredictConfig(FLAGS.infer_cfg)

    predict_image(infer_config, predictor, img_list)
