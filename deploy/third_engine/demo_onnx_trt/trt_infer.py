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

import time
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda

import tensorrt as trt
from collections import OrderedDict
import os
import yaml
import json
import glob
import argparse

from preprocess import Compose
from preprocess import coco_clsid2catid

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--infer_cfg", type=str, help="infer_cfg.yml")
parser.add_argument(
    "--trt_engine", required=True, type=str, help="trt engine path")
parser.add_argument("--image_dir", type=str)
parser.add_argument("--image_file", type=str)
parser.add_argument(
    "--repeats",
    type=int,
    default=1,
    help="Repeat the running test `repeats` times in benchmark")
parser.add_argument(
    "--save_coco",
    action='store_true',
    default=False,
    help="Whether to save coco results")
parser.add_argument(
    "--coco_file", type=str, default="results.json", help="coco results path")

TRT_LOGGER = trt.Logger()
trt.init_libnvinfer_plugins(TRT_LOGGER, namespace="")
# Global dictionary
SUPPORT_MODELS = {
    'YOLO', 'PPYOLOE', 'RCNN', 'SSD', 'Face', 'FCOS', 'SOLOv2', 'TTFNet',
    'S2ANet', 'JDE', 'FairMOT', 'DeepSORT', 'GFL', 'PicoDet', 'CenterNet',
    'TOOD', 'RetinaNet', 'StrongBaseline', 'STGCN', 'YOLOX', 'HRNet'
}


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


def load_trt_engine(engine_path):
    assert os.path.exists(engine_path)
    print("Reading engine from file {}".format(engine_path))
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def predict_image(infer_config, engine, img_list, save_coco=False, repeats=1):
    # load preprocess transforms
    transforms = Compose(infer_config.preprocess_infos)

    stream = cuda.Stream()
    coco_results = []
    num_data = len(img_list)
    avg_time = []
    with engine.create_execution_context() as context:
        # Allocate host and device buffers
        bindings = create_trt_bindings(engine, context)
        # warmup
        run_trt_context(context, bindings, stream, repeats=10)
        # predict image
        for i, img_path in enumerate(img_list):
            inputs = transforms(img_path)
            inputs_name = [k for k, v in bindings.items() if v['is_input']]
            inputs = {
                k: inputs[k][None, ]
                for k in inputs.keys() if k in inputs_name
            }
            # run infer
            for k, v in inputs.items():
                bindings[k]['cpu_data'][...] = v
            output = run_trt_context(context, bindings, stream, repeats=repeats)
            print(f"{i + 1}/{num_data} infer time: {output['infer_time']} ms.")
            avg_time.append(output['infer_time'])
            # get output
            for k, v in output.items():
                if k in bindings.keys():
                    output[k] = np.reshape(v, bindings[k]['shape'])
            if save_coco:
                coco_results.extend(
                    format_coco_results(os.path.split(img_path)[-1], output))
    avg_time = np.mean(avg_time)
    print(
        f"Run on {num_data} data, repeats {repeats} times, avg time: {avg_time} ms."
    )
    if save_coco:
        with open(FLAGS.coco_file, 'w') as f:
            json.dump(coco_results, f)
        print(f"save coco json to {FLAGS.coco_file}")


def create_trt_bindings(engine, context):
    bindings = OrderedDict()
    for name in engine:
        binding_idx = engine.get_binding_index(name)
        size = trt.volume(context.get_binding_shape(binding_idx))
        dtype = trt.nptype(engine.get_binding_dtype(name))
        shape = list(engine.get_binding_shape(binding_idx))
        if shape[0] == -1:
            shape[0] = 1
        bindings[name] = {
            "idx": binding_idx,
            "size": size,
            "dtype": dtype,
            "shape": shape,
            "cpu_data": None,
            "cuda_ptr": None,
            "is_input": True if engine.binding_is_input(name) else False
        }
        if engine.binding_is_input(name):
            bindings[name]['cpu_data'] = np.random.randn(*shape).astype(
                np.float32)
            bindings[name]['cuda_ptr'] = cuda.mem_alloc(bindings[name][
                'cpu_data'].nbytes)
        else:
            bindings[name]['cpu_data'] = cuda.pagelocked_empty(size, dtype)
            bindings[name]['cuda_ptr'] = cuda.mem_alloc(bindings[name][
                'cpu_data'].nbytes)
    return bindings


def run_trt_context(context, bindings, stream, repeats=1):
    # Transfer input data to the GPU.
    for k, v in bindings.items():
        if v['is_input']:
            cuda.memcpy_htod_async(v['cuda_ptr'], v['cpu_data'], stream)
    in_bindings = [int(v['cuda_ptr']) for k, v in bindings.items()]
    output_data = {}
    avg_time = []
    for _ in range(repeats):
        # Run inference
        t1 = time.time()
        context.execute_async_v2(
            bindings=in_bindings, stream_handle=stream.handle)
        # Transfer prediction output from the GPU.
        for k, v in bindings.items():
            if not v['is_input']:
                cuda.memcpy_dtoh_async(v['cpu_data'], v['cuda_ptr'], stream)
                output_data[k] = v['cpu_data']
        # Synchronize the stream
        stream.synchronize()
        t2 = time.time()
        avg_time.append(t2 - t1)
    output_data['infer_time'] = np.mean(avg_time) * 1000
    return output_data


def format_coco_results(file_name, result):
    try:
        image_id = int(os.path.splitext(file_name)[0])
    except:
        image_id = file_name
    num_dets = result['num_dets'].tolist()
    det_classes = result['det_classes'].tolist()
    det_scores = result['det_scores'].tolist()
    det_boxes = result['det_boxes'].tolist()
    per_result = [
        {
            'image_id': image_id,
            'category_id': coco_clsid2catid[int(det_classes[0][idx])],
            'file_name': file_name,
            'bbox': [
                det_boxes[0][idx][0], det_boxes[0][idx][1],
                det_boxes[0][idx][2] - det_boxes[0][idx][0],
                det_boxes[0][idx][3] - det_boxes[0][idx][1]
            ],  # xyxy -> xywh
            'score': det_scores[0][idx]
        } for idx in range(num_dets[0][0])
    ]

    return per_result


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    # load image list
    img_list = get_test_images(FLAGS.image_dir, FLAGS.image_file)
    # load trt engine
    engine = load_trt_engine(FLAGS.trt_engine)
    # load infer config
    infer_config = PredictConfig(FLAGS.infer_cfg)

    predict_image(infer_config, engine, img_list, FLAGS.save_coco,
                  FLAGS.repeats)
    print('Done!')
