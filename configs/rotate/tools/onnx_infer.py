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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import six
import glob
import copy
import yaml
import argparse
import cv2
import numpy as np
from shapely.geometry import Polygon
from onnxruntime import InferenceSession


# preprocess ops
def decode_image(img_path):
    with open(img_path, 'rb') as f:
        im_read = f.read()
    data = np.frombuffer(im_read, dtype='uint8')
    im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    img_info = {
        "im_shape": np.array(
            im.shape[:2], dtype=np.float32),
        "scale_factor": np.array(
            [1., 1.], dtype=np.float32)
    }
    return im, img_info


class Resize(object):
    def __init__(self, target_size, keep_ratio=True, interp=cv2.INTER_LINEAR):
        if isinstance(target_size, int):
            target_size = [target_size, target_size]
        self.target_size = target_size
        self.keep_ratio = keep_ratio
        self.interp = interp

    def __call__(self, im, im_info):
        assert len(self.target_size) == 2
        assert self.target_size[0] > 0 and self.target_size[1] > 0
        im_channel = im.shape[2]
        im_scale_y, im_scale_x = self.generate_scale(im)
        im = cv2.resize(
            im,
            None,
            None,
            fx=im_scale_x,
            fy=im_scale_y,
            interpolation=self.interp)
        im_info['im_shape'] = np.array(im.shape[:2]).astype('float32')
        im_info['scale_factor'] = np.array(
            [im_scale_y, im_scale_x]).astype('float32')
        return im, im_info

    def generate_scale(self, im):
        origin_shape = im.shape[:2]
        im_c = im.shape[2]
        if self.keep_ratio:
            im_size_min = np.min(origin_shape)
            im_size_max = np.max(origin_shape)
            target_size_min = np.min(self.target_size)
            target_size_max = np.max(self.target_size)
            im_scale = float(target_size_min) / float(im_size_min)
            if np.round(im_scale * im_size_max) > target_size_max:
                im_scale = float(target_size_max) / float(im_size_max)
            im_scale_x = im_scale
            im_scale_y = im_scale
        else:
            resize_h, resize_w = self.target_size
            im_scale_y = resize_h / float(origin_shape[0])
            im_scale_x = resize_w / float(origin_shape[1])
        return im_scale_y, im_scale_x


class Permute(object):
    def __init__(self, ):
        super(Permute, self).__init__()

    def __call__(self, im, im_info):
        im = im.transpose((2, 0, 1))
        return im, im_info


class NormalizeImage(object):
    def __init__(self, mean, std, is_scale=True, norm_type='mean_std'):
        self.mean = mean
        self.std = std
        self.is_scale = is_scale
        self.norm_type = norm_type

    def __call__(self, im, im_info):
        im = im.astype(np.float32, copy=False)
        if self.is_scale:
            scale = 1.0 / 255.0
            im *= scale

        if self.norm_type == 'mean_std':
            mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
            std = np.array(self.std)[np.newaxis, np.newaxis, :]
            im -= mean
            im /= std
        return im, im_info


class PadStride(object):
    def __init__(self, stride=0):
        self.coarsest_stride = stride

    def __call__(self, im, im_info):
        coarsest_stride = self.coarsest_stride
        if coarsest_stride <= 0:
            return im, im_info
        im_c, im_h, im_w = im.shape
        pad_h = int(np.ceil(float(im_h) / coarsest_stride) * coarsest_stride)
        pad_w = int(np.ceil(float(im_w) / coarsest_stride) * coarsest_stride)
        padding_im = np.zeros((im_c, pad_h, pad_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = im
        return padding_im, im_info


class Compose:
    def __init__(self, transforms):
        self.transforms = []
        for op_info in transforms:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            self.transforms.append(eval(op_type)(**new_op_info))

    def __call__(self, img_path):
        img, im_info = decode_image(img_path)
        for t in self.transforms:
            img, im_info = t(img, im_info)
        inputs = copy.deepcopy(im_info)
        inputs['image'] = img
        return inputs


# postprocess
def rbox_iou(g, p):
    g = np.array(g)
    p = np.array(p)
    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    g = g.buffer(0)
    p = p.buffer(0)
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter / union


def multiclass_nms_rotated(pred_bboxes,
                           pred_scores,
                           iou_threshlod=0.1,
                           score_threshold=0.1):
    """
    Args:
        pred_bboxes (numpy.ndarray): [B, N, 8]
        pred_scores (numpy.ndarray): [B, C, N]
    
    Return:
        bboxes (numpy.ndarray): [N, 10]
        bbox_num (numpy.ndarray): [B]
    """
    bbox_num = []
    bboxes = []
    for bbox_per_img, score_per_img in zip(pred_bboxes, pred_scores):
        num_per_img = 0
        for cls_id, score_per_cls in enumerate(score_per_img):
            keep_mask = score_per_cls > score_threshold
            bbox = bbox_per_img[keep_mask]
            score = score_per_cls[keep_mask]

            idx = score.argsort()[::-1]
            bbox = bbox[idx]
            score = score[idx]
            keep_idx = []
            for i, b in enumerate(bbox):
                supressed = False
                for gi in keep_idx:
                    g = bbox[gi]
                    if rbox_iou(b, g) > iou_threshlod:
                        supressed = True
                        break

                if supressed:
                    continue

                keep_idx.append(i)

            keep_box = bbox[keep_idx]
            keep_score = score[keep_idx]
            keep_cls_ids = np.ones(len(keep_idx)) * cls_id
            bboxes.append(
                np.concatenate(
                    [keep_cls_ids[:, None], keep_score[:, None], keep_box],
                    axis=-1))
            num_per_img += len(keep_idx)

        bbox_num.append(num_per_img)

    return np.concatenate(bboxes, axis=0), np.array(bbox_num)


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


def predict_image(infer_config, predictor, img_list):
    # load preprocess transforms
    transforms = Compose(infer_config['Preprocess'])
    # predict image
    for img_path in img_list:
        inputs = transforms(img_path)
        inputs_name = [var.name for var in predictor.get_inputs()]
        inputs = {k: inputs[k][None, ] for k in inputs_name}

        outputs = predictor.run(output_names=None, input_feed=inputs)

        bboxes, bbox_num = multiclass_nms_rotated(
            np.array(outputs[0]), np.array(outputs[1]))
        print("ONNXRuntime predict: ")
        for bbox in bboxes:
            if bbox[0] > -1 and bbox[1] > infer_config['draw_threshold']:
                print(f"{int(bbox[0])} {bbox[1]} "
                      f"{bbox[2]} {bbox[3]} {bbox[4]} {bbox[5]}"
                      f"{bbox[6]} {bbox[7]} {bbox[8]} {bbox[9]}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infer_cfg", type=str, help="infer_cfg.yml")
    parser.add_argument(
        '--onnx_file',
        type=str,
        default="model.onnx",
        help="onnx model file path")
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--image_file", type=str)
    return parser.parse_args()


if __name__ == '__main__':
    FLAGS = parse_args()
    # load image list
    img_list = get_test_images(FLAGS.image_dir, FLAGS.image_file)
    # load predictor
    predictor = InferenceSession(FLAGS.onnx_file)
    # load infer config
    with open(FLAGS.infer_cfg) as f:
        infer_config = yaml.safe_load(f)

    predict_image(infer_config, predictor, img_list)
