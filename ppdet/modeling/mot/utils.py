# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import cv2
import time
import paddle
import numpy as np

__all__ = [
    'Timer',
    'Detection',
    'load_det_results',
    'preprocess_reid',
    'get_crops',
    'clip_box',
    'scale_coords',
]


class Timer(object):
    """
    This class used to compute and print the current FPS while evaling.
    """

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self.duration = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            self.duration = self.average_time
        else:
            self.duration = self.diff
        return self.duration

    def clear(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self.duration = 0.


class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Args:
        tlwh (ndarray): Bounding box in format `(top left x, top left y,
            width, height)`.
        confidence (ndarray): Detector confidence score.
        feature (Tensor): A feature vector that describes the object 
            contained in this image.
    """

    def __init__(self, tlwh, confidence, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.confidence = np.asarray(confidence, dtype=np.float32)
        self.feature = feature

    def to_tlbr(self):
        """
        Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """
        Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret


def load_det_results(det_file, num_frames):
    assert os.path.exists(det_file) and os.path.isfile(det_file), \
        'Error: det_file: {} not exist or not a file.'.format(det_file)
    labels = np.loadtxt(det_file, dtype='float32', delimiter=',')
    results_list = []
    for frame_i in range(0, num_frames):
        results = {'bbox': [], 'score': []}
        lables_with_frame = labels[labels[:, 0] == frame_i + 1]
        for l in lables_with_frame:
            results['bbox'].append(l[1:5])
            results['score'].append(l[5])
        results_list.append(results)
    return results_list


def scale_coords(coords, input_shape, im_shape, scale_factor):
    im_shape = im_shape.numpy()[0]
    ratio = scale_factor[0][0]
    pad_w = (input_shape[1] - int(im_shape[1])) / 2
    pad_h = (input_shape[0] - int(im_shape[0])) / 2
    coords = paddle.cast(coords, 'float32')
    coords[:, 0::2] -= pad_w
    coords[:, 1::2] -= pad_h
    coords[:, 0:4] /= ratio
    coords[:, :4] = paddle.clip(coords[:, :4], min=0, max=coords[:, :4].max())
    return coords.round()


def clip_box(xyxy, input_shape, im_shape, scale_factor):
    im_shape = im_shape.numpy()[0]
    ratio = scale_factor.numpy()[0][0]
    img0_shape = [int(im_shape[0] / ratio), int(im_shape[1] / ratio)]

    xyxy[:, 0::2] = paddle.clip(xyxy[:, 0::2], min=0, max=img0_shape[1])
    xyxy[:, 1::2] = paddle.clip(xyxy[:, 1::2], min=0, max=img0_shape[0])
    return xyxy


def get_crops(xyxy, ori_img, pred_scores, w, h):
    crops = []
    keep_scores = []
    xyxy = xyxy.numpy().astype(np.int64)
    ori_img = ori_img.numpy()
    ori_img = np.squeeze(ori_img, axis=0).transpose(1, 0, 2)
    pred_scores = pred_scores.numpy()
    for i, bbox in enumerate(xyxy):
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        crop = ori_img[bbox[0]:bbox[2], bbox[1]:bbox[3], :]
        crops.append(crop)
        keep_scores.append(pred_scores[i])
    if len(crops) == 0:
        return [], []
    crops = preprocess_reid(crops, w, h)
    return crops, keep_scores


def preprocess_reid(imgs,
                    w=64,
                    h=192,
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]):
    im_batch = []
    for img in imgs:
        img = cv2.resize(img, (w, h))
        img = img[:, :, ::-1].astype('float32').transpose((2, 0, 1)) / 255
        img_mean = np.array(mean).reshape((3, 1, 1))
        img_std = np.array(std).reshape((3, 1, 1))
        img -= img_mean
        img /= img_std
        img = np.expand_dims(img, axis=0)
        im_batch.append(img)
    im_batch = np.concatenate(im_batch, 0)
    return im_batch
