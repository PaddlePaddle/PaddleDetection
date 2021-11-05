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
from .visualization import plot_tracking_dict

__all__ = [
    'MOTTimer',
    'Detection',
    'write_mot_results',
    'save_vis_results',
    'load_det_results',
    'preprocess_reid',
    'get_crops',
    'clip_box',
    'scale_coords',
]


class MOTTimer(object):
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
        tlwh (Tensor): Bounding box in format `(top left x, top left y,
            width, height)`.
        score (Tensor): Bounding box confidence score.
        feature (Tensor): A feature vector that describes the object 
            contained in this image.
        cls_id (Tensor): Bounding box category id.
    """

    def __init__(self, tlwh, score, feature, cls_id):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.score = float(score)
        self.feature = np.asarray(feature, dtype=np.float32)
        self.cls_id = int(cls_id)

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


def write_mot_results(filename, results, data_type='mot', num_classes=1):
    # support single and multi classes
    if data_type in ['mot', 'mcmot']:
        save_format = '{frame},{id},{x1},{y1},{w},{h},{score},{cls_id},-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} car 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    f = open(filename, 'w')
    for cls_id in range(num_classes):
        for frame_id, tlwhs, tscores, track_ids in results[cls_id]:
            for tlwh, score, track_id in zip(tlwhs, tscores, track_ids):
                if track_id < 0: continue
                if data_type == 'kitti':
                    frame_id -= 1
                elif data_type == 'mot':
                    cls_id = -1
                elif data_type == 'mcmot':
                    cls_id = cls_id

                x1, y1, w, h = tlwh
                line = save_format.format(
                    frame=frame_id,
                    id=track_id,
                    x1=x1,
                    y1=y1,
                    w=w,
                    h=h,
                    score=score,
                    cls_id=cls_id)
                f.write(line)
    print('MOT results save in {}'.format(filename))


def save_vis_results(data,
                     frame_id,
                     online_ids,
                     online_tlwhs,
                     online_scores,
                     average_time,
                     show_image,
                     save_dir,
                     num_classes=1):
    if show_image or save_dir is not None:
        assert 'ori_image' in data
        img0 = data['ori_image'].numpy()[0]
        online_im = plot_tracking_dict(
            img0,
            num_classes,
            online_tlwhs,
            online_ids,
            online_scores,
            frame_id=frame_id,
            fps=1. / average_time)
    if show_image:
        cv2.imshow('online_im', online_im)
    if save_dir is not None:
        cv2.imwrite(
            os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)


def load_det_results(det_file, num_frames):
    assert os.path.exists(det_file) and os.path.isfile(det_file), \
        '{} is not exist or not a file.'.format(det_file)
    labels = np.loadtxt(det_file, dtype='float32', delimiter=',')
    assert labels.shape[1] == 7, \
        "Each line of {} should have 7 items: '[frame_id],[x0],[y0],[w],[h],[score],[class_id]'.".format(det_file)
    results_list = []
    for frame_i in range(num_frames):
        results = {'bbox': [], 'score': [], 'cls_id': []}
        lables_with_frame = labels[labels[:, 0] == frame_i + 1]
        # each line of lables_with_frame:
        # [frame_id],[x0],[y0],[w],[h],[score],[class_id]
        for l in lables_with_frame:
            results['bbox'].append(l[1:5])
            results['score'].append(l[5])
            results['cls_id'].append(l[6])
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
    w = xyxy[:, 2:3] - xyxy[:, 0:1]
    h = xyxy[:, 3:4] - xyxy[:, 1:2]
    mask = paddle.logical_and(h > 0, w > 0)
    keep_idx = paddle.nonzero(mask)
    xyxy = paddle.gather_nd(xyxy, keep_idx[:, :1])
    return xyxy, keep_idx


def get_crops(xyxy, ori_img, w, h):
    crops = []
    xyxy = xyxy.numpy().astype(np.int64)
    ori_img = ori_img.numpy()
    ori_img = np.squeeze(ori_img, axis=0).transpose(1, 0, 2)
    for i, bbox in enumerate(xyxy):
        crop = ori_img[bbox[0]:bbox[2], bbox[1]:bbox[3], :]
        crops.append(crop)
    crops = preprocess_reid(crops, w, h)
    return crops


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
