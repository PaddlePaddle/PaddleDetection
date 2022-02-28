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
import numpy as np
import collections

__all__ = [
    'MOTTimer', 'Detection', 'write_mot_results', 'load_det_results',
    'preprocess_reid', 'get_crops', 'clip_box', 'scale_coords', 'flow_statistic',
    'plot_tracking'
]


class MOTTimer(object):
    """
    This class used to compute and print the current FPS while evaling.
    """

    def __init__(self, window_size=20):
        self.start_time = 0.
        self.diff = 0.
        self.duration = 0.
        self.deque = collections.deque(maxlen=window_size)

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.deque.append(self.diff)
        if average:
            self.duration = np.mean(self.deque)
        else:
            self.duration = np.sum(self.deque)
        return self.duration

    def clear(self):
        self.start_time = 0.
        self.diff = 0.
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
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, score, track_id in zip(tlwhs, tscores, track_ids):
                if track_id < 0: continue
                if data_type == 'mot':
                    cls_id = -1

                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(
                    frame=frame_id,
                    id=track_id,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    w=w,
                    h=h,
                    score=score,
                    cls_id=cls_id)
                f.write(line)
    print('MOT results save in {}'.format(filename))


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
            results['score'].append(l[5:6])
            results['cls_id'].append(l[6:7])
        results_list.append(results)
    return results_list


def scale_coords(coords, input_shape, im_shape, scale_factor):
    # Note: ratio has only one value, scale_factor[0] == scale_factor[1]
    # 
    # This function only used for JDE YOLOv3 or other detectors with 
    # LetterBoxResize and JDEBBoxPostProcess, coords output from detector had
    # not scaled back to the origin image.

    ratio = scale_factor[0]
    pad_w = (input_shape[1] - int(im_shape[1])) / 2
    pad_h = (input_shape[0] - int(im_shape[0])) / 2
    coords[:, 0::2] -= pad_w
    coords[:, 1::2] -= pad_h
    coords[:, 0:4] /= ratio
    coords[:, :4] = np.clip(coords[:, :4], a_min=0, a_max=coords[:, :4].max())
    return coords.round()


def clip_box(xyxy, ori_image_shape):
    H, W = ori_image_shape
    xyxy[:, 0::2] = np.clip(xyxy[:, 0::2], a_min=0, a_max=W)
    xyxy[:, 1::2] = np.clip(xyxy[:, 1::2], a_min=0, a_max=H)
    w = xyxy[:, 2:3] - xyxy[:, 0:1]
    h = xyxy[:, 3:4] - xyxy[:, 1:2]
    mask = np.logical_and(h > 0, w > 0)
    keep_idx = np.nonzero(mask)
    return xyxy[keep_idx[0]], keep_idx


def get_crops(xyxy, ori_img, w, h):
    crops = []
    xyxy = xyxy.astype(np.int64)
    ori_img = ori_img.numpy()
    ori_img = np.squeeze(ori_img, axis=0).transpose(1, 0, 2)  # [h,w,3]->[w,h,3]
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


def flow_statistic(result,
                   secs_interval,
                   do_entrance_counting,
                   video_fps,
                   entrance,
                   id_set,
                   interval_id_set,
                   in_id_list,
                   out_id_list,
                   prev_center,
                   records,
                   data_type='mot',
                   num_classes=1):
    # Count in and out number: 
    # Use horizontal center line as the entrance just for simplification.
    # If a person located in the above the horizontal center line 
    # at the previous frame and is in the below the line at the current frame,
    # the in number is increased by one.
    # If a person was in the below the horizontal center line 
    # at the previous frame and locates in the below the line at the current frame,
    # the out number is increased by one.
    # TODO: if the entrance is not the horizontal center line,
    # the counting method should be optimized.
    if do_entrance_counting:
        entrance_y = entrance[1]  # xmin, ymin, xmax, ymax
        frame_id, tlwhs, tscores, track_ids = result
        for tlwh, score, track_id in zip(tlwhs, tscores, track_ids):
            if track_id < 0: continue
            if data_type == 'kitti':
                frame_id -= 1

            x1, y1, w, h = tlwh
            center_x = x1 + w / 2.
            center_y = y1 + h / 2.
            if track_id in prev_center:
                if prev_center[track_id][1] <= entrance_y and \
                   center_y > entrance_y:
                    in_id_list.append(track_id)
                if prev_center[track_id][1] >= entrance_y and \
                   center_y < entrance_y:
                    out_id_list.append(track_id)
                prev_center[track_id][0] = center_x
                prev_center[track_id][1] = center_y
            else:
                prev_center[track_id] = [center_x, center_y]
    # Count totol number, number at a manual-setting interval
    frame_id, tlwhs, tscores, track_ids = result
    for tlwh, score, track_id in zip(tlwhs, tscores, track_ids):
        if track_id < 0: continue
        id_set.add(track_id)
        interval_id_set.add(track_id)

    # Reset counting at the interval beginning
    if frame_id % video_fps == 0 and frame_id / video_fps % secs_interval == 0:
        curr_interval_count = len(interval_id_set)
        interval_id_set.clear()
    info = "Frame id: {}, Total count: {}".format(frame_id, len(id_set))
    if do_entrance_counting:
        info += ", In count: {}, Out count: {}".format(
            len(in_id_list), len(out_id_list))
    if frame_id % video_fps == 0 and frame_id / video_fps % secs_interval == 0:
        info += ", Count during {} secs: {}".format(secs_interval,
                                                    curr_interval_count)
        interval_id_set.clear()
    print(info)
    info += "\n"
    records.append(info)

    return {
        "id_set": id_set,
        "interval_id_set": interval_id_set,
        "in_id_list": in_id_list,
        "out_id_list": out_id_list,
        "prev_center": prev_center,
        "records": records
    }


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


def plot_tracking(image,
                  tlwhs,
                  obj_ids,
                  scores=None,
                  frame_id=0,
                  fps=0.,
                  ids2names=[],
                  do_entrance_counting=False,
                  entrance=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    text_scale = max(1, image.shape[1] / 1600.)
    text_thickness = 2
    line_thickness = max(1, int(image.shape[1] / 500.))

    if fps > 0:
        _line = 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs))
    else:
        _line = 'frame: %d num: %d' % (frame_id, len(tlwhs))
    cv2.putText(
        im,
        _line,
        (0, int(15 * text_scale)),
        cv2.FONT_HERSHEY_PLAIN,
        text_scale, (0, 0, 255),
        thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2names != []:
            assert len(
                ids2names) == 1, "plot_tracking only supports single classes."
            id_text = '{}_'.format(ids2names[0]) + id_text
        _line_thickness = 1 if obj_id <= 0 else line_thickness
        color = get_color(abs(obj_id))
        cv2.rectangle(
            im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(
            im,
            id_text, (intbox[0], intbox[1] - 10),
            cv2.FONT_HERSHEY_PLAIN,
            text_scale, (0, 0, 255),
            thickness=text_thickness)

        if scores is not None:
            text = '{:.2f}'.format(float(scores[i]))
            cv2.putText(
                im,
                text, (intbox[0], intbox[1] + 10),
                cv2.FONT_HERSHEY_PLAIN,
                text_scale, (0, 255, 255),
                thickness=text_thickness)

    if do_entrance_counting:
        entrance_line = tuple(map(int, entrance))
        cv2.rectangle(
            im,
            entrance_line[0:2],
            entrance_line[2:4],
            color=(0, 255, 255),
            thickness=line_thickness)
    return im
