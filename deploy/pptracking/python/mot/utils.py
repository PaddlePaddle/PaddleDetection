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
import math

__all__ = [
    'MOTTimer', 'Detection', 'write_mot_results', 'load_det_results',
    'preprocess_reid', 'get_crops', 'clip_box', 'scale_coords',
    'flow_statistic', 'update_object_info'
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
    ori_img = ori_img.transpose(1, 0, 2)  # [h,w,3]->[w,h,3]
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
                   do_break_in_counting,
                   region_type,
                   video_fps,
                   entrance,
                   id_set,
                   interval_id_set,
                   in_id_list,
                   out_id_list,
                   prev_center,
                   records,
                   data_type='mot',
                   ids2names=['pedestrian']):
    # Count in/out number: 
    # Note that 'region_type' should be one of ['horizontal', 'vertical', 'custom'],
    # 'horizontal' and 'vertical' means entrance is the center line as the entrance when do_entrance_counting, 
    # 'custom' means entrance is a region defined by users when do_break_in_counting.

    if do_entrance_counting:
        assert region_type in [
            'horizontal', 'vertical'
        ], "region_type should be 'horizontal' or 'vertical' when do entrance counting."
        entrance_x, entrance_y = entrance[0], entrance[1]
        frame_id, tlwhs, tscores, track_ids = result
        for tlwh, score, track_id in zip(tlwhs, tscores, track_ids):
            if track_id < 0: continue
            if data_type == 'kitti':
                frame_id -= 1
            x1, y1, w, h = tlwh
            center_x = x1 + w / 2.
            center_y = y1 + h / 2.
            if track_id in prev_center:
                if region_type == 'horizontal':
                    # horizontal center line
                    if prev_center[track_id][1] <= entrance_y and \
                    center_y > entrance_y:
                        in_id_list.append(track_id)
                    if prev_center[track_id][1] >= entrance_y and \
                    center_y < entrance_y:
                        out_id_list.append(track_id)
                else:
                    # vertical center line
                    if prev_center[track_id][0] <= entrance_x and \
                    center_x > entrance_x:
                        in_id_list.append(track_id)
                    if prev_center[track_id][0] >= entrance_x and \
                    center_x < entrance_x:
                        out_id_list.append(track_id)
                prev_center[track_id][0] = center_x
                prev_center[track_id][1] = center_y
            else:
                prev_center[track_id] = [center_x, center_y]

    if do_break_in_counting:
        assert region_type in [
            'custom'
        ], "region_type should be 'custom' when do break_in counting."
        assert len(
            entrance
        ) >= 4, "entrance should be at least 3 points and (w,h) of image when do break_in counting."
        im_w, im_h = entrance[-1][:]
        entrance = np.array(entrance[:-1])

        frame_id, tlwhs, tscores, track_ids = result
        for tlwh, score, track_id in zip(tlwhs, tscores, track_ids):
            if track_id < 0: continue
            if data_type == 'kitti':
                frame_id -= 1
            x1, y1, w, h = tlwh
            center_x = min(x1 + w / 2., im_w - 1)
            if ids2names[0] == 'pedestrian':
                center_y = min(y1 + h, im_h - 1)
            else:
                center_y = min(y1 + h / 2, im_h - 1)

            # counting objects in region of the first frame
            if frame_id == 1:
                if in_quadrangle([center_x, center_y], entrance, im_h, im_w):
                    in_id_list.append(-1)
                else:
                    prev_center[track_id] = [center_x, center_y]
            else:
                if track_id in prev_center:
                    if not in_quadrangle(prev_center[track_id], entrance, im_h,
                                         im_w) and in_quadrangle(
                                             [center_x, center_y], entrance,
                                             im_h, im_w):
                        in_id_list.append(track_id)
                    prev_center[track_id] = [center_x, center_y]
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
    if do_break_in_counting:
        info += ", Break_in count: {}".format(len(in_id_list))
    if frame_id % video_fps == 0 and frame_id / video_fps % secs_interval == 0:
        info += ", Count during {} secs: {}".format(secs_interval,
                                                    curr_interval_count)
        interval_id_set.clear()
    # print(info)
    info += "\n"
    records.append(info)

    return {
        "id_set": id_set,
        "interval_id_set": interval_id_set,
        "in_id_list": in_id_list,
        "out_id_list": out_id_list,
        "prev_center": prev_center,
        "records": records,
    }


def distance(center_1, center_2):
    return math.sqrt(
        math.pow(center_1[0] - center_2[0], 2) + math.pow(center_1[1] -
                                                          center_2[1], 2))


# update vehicle parking info
def update_object_info(object_in_region_info,
                       result,
                       region_type,
                       entrance,
                       fps,
                       illegal_parking_time,
                       distance_threshold_frame=3,
                       distance_threshold_interval=50):
    '''
    For consecutive frames, the distance between two frame is smaller than distance_threshold_frame, regard as parking
    For parking in general, the move distance should smaller than distance_threshold_interval
    The moving distance of the vehicle is scaled according to the y, which is inversely proportional to y.
    '''

    assert region_type in [
        'custom'
    ], "region_type should be 'custom' when do break_in counting."
    assert len(
        entrance
    ) >= 4, "entrance should be at least 3 points and (w,h) of image when do break_in counting."

    frame_id, tlwhs, tscores, track_ids = result  # result from mot

    im_w, im_h = entrance[-1][:]
    entrance = np.array(entrance[:-1])

    illegal_parking_dict = {}
    for tlwh, score, track_id in zip(tlwhs, tscores, track_ids):
        if track_id < 0: continue

        x1, y1, w, h = tlwh
        center_x = min(x1 + w / 2., im_w - 1)
        center_y = min(y1 + h / 2, im_h - 1)

        if not in_quadrangle([center_x, center_y], entrance, im_h, im_w):
            continue

        current_center = (center_x, center_y)
        if track_id not in object_in_region_info.keys(
        ):  # first time appear in region
            object_in_region_info[track_id] = {}
            object_in_region_info[track_id]["start_frame"] = frame_id
            object_in_region_info[track_id]["end_frame"] = frame_id
            object_in_region_info[track_id]["prev_center"] = current_center
            object_in_region_info[track_id]["start_center"] = current_center
        else:
            prev_center = object_in_region_info[track_id]["prev_center"]

            dis = distance(current_center, prev_center)
            scaled_dis = 200 * dis / (
                current_center[1] + 1)  # scale distance according to y
            dis = scaled_dis

            if dis < distance_threshold_frame:  # not move
                object_in_region_info[track_id]["end_frame"] = frame_id
                object_in_region_info[track_id]["prev_center"] = current_center
            else:  # move
                object_in_region_info[track_id]["start_frame"] = frame_id
                object_in_region_info[track_id]["end_frame"] = frame_id
                object_in_region_info[track_id]["prev_center"] = current_center
                object_in_region_info[track_id]["start_center"] = current_center

        # whether current object parking
        distance_from_start = distance(
            object_in_region_info[track_id]["start_center"], current_center)
        if distance_from_start > distance_threshold_interval:
            # moved
            object_in_region_info[track_id]["start_frame"] = frame_id
            object_in_region_info[track_id]["end_frame"] = frame_id
            object_in_region_info[track_id]["prev_center"] = current_center
            object_in_region_info[track_id]["start_center"] = current_center
            continue

        if (object_in_region_info[track_id]["end_frame"]-object_in_region_info[track_id]["start_frame"]) /fps >= illegal_parking_time \
            and distance_from_start<distance_threshold_interval:
            illegal_parking_dict[track_id] = {"bbox": [x1, y1, w, h]}

    return object_in_region_info, illegal_parking_dict


def in_quadrangle(point, entrance, im_h, im_w):
    mask = np.zeros((im_h, im_w, 1), np.uint8)
    cv2.fillPoly(mask, [entrance], 255)
    p = tuple(map(int, point))
    if mask[p[1], p[0], :] > 0:
        return True
    else:
        return False
