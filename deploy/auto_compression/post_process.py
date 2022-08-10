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

import numpy as np
import cv2


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(
                current_box, axis=0), )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def area_of(left_top, right_bottom):
    """Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


class PPYOLOEPostProcess(object):
    """
    Args:
        input_shape (int): network input image size
        scale_factor (float): scale factor of ori image
    """

    def __init__(self,
                 score_threshold=0.4,
                 nms_threshold=0.5,
                 nms_top_k=10000,
                 keep_top_k=300):
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.nms_top_k = nms_top_k
        self.keep_top_k = keep_top_k

    def _non_max_suppression(self, prediction, scale_factor):
        batch_size = prediction.shape[0]
        out_boxes_list = []
        box_num_list = []
        for batch_id in range(batch_size):
            bboxes, confidences = prediction[batch_id][..., :4], prediction[
                batch_id][..., 4:]
            # nms
            picked_box_probs = []
            picked_labels = []
            for class_index in range(0, confidences.shape[1]):
                probs = confidences[:, class_index]
                mask = probs > self.score_threshold
                probs = probs[mask]
                if probs.shape[0] == 0:
                    continue
                subset_boxes = bboxes[mask, :]
                box_probs = np.concatenate(
                    [subset_boxes, probs.reshape(-1, 1)], axis=1)
                box_probs = hard_nms(
                    box_probs,
                    iou_threshold=self.nms_threshold,
                    top_k=self.nms_top_k)
                picked_box_probs.append(box_probs)
                picked_labels.extend([class_index] * box_probs.shape[0])

            if len(picked_box_probs) == 0:
                out_boxes_list.append(np.empty((0, 4)))

            else:
                picked_box_probs = np.concatenate(picked_box_probs)
                # resize output boxes
                picked_box_probs[:, 0] /= scale_factor[batch_id][1]
                picked_box_probs[:, 2] /= scale_factor[batch_id][1]
                picked_box_probs[:, 1] /= scale_factor[batch_id][0]
                picked_box_probs[:, 3] /= scale_factor[batch_id][0]

                # clas score box
                out_box = np.concatenate(
                    [
                        np.expand_dims(
                            np.array(picked_labels), axis=-1), np.expand_dims(
                                picked_box_probs[:, 4], axis=-1),
                        picked_box_probs[:, :4]
                    ],
                    axis=1)
                if out_box.shape[0] > self.keep_top_k:
                    out_box = out_box[out_box[:, 1].argsort()[::-1]
                                      [:self.keep_top_k]]
                out_boxes_list.append(out_box)
                box_num_list.append(out_box.shape[0])

        out_boxes_list = np.concatenate(out_boxes_list, axis=0)
        box_num_list = np.array(box_num_list)
        return out_boxes_list, box_num_list

    def __call__(self, outs, scale_factor):
        out_boxes_list, box_num_list = self._non_max_suppression(outs,
                                                                 scale_factor)
        return {'bbox': out_boxes_list, 'bbox_num': box_num_list}
