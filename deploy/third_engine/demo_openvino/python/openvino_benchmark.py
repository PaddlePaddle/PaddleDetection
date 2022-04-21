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

import cv2
import numpy as np
import time
import argparse
from scipy.special import softmax
from openvino.runtime import Core


def image_preprocess(img_path, re_shape):
    img = cv2.imread(img_path)
    img = cv2.resize(
        img, (re_shape, re_shape), interpolation=cv2.INTER_LANCZOS4)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, [2, 0, 1]) / 255
    img = np.expand_dims(img, 0)
    img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std
    return img.astype(np.float32)


def draw_box(img, results, class_label, scale_x, scale_y):

    label_list = list(
        map(lambda x: x.strip(), open(class_label, 'r').readlines()))

    for i in range(len(results)):
        print(label_list[int(results[i][0])], ':', results[i][1])
        bbox = results[i, 2:]
        label_id = int(results[i, 0])
        score = results[i, 1]
        if (score > 0.20):
            xmin, ymin, xmax, ymax = [
                int(bbox[0] * scale_x), int(bbox[1] * scale_y),
                int(bbox[2] * scale_x), int(bbox[3] * scale_y)
            ]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            label_text = label_list[label_id]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymin - 60), (0, 255, 0), -1)
            cv2.putText(img, "#" + label_text, (xmin, ymin - 10), font, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img,
                        str(round(score, 3)), (xmin, ymin - 40), font, 0.8,
                        (255, 255, 255), 2, cv2.LINE_AA)
    return img


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


class PicoDetPostProcess(object):
    """
    Args:
        input_shape (int): network input image size
        ori_shape (int): ori image shape of before padding
        scale_factor (float): scale factor of ori image
        enable_mkldnn (bool): whether to open MKLDNN
    """

    def __init__(self,
                 input_shape,
                 ori_shape,
                 scale_factor,
                 strides=[8, 16, 32, 64],
                 score_threshold=0.4,
                 nms_threshold=0.5,
                 nms_top_k=1000,
                 keep_top_k=100):
        self.ori_shape = ori_shape
        self.input_shape = input_shape
        self.scale_factor = scale_factor
        self.strides = strides
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.nms_top_k = nms_top_k
        self.keep_top_k = keep_top_k

    def warp_boxes(self, boxes, ori_shape):
        """Apply transform to boxes
        """
        width, height = ori_shape[1], ori_shape[0]
        n = len(boxes)
        if n:
            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
                n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            # xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate(
                (x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
            # clip boxes
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
            return xy.astype(np.float32)
        else:
            return boxes

    def __call__(self, scores, raw_boxes):
        batch_size = raw_boxes[0].shape[0]
        reg_max = int(raw_boxes[0].shape[-1] / 4 - 1)
        out_boxes_num = []
        out_boxes_list = []
        for batch_id in range(batch_size):
            # generate centers
            decode_boxes = []
            select_scores = []
            for stride, box_distribute, score in zip(self.strides, raw_boxes,
                                                     scores):
                box_distribute = box_distribute[batch_id]
                score = score[batch_id]
                # centers
                fm_h = self.input_shape[0] / stride
                fm_w = self.input_shape[1] / stride
                h_range = np.arange(fm_h)
                w_range = np.arange(fm_w)
                ww, hh = np.meshgrid(w_range, h_range)
                ct_row = (hh.flatten() + 0.5) * stride
                ct_col = (ww.flatten() + 0.5) * stride
                center = np.stack((ct_col, ct_row, ct_col, ct_row), axis=1)

                # box distribution to distance
                reg_range = np.arange(reg_max + 1)
                box_distance = box_distribute.reshape((-1, reg_max + 1))
                box_distance = softmax(box_distance, axis=1)
                box_distance = box_distance * np.expand_dims(reg_range, axis=0)
                box_distance = np.sum(box_distance, axis=1).reshape((-1, 4))
                box_distance = box_distance * stride

                # top K candidate
                topk_idx = np.argsort(score.max(axis=1))[::-1]
                topk_idx = topk_idx[:self.nms_top_k]
                center = center[topk_idx]
                score = score[topk_idx]
                box_distance = box_distance[topk_idx]

                # decode box
                decode_box = center + [-1, -1, 1, 1] * box_distance

                select_scores.append(score)
                decode_boxes.append(decode_box)

            # nms
            bboxes = np.concatenate(decode_boxes, axis=0)
            confidences = np.concatenate(select_scores, axis=0)
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
                    top_k=self.keep_top_k, )
                picked_box_probs.append(box_probs)
                picked_labels.extend([class_index] * box_probs.shape[0])

            if len(picked_box_probs) == 0:
                out_boxes_list.append(np.empty((0, 4)))
                out_boxes_num.append(0)

            else:
                picked_box_probs = np.concatenate(picked_box_probs)

                # resize output boxes
                picked_box_probs[:, :4] = self.warp_boxes(
                    picked_box_probs[:, :4], self.ori_shape[batch_id])
                im_scale = np.concatenate([
                    self.scale_factor[batch_id][::-1],
                    self.scale_factor[batch_id][::-1]
                ])
                picked_box_probs[:, :4] /= im_scale
                # clas score box
                out_boxes_list.append(
                    np.concatenate(
                        [
                            np.expand_dims(
                                np.array(picked_labels),
                                axis=-1), np.expand_dims(
                                    picked_box_probs[:, 4], axis=-1),
                            picked_box_probs[:, :4]
                        ],
                        axis=1))
                out_boxes_num.append(len(picked_labels))

        out_boxes_list = np.concatenate(out_boxes_list, axis=0)
        out_boxes_num = np.asarray(out_boxes_num).astype(np.int32)
        return out_boxes_list, out_boxes_num


def detect(img_file, compiled_model, re_shape, class_label):
    output = compiled_model.infer_new_request({0: test_image})
    result_ie = list(output.values())  #[0]

    test_im_shape = np.array([[re_shape, re_shape]]).astype('float32')
    test_scale_factor = np.array([[1, 1]]).astype('float32')

    np_score_list = []
    np_boxes_list = []

    num_outs = int(len(result_ie) / 2)
    for out_idx in range(num_outs):
        np_score_list.append(result_ie[out_idx])
        np_boxes_list.append(result_ie[out_idx + num_outs])

    postprocess = PicoDetPostProcess(test_image.shape[2:], test_im_shape,
                                     test_scale_factor)

    np_boxes, np_boxes_num = postprocess(np_score_list, np_boxes_list)

    image = cv2.imread(img_file, 1)
    scale_x = image.shape[1] / test_image.shape[3]
    scale_y = image.shape[0] / test_image.shape[2]
    res_image = draw_box(image, np_boxes, class_label, scale_x, scale_y)

    cv2.imwrite('res.jpg', res_image)
    cv2.imshow("res", res_image)
    cv2.waitKey()


def benchmark(test_image, compiled_model):

    # benchmark       
    loop_num = 100
    warm_up = 8
    timeall = 0
    time_min = float("inf")
    time_max = float('-inf')

    for i in range(loop_num + warm_up):
        time0 = time.time()
        #perform the inference step

        output = compiled_model.infer_new_request({0: test_image})
        time1 = time.time()
        timed = time1 - time0

        if i >= warm_up:
            timeall = timeall + timed
            time_min = min(time_min, timed)
            time_max = max(time_max, timed)

    time_avg = timeall / loop_num

    print('inference_time(ms): min={}, max={}, avg={}'.format(
        round(time_min * 1000, 2),
        round(time_max * 1000, 1), round(time_avg * 1000, 1)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--benchmark', type=int, default=1, help="0:detect; 1:benchmark")
    parser.add_argument(
        '--img_path',
        type=str,
        default='../../../../demo/000000014439.jpg',
        help="image path")
    parser.add_argument(
        '--onnx_path',
        type=str,
        default='out_onnxsim/picodet_s_320_processed.onnx',
        help="onnx filepath")
    parser.add_argument('--in_shape', type=int, default=320, help="input_size")
    parser.add_argument(
        '--class_label',
        type=str,
        default='coco_label.txt',
        help="class label file")
    args = parser.parse_args()

    ie = Core()
    net = ie.read_model(args.onnx_path)
    test_image = image_preprocess(args.img_path, args.in_shape)
    compiled_model = ie.compile_model(net, 'CPU')

    if args.benchmark == 0:
        detect(args.img_path, compiled_model, args.in_shape, args.class_label)
    if args.benchmark == 1:
        benchmark(test_image, compiled_model)
