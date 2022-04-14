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


def get_color_map_list(num_classes):
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    return color_map


def draw_box(srcimg, results, class_label):
    label_list = list(
        map(lambda x: x.strip(), open(class_label, 'r').readlines()))
    for i in range(len(results)):
        color_list = get_color_map_list(len(label_list))
        clsid2color = {}
        classid, conf = int(results[i, 0]), results[i, 1]
        xmin, ymin, xmax, ymax = int(results[i, 2]), int(results[i, 3]), int(
            results[i, 4]), int(results[i, 5])

        if classid not in clsid2color:
            clsid2color[classid] = color_list[classid]
        color = tuple(clsid2color[classid])

        cv2.rectangle(srcimg, (xmin, ymin), (xmax, ymax), color, thickness=2)
        print(label_list[classid] + ': ' + str(round(conf, 3)))
        cv2.putText(
            srcimg,
            label_list[classid] + ':' + str(round(conf, 3)), (xmin, ymin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0),
            thickness=2)
    return srcimg


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


class PicoDetNMS(object):
    """
    Args:
        input_shape (int): network input image size
        scale_factor (float): scale factor of ori image
    """

    def __init__(self,
                 input_shape,
                 scale_x,
                 scale_y,
                 strides=[8, 16, 32, 64],
                 score_threshold=0.4,
                 nms_threshold=0.5,
                 nms_top_k=1000,
                 keep_top_k=100):
        self.input_shape = input_shape
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.strides = strides
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.nms_top_k = nms_top_k
        self.keep_top_k = keep_top_k

    def __call__(self, decode_boxes, select_scores):
        batch_size = 1
        out_boxes_list = []
        for batch_id in range(batch_size):
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

            else:
                picked_box_probs = np.concatenate(picked_box_probs)

                # resize output boxes
                picked_box_probs[:, 0] *= self.scale_x
                picked_box_probs[:, 2] *= self.scale_x
                picked_box_probs[:, 1] *= self.scale_y
                picked_box_probs[:, 3] *= self.scale_y

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

        out_boxes_list = np.concatenate(out_boxes_list, axis=0)
        return out_boxes_list


def detect(img_file, compiled_model, class_label):
    output = compiled_model.infer_new_request({0: test_image})
    result_ie = list(output.values())

    decode_boxes = []
    select_scores = []
    num_outs = int(len(result_ie) / 2)
    for out_idx in range(num_outs):
        decode_boxes.append(result_ie[out_idx])
        select_scores.append(result_ie[out_idx + num_outs])

    image = cv2.imread(img_file, 1)
    scale_x = image.shape[1] / test_image.shape[3]
    scale_y = image.shape[0] / test_image.shape[2]

    nms = PicoDetNMS(test_image.shape[2:], scale_x, scale_y)
    np_boxes = nms(decode_boxes, select_scores)

    res_image = draw_box(image, np_boxes, class_label)

    cv2.imwrite('res.jpg', res_image)
    cv2.imshow("res", res_image)
    cv2.waitKey()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--img_path',
        type=str,
        default='../../demo_onnxruntime/imgs/bus.jpg',
        help="image path")
    parser.add_argument(
        '--onnx_path',
        type=str,
        default='out_onnxsim_infer/picodet_s_320_postproccesed_woNMS.onnx',
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

    detect(args.img_path, compiled_model, args.class_label)
