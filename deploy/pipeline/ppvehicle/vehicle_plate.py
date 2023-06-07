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
import glob
from functools import reduce

import time
import cv2
import numpy as np
import math
import paddle

import sys
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
sys.path.insert(0, parent_path)

from python.infer import get_test_images
from python.preprocess import preprocess, NormalizeImage, Permute, Resize_Mult32
from pipeline.ppvehicle.vehicle_plateutils import create_predictor, get_infer_gpuid, get_rotate_crop_image, draw_boxes
from pipeline.ppvehicle.vehicleplate_postprocess import build_post_process
from pipeline.cfg_utils import merge_cfg, print_arguments, argsparser


class PlateDetector(object):
    def __init__(self, args, cfg):
        self.args = args
        self.pre_process_list = {
            'Resize_Mult32': {
                'limit_side_len': cfg['det_limit_side_len'],
                'limit_type': cfg['det_limit_type'],
            },
            'NormalizeImage': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'is_scale': True,
            },
            'Permute': {}
        }
        postprocess_params = {}
        postprocess_params['name'] = 'DBPostProcess'
        postprocess_params["thresh"] = 0.3
        postprocess_params["box_thresh"] = 0.6
        postprocess_params["max_candidates"] = 1000
        postprocess_params["unclip_ratio"] = 1.5
        postprocess_params["use_dilation"] = False
        postprocess_params["score_mode"] = "fast"

        self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.input_tensor, self.output_tensors, self.config = create_predictor(
            args, cfg, 'det')

    def preprocess(self, im_path):
        preprocess_ops = []
        for op_type, new_op_info in self.pre_process_list.items():
            preprocess_ops.append(eval(op_type)(**new_op_info))

        input_im_lst = []
        input_im_info_lst = []

        im, im_info = preprocess(im_path, preprocess_ops)
        input_im_lst.append(im)
        input_im_info_lst.append(im_info['im_shape'] / im_info['scale_factor'])

        return np.stack(input_im_lst, axis=0), input_im_info_lst

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def predict_image(self, img_list):
        st = time.time()

        dt_batch_boxes = []
        for image in img_list:
            img, shape_list = self.preprocess(image)
            if img is None:
                return None, 0
            self.input_tensor.copy_from_cpu(img)
            self.predictor.run()
            outputs = []
            for output_tensor in self.output_tensors:
                output = output_tensor.copy_to_cpu()
                outputs.append(output)

            preds = {}
            preds['maps'] = outputs[0]

            #self.predictor.try_shrink_memory()
            post_result = self.postprocess_op(preds, shape_list)
            # print("post_result length:{}".format(len(post_result)))

            org_shape = image.shape
            dt_boxes = post_result[0]['points']
            dt_boxes = self.filter_tag_det_res(dt_boxes, org_shape)
            dt_batch_boxes.append(dt_boxes)

        et = time.time()
        return dt_batch_boxes, et - st


class TextRecognizer(object):
    def __init__(self, args, cfg, use_gpu=True):
        self.rec_image_shape = cfg['rec_image_shape']
        self.rec_batch_num = cfg['rec_batch_num']
        word_dict_path = cfg['word_dict_path']
        use_space_char = True

        postprocess_params = {
            'name': 'CTCLabelDecode',
            "character_dict_path": word_dict_path,
            "use_space_char": use_space_char
        }
        self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.input_tensor, self.output_tensors, self.config = \
            create_predictor(args, cfg, 'rec')
        self.use_onnx = False

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape

        assert imgC == img.shape[2]
        imgW = int((imgH * max_wh_ratio))
        if self.use_onnx:
            w = self.input_tensor.shape[3:][0]
            if w is not None and w > 0:
                imgW = w

        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def predict_text(self, img_list):
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))
        rec_res = [['', 0.0]] * img_num
        batch_num = self.rec_batch_num
        st = time.time()
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            imgC, imgH, imgW = self.rec_image_shape
            max_wh_ratio = imgW / imgH
            # max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]],
                                                max_wh_ratio)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()
            if self.use_onnx:
                input_dict = {}
                input_dict[self.input_tensor.name] = norm_img_batch
                outputs = self.predictor.run(self.output_tensors, input_dict)
                preds = outputs[0]
            else:
                self.input_tensor.copy_from_cpu(norm_img_batch)
                self.predictor.run()
                outputs = []
                for output_tensor in self.output_tensors:
                    output = output_tensor.copy_to_cpu()
                    outputs.append(output)
                if len(outputs) != 1:
                    preds = outputs
                else:
                    preds = outputs[0]
            rec_result = self.postprocess_op(preds)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]
        return rec_res, time.time() - st


class PlateRecognizer(object):
    def __init__(self, args, cfg):
        use_gpu = args.device.lower() == "gpu"
        self.platedetector = PlateDetector(args, cfg)
        self.textrecognizer = TextRecognizer(args, cfg, use_gpu=use_gpu)

    def get_platelicense(self, image_list):
        plate_text_list = []
        plateboxes, det_time = self.platedetector.predict_image(image_list)
        for idx, boxes_pcar in enumerate(plateboxes):
            plate_pcar_list = []
            for box in boxes_pcar:
                plate_images = get_rotate_crop_image(image_list[idx], box)
                plate_texts = self.textrecognizer.predict_text([plate_images])
                plate_pcar_list.append(plate_texts)
            plate_text_list.append(plate_pcar_list)
        return self.check_plate(plate_text_list)

    def check_plate(self, text_list):
        plate_all = {"plate": []}
        for text_pcar in text_list:
            platelicense = ""
            for text_info in text_pcar:
                text = text_info[0][0][0]
                if len(text) > 2 and len(text) < 10:
                    platelicense = self.replace_cn_code(text)
            plate_all["plate"].append(platelicense)
        return plate_all

    def replace_cn_code(self, text):
        simcode = {
            '浙': 'ZJ-',
            '粤': 'GD-',
            '京': 'BJ-',
            '津': 'TJ-',
            '冀': 'HE-',
            '晋': 'SX-',
            '蒙': 'NM-',
            '辽': 'LN-',
            '黑': 'HLJ-',
            '沪': 'SH-',
            '吉': 'JL-',
            '苏': 'JS-',
            '皖': 'AH-',
            '赣': 'JX-',
            '鲁': 'SD-',
            '豫': 'HA-',
            '鄂': 'HB-',
            '湘': 'HN-',
            '桂': 'GX-',
            '琼': 'HI-',
            '渝': 'CQ-',
            '川': 'SC-',
            '贵': 'GZ-',
            '云': 'YN-',
            '藏': 'XZ-',
            '陕': 'SN-',
            '甘': 'GS-',
            '青': 'QH-',
            '宁': 'NX-',
            '闽': 'FJ-',
            '·': ' '
        }
        for _char in text:
            if _char in simcode:
                text = text.replace(_char, simcode[_char])
        return text


def main():
    cfg = merge_cfg(FLAGS)
    print_arguments(cfg)
    vehicleplate_cfg = cfg['VEHICLE_PLATE']
    detector = PlateRecognizer(FLAGS, vehicleplate_cfg)
    # predict from image
    img_list = get_test_images(FLAGS.image_dir, FLAGS.image_file)
    for img in img_list:
        image = cv2.imread(img)
        results = detector.get_platelicense([image])
        print(results)


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()
    FLAGS.device = FLAGS.device.upper()
    assert FLAGS.device in ['CPU', 'GPU', 'XPU', 'NPU'
                            ], "device should be CPU, GPU, NPU or XPU"

    main()
