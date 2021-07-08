# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import time
from functools import reduce
import cv2
import numpy as np
from paddlehub.module.module import moduleinfo
import solov2.processor as P
import solov2.data_feed as D


class Detector(object):
    """
    Args:
        model_dir (str): root path of __model__, __params__ and infer_cfg.yml
        use_gpu (bool): whether use gpu
        run_mode (str): mode of running(fluid/trt_fp32/trt_fp16)
        threshold (float): threshold to reserve the result for output.
    """

    def __init__(self,
                 min_subgraph_size=60,
                 use_gpu=False,
                 run_mode='fluid',
                 threshold=0.5):

        model_dir = os.path.join(self.directory, 'solov2_r101_vd_fpn_3x')
        self.predictor = D.load_predictor(
            model_dir,
            run_mode=run_mode,
            min_subgraph_size=min_subgraph_size,
            use_gpu=use_gpu)
        self.compose = [
            P.Resize(max_size=1333), P.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            P.Permute(), P.PadStride(stride=32)
        ]

    def transform(self, im):
        im, im_info = P.preprocess(im, self.compose)
        inputs = D.create_inputs(im, im_info)
        return inputs, im_info

    def postprocess(self, np_boxes, np_masks, im_info, threshold=0.5):
        # postprocess output of predictor
        results = {}
        expect_boxes = (np_boxes[:, 1] > threshold) & (np_boxes[:, 0] > -1)
        np_boxes = np_boxes[expect_boxes, :]
        for box in np_boxes:
            print('class_id:{:d}, confidence:{:.4f},'
                  'left_top:[{:.2f},{:.2f}],'
                  ' right_bottom:[{:.2f},{:.2f}]'.format(
                      int(box[0]), box[1], box[2], box[3], box[4], box[5]))
        results['boxes'] = np_boxes
        if np_masks is not None:
            np_masks = np_masks[expect_boxes, :, :, :]
            results['masks'] = np_masks
        return results

    def predict(self, image, threshold=0.5, warmup=0, repeats=1):
        '''
        Args:
            image (str/np.ndarray): path of image/ np.ndarray read by cv2
            threshold (float): threshold of predicted box' score
        Returns:
            results (dict): include 'boxes': np.ndarray: shape:[N,6], N: number of box,
                            matix element:[class, score, x_min, y_min, x_max, y_max]
                            MaskRCNN's results include 'masks': np.ndarray:
                            shape:[N, class_num, mask_resolution, mask_resolution]
        '''
        inputs, im_info = self.transform(image)
        np_boxes, np_masks = None, None

        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_tensor(input_names[i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])

        for i in range(warmup):
            self.predictor.zero_copy_run()
            output_names = self.predictor.get_output_names()
            boxes_tensor = self.predictor.get_output_tensor(output_names[0])
            np_boxes = boxes_tensor.copy_to_cpu()

        for i in range(repeats):
            self.predictor.zero_copy_run()
            output_names = self.predictor.get_output_names()
            boxes_tensor = self.predictor.get_output_tensor(output_names[0])
            np_boxes = boxes_tensor.copy_to_cpu()

        # do not perform postprocess in benchmark mode
        results = []

        if reduce(lambda x, y: x * y, np_boxes.shape) < 6:
            print('[WARNNING] No object detected.')
            results = {'boxes': np.array([])}
        else:
            results = self.postprocess(
                np_boxes, np_masks, im_info, threshold=threshold)

        return results


@moduleinfo(
    name="solov2",
    type="CV/image_editing",
    author="paddlepaddle",
    author_email="",
    summary="solov2 is a detection model, this module is trained with COCO dataset.",
    version="1.0.0")
class DetectorSOLOv2(Detector):
    def __init__(self, use_gpu=False, run_mode='fluid', threshold=0.5):
        super(DetectorSOLOv2, self).__init__(
            use_gpu=use_gpu, run_mode=run_mode, threshold=threshold)

    def predict(self,
                image,
                threshold=0.5,
                warmup=0,
                repeats=1,
                visualization=False,
                save_dir='solov2_result'):
        inputs, im_info = self.transform(image)
        np_label, np_score, np_segms = None, None, None

        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_tensor(input_names[i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])
        for i in range(warmup):
            self.predictor.zero_copy_run()
            output_names = self.predictor.get_output_names()
            np_label = self.predictor.get_output_tensor(output_names[
                0]).copy_to_cpu()
            np_score = self.predictor.get_output_tensor(output_names[
                1]).copy_to_cpu()
            np_segms = self.predictor.get_output_tensor(output_names[
                2]).copy_to_cpu()

        for i in range(repeats):
            self.predictor.zero_copy_run()
            output_names = self.predictor.get_output_names()
            np_label = self.predictor.get_output_tensor(output_names[
                0]).copy_to_cpu()
            np_score = self.predictor.get_output_tensor(output_names[
                1]).copy_to_cpu()
            np_segms = self.predictor.get_output_tensor(output_names[
                2]).copy_to_cpu()
        output = dict(segm=np_segms, label=np_label, score=np_score)

        if visualization:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            image = D.visualize_box_mask(im=image, results=output)
            name = str(time.time()) + '.png'
            save_path = os.path.join(save_dir, name)
            image.save(save_path)
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        output['image'] = img
        return output
