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

import blazeface.data_feed as D


@moduleinfo(
    name="blazeface",
    type="CV/image_editing",
    author="paddlepaddle",
    author_email="",
    summary="blazeface is a face key point detection model.",
    version="1.0.0")
class Detector(object):
    """
    Args:
        config (object): config of model, defined by `Config(model_dir)`
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

        model_dir = os.path.join(self.directory, 'blazeface_keypoint')
        self.predictor = D.load_predictor(
            model_dir,
            run_mode=run_mode,
            min_subgraph_size=min_subgraph_size,
            use_gpu=use_gpu)

    def face_img_process(self,
                         image,
                         mean=[104., 117., 123.],
                         std=[127.502231, 127.502231, 127.502231]):
        image = np.array(image)
        # HWC to CHW
        if len(image.shape) == 3:
            image = np.swapaxes(image, 1, 2)
            image = np.swapaxes(image, 1, 0)
        # RBG to BGR
        image = image[[2, 1, 0], :, :]
        image = image.astype('float32')
        image -= np.array(mean)[:, np.newaxis, np.newaxis].astype('float32')
        image /= np.array(std)[:, np.newaxis, np.newaxis].astype('float32')
        image = [image]
        image = np.array(image)

        return image

    def transform(self, image, shrink):
        im_info = {
            'scale': [1., 1.],
            'origin_shape': None,
            'resize_shape': None,
            'pad_shape': None,
        }
        if isinstance(image, str):
            with open(image, 'rb') as f:
                im_read = f.read()
            image = np.frombuffer(im_read, dtype='uint8')
            image = cv2.imdecode(image, 1)  # BGR mode, but need RGB mode
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            im_info['origin_shape'] = image.shape[:2]
        else:
            im_info['origin_shape'] = image.shape[:2]

        image_shape = [3, image.shape[0], image.shape[1]]
        h, w = shrink, shrink
        image = cv2.resize(image, (w, h))
        im_info['resize_shape'] = image.shape[:2]

        image = self.face_img_process(image)

        inputs = D.create_inputs(image, im_info)
        return inputs, im_info

    def postprocess(self, boxes_list, lmks_list, im_info, threshold=0.5):
        assert len(boxes_list) == len(lmks_list)
        best_np_boxes, best_np_lmk = boxes_list[0], lmks_list[0]
        for i in range(1, len(boxes_list)):
            #judgment detection score
            if boxes_list[i][0][1] > 0.9:
                break
            face_width = boxes_list[i][0][4] - boxes_list[i][0][2]
            if boxes_list[i][0][1] - best_np_boxes[0][
                    1] > 0.01 and face_width > 0.2:
                best_np_boxes, best_np_lmk = boxes_list[i], lmks_list[i]
        # postprocess output of predictor
        results = {}
        results['landmark'] = D.lmk2out(best_np_boxes, best_np_lmk, im_info,
                                        threshold)

        w, h = im_info['origin_shape']
        best_np_boxes[:, 2] *= h
        best_np_boxes[:, 3] *= w
        best_np_boxes[:, 4] *= h
        best_np_boxes[:, 5] *= w
        expect_boxes = (best_np_boxes[:, 1] > threshold) & (
            best_np_boxes[:, 0] > -1)
        best_np_boxes = best_np_boxes[expect_boxes, :]
        for box in best_np_boxes:
            print('class_id:{:d}, confidence:{:.4f},'
                  'left_top:[{:.2f},{:.2f}],'
                  ' right_bottom:[{:.2f},{:.2f}]'.format(
                      int(box[0]), box[1], box[2], box[3], box[4], box[5]))
        results['boxes'] = best_np_boxes
        return results

    def predict(self,
                image,
                threshold=0.5,
                repeats=1,
                visualization=False,
                with_lmk=True,
                save_dir='blaze_result'):
        '''
        Args:
            image (str/np.ndarray): path of image/ np.ndarray read by cv2
            threshold (float): threshold of predicted box' score
        Returns:
            results (dict): include 'boxes': np.ndarray: shape:[N,6], N: number of box,
                            matix element:[class, score, x_min, y_min, x_max, y_max]
        '''
        shrink = [960, 640, 480, 320, 180]
        boxes_list = []
        lmks_list = []
        for sh in shrink:
            inputs, im_info = self.transform(image, shrink=sh)
            np_boxes, np_lmk = None, None

            input_names = self.predictor.get_input_names()
            for i in range(len(input_names)):
                input_tensor = self.predictor.get_input_tensor(input_names[i])
                input_tensor.copy_from_cpu(inputs[input_names[i]])

            t1 = time.time()
            for i in range(repeats):
                self.predictor.zero_copy_run()
                output_names = self.predictor.get_output_names()
                boxes_tensor = self.predictor.get_output_tensor(output_names[0])
                np_boxes = boxes_tensor.copy_to_cpu()
                if with_lmk == True:
                    face_index = self.predictor.get_output_tensor(output_names[
                        1])
                    landmark = self.predictor.get_output_tensor(output_names[2])
                    prior_boxes = self.predictor.get_output_tensor(output_names[
                        3])
                    np_face_index = face_index.copy_to_cpu()
                    np_prior_boxes = prior_boxes.copy_to_cpu()
                    np_landmark = landmark.copy_to_cpu()
                    np_lmk = [np_face_index, np_landmark, np_prior_boxes]

            t2 = time.time()
            ms = (t2 - t1) * 1000.0 / repeats
            print("Inference: {} ms per batch image".format(ms))

            # do not perform postprocess in benchmark mode
            results = []
            if reduce(lambda x, y: x * y, np_boxes.shape) < 6:
                print('[WARNNING] No object detected.')
                results = {'boxes': np.array([])}
            else:
                boxes_list.append(np_boxes)
                lmks_list.append(np_lmk)

        results = self.postprocess(
            boxes_list, lmks_list, im_info, threshold=threshold)

        if visualization:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            output = D.visualize_box_mask(
                im=image, results=results, labels=["background", "face"])
            name = str(time.time()) + '.png'
            save_path = os.path.join(save_dir, name)
            output.save(save_path)
            img = cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR)
            results['image'] = img

        return results
