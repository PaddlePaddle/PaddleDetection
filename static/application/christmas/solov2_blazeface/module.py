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
import base64
import json

import cv2
import numpy as np
import paddle.nn as nn
import paddlehub as hub
from paddlehub.module.module import moduleinfo, serving, Module

import solov2_blazeface.processor as P


def cv2_to_base64(image):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')


def base64_to_cv2(b64str):
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data


@moduleinfo(
    name="solov2_blazeface",
    type="CV/image_editing",
    author="paddlepaddle",
    author_email="",
    summary="solov2_blaceface is a segmentation and face detection model based on solov2 and blaceface.",
    version="1.0.0")
class SoloV2BlazeFaceModel(nn.Layer):
    """
    SoloV2BlazeFaceModel
    """

    def __init__(self, use_gpu=True):
        super(SoloV2BlazeFaceModel, self).__init__()
        self.solov2 = hub.Module(name='solov2', use_gpu=use_gpu)
        self.blaceface = hub.Module(name='blazeface', use_gpu=use_gpu)

    def predict(self,
                image,
                background,
                beard_file=None,
                glasses_file=None,
                hat_file=None,
                visualization=False,
                threshold=0.5):
        # instance segmention
        solov2_output = self.solov2.predict(
            image=image, threshold=threshold, visualization=visualization)
        # Set background pixel to 0
        im_segm, x0, x1, y0, y1, _, _, _, _, flag_seg = P.visualize_box_mask(
            image, solov2_output, threshold=threshold)

        if flag_seg == 0:
            return im_segm

        h, w = y1 - y0, x1 - x0
        back_json = background[:-3] + 'json'
        stand_box = json.load(open(back_json))
        stand_box = stand_box['outputs']['object'][0]['bndbox']
        stand_xmin, stand_xmax, stand_ymin, stand_ymax = stand_box[
            'xmin'], stand_box['xmax'], stand_box['ymin'], stand_box['ymax']
        im_path = np.asarray(im_segm)

        # face detection
        blaceface_output = self.blaceface.predict(
            image=im_path, threshold=threshold, visualization=visualization)
        im_face_kp, p_left, p_right, p_up, p_bottom, h_xmin, h_ymin, h_xmax, h_ymax, flag_face = P.visualize_box_mask(
            im_path,
            blaceface_output,
            threshold=threshold,
            beard_file=beard_file,
            glasses_file=glasses_file,
            hat_file=hat_file)
        if flag_face == 1:
            if x0 > h_xmin:
                shift_x_ = x0 - h_xmin
            else:
                shift_x_ = 0
            if y0 > h_ymin:
                shift_y_ = y0 - h_ymin
            else:
                shift_y_ = 0
            h += p_up + p_bottom + shift_y_
            w += p_left + p_right + shift_x_
            x0 = min(x0, h_xmin)
            y0 = min(y0, h_ymin)
            x1 = max(x1, h_xmax) + shift_x_ + p_left + p_right
            y1 = max(y1, h_ymax) + shift_y_ + p_up + p_bottom
        # Fill the background image
        cropped = im_face_kp.crop((x0, y0, x1, y1))
        resize_scale = min((stand_xmax - stand_xmin) / (x1 - x0),
                           (stand_ymax - stand_ymin) / (y1 - y0))
        h, w = int(h * resize_scale), int(w * resize_scale)
        cropped = cropped.resize((w, h), cv2.INTER_LINEAR)
        cropped = cv2.cvtColor(np.asarray(cropped), cv2.COLOR_RGB2BGR)
        shift_x = int((stand_xmax - stand_xmin - cropped.shape[1]) / 2)
        shift_y = int((stand_ymax - stand_ymin - cropped.shape[0]) / 2)
        out_image = cv2.imread(background)
        e2gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(e2gray, 1, 255, cv2.THRESH_BINARY_INV)
        mask_inv = cv2.bitwise_not(mask)
        roi = out_image[stand_ymin + shift_y:stand_ymin + cropped.shape[
            0] + shift_y, stand_xmin + shift_x:stand_xmin + cropped.shape[1] +
                        shift_x]
        person_bg = cv2.bitwise_and(roi, roi, mask=mask)
        element_fg = cv2.bitwise_and(cropped, cropped, mask=mask_inv)
        dst = cv2.add(person_bg, element_fg)
        out_image[stand_ymin + shift_y:stand_ymin + cropped.shape[
            0] + shift_y, stand_xmin + shift_x:stand_xmin + cropped.shape[1] +
                  shift_x] = dst

        return out_image

    @serving
    def serving_method(self, images, background, beard, glasses, hat, **kwargs):
        """
        Run as a service.
        """
        final = {}
        background_path = os.path.join(
            self.directory,
            'element_source/background/{}.png'.format(background))
        beard_path = os.path.join(self.directory,
                                  'element_source/beard/{}.png'.format(beard))
        glasses_path = os.path.join(
            self.directory, 'element_source/glasses/{}.png'.format(glasses))
        hat_path = os.path.join(self.directory,
                                'element_source/hat/{}.png'.format(hat))
        images_decode = base64_to_cv2(images[0])
        output = self.predict(
            image=images_decode,
            background=background_path,
            hat_file=hat_path,
            beard_file=beard_path,
            glasses_file=glasses_path,
            **kwargs)
        final['image'] = cv2_to_base64(output)

        return final
