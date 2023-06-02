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
import sys
import cv2
import numpy as np
# add deploy path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

from python.infer import PredictConfig
from pptracking.python.det_infer import load_predictor
from python.utils import Timer


class ReID(object):
    """
    ReID of SDE methods

    Args:
        pred_config (object): config of model, defined by `Config(model_dir)`
        model_dir (str): root path of model.pdiparams, model.pdmodel and infer_cfg.yml
        device (str): Choose the device you want to run, it can be: CPU/GPU/XPU/NPU, default is CPU
        run_mode (str): mode of running(paddle/trt_fp32/trt_fp16)
        batch_size (int): size of per batch in inference, default 50 means at most
            50 sub images can be made a batch and send into ReID model
        trt_min_shape (int): min shape for dynamic shape in trt
        trt_max_shape (int): max shape for dynamic shape in trt
        trt_opt_shape (int): opt shape for dynamic shape in trt
        trt_calib_mode (bool): If the model is produced by TRT offline quantitative
            calibration, trt_calib_mode need to set True
        cpu_threads (int): cpu threads
        enable_mkldnn (bool): whether to open MKLDNN
    """

    def __init__(self,
                 model_dir,
                 device='CPU',
                 run_mode='paddle',
                 batch_size=50,
                 trt_min_shape=1,
                 trt_max_shape=1088,
                 trt_opt_shape=608,
                 trt_calib_mode=False,
                 cpu_threads=4,
                 enable_mkldnn=False):
        self.pred_config = self.set_config(model_dir)
        self.predictor, self.config = load_predictor(
            model_dir,
            run_mode=run_mode,
            batch_size=batch_size,
            min_subgraph_size=self.pred_config.min_subgraph_size,
            device=device,
            use_dynamic_shape=self.pred_config.use_dynamic_shape,
            trt_min_shape=trt_min_shape,
            trt_max_shape=trt_max_shape,
            trt_opt_shape=trt_opt_shape,
            trt_calib_mode=trt_calib_mode,
            cpu_threads=cpu_threads,
            enable_mkldnn=enable_mkldnn)
        self.det_times = Timer()
        self.cpu_mem, self.gpu_mem, self.gpu_util = 0, 0, 0
        self.batch_size = batch_size
        self.input_wh = (128, 256)

    @classmethod
    def init_with_cfg(cls, args, cfg):
        return cls(model_dir=cfg['model_dir'],
                   batch_size=cfg['batch_size'],
                   device=args.device,
                   run_mode=args.run_mode,
                   trt_min_shape=args.trt_min_shape,
                   trt_max_shape=args.trt_max_shape,
                   trt_opt_shape=args.trt_opt_shape,
                   trt_calib_mode=args.trt_calib_mode,
                   cpu_threads=args.cpu_threads,
                   enable_mkldnn=args.enable_mkldnn)

    def set_config(self, model_dir):
        return PredictConfig(model_dir)

    def check_img_quality(self, crop, bbox, xyxy):
        if crop is None:
            return None
        #eclipse
        eclipse_quality = 1.0
        inner_rect = np.zeros(xyxy.shape)
        inner_rect[:, :2] = np.maximum(xyxy[:, :2], bbox[None, :2])
        inner_rect[:, 2:] = np.minimum(xyxy[:, 2:], bbox[None, 2:])
        wh_array = inner_rect[:, 2:] - inner_rect[:, :2]
        filt = np.logical_and(wh_array[:, 0] > 0, wh_array[:, 1] > 0)
        wh_array = wh_array[filt]
        if wh_array.shape[0] > 1:
            eclipse_ratio = wh_array / (bbox[2:] - bbox[:2])
            eclipse_area_ratio = eclipse_ratio[:, 0] * eclipse_ratio[:, 1]
            ear_lst = eclipse_area_ratio.tolist()
            ear_lst.sort(reverse=True)
            eclipse_quality = 1.0 - ear_lst[1]
        bbox_wh = (bbox[2:] - bbox[:2])
        height_quality = bbox_wh[1] / (bbox_wh[0] * 2)
        eclipse_quality = min(eclipse_quality, height_quality)

        #definition
        cropgray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        definition = int(cv2.Laplacian(cropgray, cv2.CV_64F, ksize=3).var())
        brightness = int(cropgray.mean())
        bd_quality = min(1., brightness / 50.)

        eclipse_weight = 0.7
        return eclipse_quality * eclipse_weight + bd_quality * (1 -
                                                                eclipse_weight)

    def normal_crop(self, image, rect):
        imgh, imgw, c = image.shape
        label, conf, xmin, ymin, xmax, ymax = [int(x) for x in rect.tolist()]
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(imgw, xmax)
        ymax = min(imgh, ymax)
        if label != 0 or xmax <= xmin or ymax <= ymin:
            print("Warning! label missed!!")
            return None, None, None
        return image[ymin:ymax, xmin:xmax, :]

    def crop_image_with_mot(self, image, mot_res):
        res = mot_res['boxes']
        crop_res = []
        img_quality = []
        rects = []
        for box in res:
            crop_image = self.normal_crop(image, box[1:])
            quality_item = self.check_img_quality(crop_image, box[3:],
                                                  res[:, 3:])
            if crop_image is not None:
                crop_res.append(crop_image)
                img_quality.append(quality_item)
                rects.append(box)
        return crop_res, img_quality, rects

    def preprocess(self,
                   imgs,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]):
        im_batch = []
        for img in imgs:
            img = cv2.resize(img, self.input_wh)
            img = img.astype('float32') / 255.
            img -= np.array(mean)
            img /= np.array(std)
            im_batch.append(img.transpose((2, 0, 1)))
        inputs = {}
        inputs['x'] = np.array(im_batch).astype('float32')
        return inputs

    def predict(self, crops, repeats=1, add_timer=True, seq_name=''):
        # preprocess
        if add_timer:
            self.det_times.preprocess_time_s.start()
        inputs = self.preprocess(crops)
        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])

        if add_timer:
            self.det_times.preprocess_time_s.end()
            self.det_times.inference_time_s.start()

        # model prediction
        for i in range(repeats):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            feature_tensor = self.predictor.get_output_handle(output_names[0])
            pred_embs = feature_tensor.copy_to_cpu()
        if add_timer:
            self.det_times.inference_time_s.end(repeats=repeats)
            self.det_times.postprocess_time_s.start()

        if add_timer:
            self.det_times.postprocess_time_s.end()
            self.det_times.img_num += 1
        return pred_embs

    def predict_batch(self, imgs, batch_size=4):
        batch_feat = []
        for b in range(0, len(imgs), batch_size):
            b_end = min(len(imgs), b + batch_size)
            batch_imgs = imgs[b:b_end]
            feat = self.predict(batch_imgs)
            batch_feat.extend(feat.tolist())

        return batch_feat
