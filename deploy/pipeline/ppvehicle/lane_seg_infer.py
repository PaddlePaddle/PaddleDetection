# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import codecs
import os

import yaml
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from paddle.inference import create_predictor, PrecisionType
from paddle.inference import Config as PredictConfig


def use_auto_tune(args):
    return hasattr(PredictConfig, "collect_shape_range_info") \
        and hasattr(PredictConfig, "enable_tuned_tensorrt_dynamic_shape") \
        and args['device'] == "gpu" and args['use_trt'] and args['enable_auto_tune']


def auto_tune(args, imgs, img_nums):
    """
    Use images to auto tune the dynamic shape for trt sub graph.
    The tuned shape saved in args.auto_tuned_shape_file.

    Args:
        args(dict): input args.
        imgs(str, list[str], numpy): the path for images or the origin images.
        img_nums(int): the nums of images used for auto tune.
    Returns:
        None
    """
    print("Auto tune the dynamic shape for GPU TRT.")

    assert use_auto_tune(args), "Do not support auto_tune, which requires " \
        "device==gpu && use_trt==True && paddle >= 2.2"

    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
    num = min(len(imgs), img_nums)

    cfg = DeployConfig(args.cfg)
    pred_cfg = PredictConfig(cfg.model, cfg.params)
    pred_cfg.enable_use_gpu(100, 0)

    pred_cfg.collect_shape_range_info(args['auto_tuned_shape_file'])

    predictor = create_predictor(pred_cfg)
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])

    for i in range(0, num):
        if isinstance(imgs[i], str):
            data = {'img': imgs[i]}
            data = np.array([cfg.transforms(data)['img']])
        else:
            data = imgs[i]
        input_handle.reshape(data.shape)
        input_handle.copy_from_cpu(data)
        try:
            predictor.run()
        except Exception as e:
            print(str(e))
            print("Auto tune failed. Usually, the error is out of GPU memory "
                  "for the model or image is too large. \n")
            del predictor
            if os.path.exists(args['auto_tuned_shape_file']):
                os.remove(args['auto_tuned_shape_file'])
            return

    print("Auto tune success.\n")


class DeployConfig:
    def __init__(self, path):
        with codecs.open(path, 'r', 'utf-8') as file:
            self.dic = yaml.load(file, Loader=yaml.FullLoader)

        # self._transforms = Compose()
        self._dir = os.path.dirname(path)

    @property
    def transforms(self):
        return self._transforms

    @property
    def model(self):
        return os.path.join(self._dir, self.dic['Deploy']['model'])

    @property
    def params(self):
        return os.path.join(self._dir, self.dic['Deploy']['params'])


class LaneSegPredictor:
    def __init__(self, lane_seg_config, model_dir, device):
        """
        Prepare for prediction.
        The usage and docs of paddle inference, please refer to
        https://paddleinference.paddlepaddle.org.cn/product_introduction/summary.html
        """
        if not os.path.exists(lane_seg_config):
            raise ValueError("Cannot find : {},".format(lane_seg_config))

        args = yaml.safe_load(open(lane_seg_config))
        self.model_dir = model_dir
        args = args[args['type']]
        cfg_path = os.path.join(self.model_dir, "deploy.yaml")
        if not os.path.exists(cfg_path):
            raise ValueError("Cannot find deploy.yaml in dir: {},".format(
                model_dir))

        self.cfg = DeployConfig(cfg_path)
        self.args = args
        self.shape = None
        self.filter_horizontal_flag = args['filter_horizontal_flag']
        self.horizontal_filtration_degree = args['horizontal_filtration_degree']
        self.horizontal_filtering_threshold = args[
            'horizontal_filtering_threshold']

        self.init_base_config()

        args['device'] = device
        if args['device'] == 'cpu':
            self.init_cpu_config()
        else:
            self.init_gpu_config()

        try:
            self.predictor = create_predictor(self.pred_cfg)
        except Exception as e:
            print(str(e))
            print(
                "If the above error is '(InvalidArgument) some trt inputs dynamic shape info not set, "
                "..., Expected all_dynamic_shape_set == true, ...', "
                "please set --enable_auto_tune=True to use auto_tune. \n")
            exit()

    def init_base_config(self):
        self.pred_cfg = PredictConfig(self.cfg.model, self.cfg.params)

        self.pred_cfg.enable_memory_optim()
        self.pred_cfg.switch_ir_optim(True)

    def init_cpu_config(self):
        """
        Init the config for x86 cpu.
        """
        print("Use CPU")
        self.pred_cfg.disable_gpu()
        if self.args['enable_mkldnn']:
            print("Use MKLDNN")
            # cache 10 different shapes for mkldnn
            self.pred_cfg.set_mkldnn_cache_capacity(10)
            self.pred_cfg.enable_mkldnn()
        self.pred_cfg.set_cpu_math_library_num_threads(self.args['cpu_threads'])

    def init_gpu_config(self):
        """
        Init the config for nvidia gpu.
        """
        print("Use GPU")
        self.pred_cfg.enable_use_gpu(100, 0)
        precision_map = {
            "fp16": PrecisionType.Half,
            "fp32": PrecisionType.Float32,
            "int8": PrecisionType.Int8
        }
        precision_mode = precision_map[self.args['precision']]

        if self.args['use_trt']:
            logger.info("Use TRT")
            self.pred_cfg.enable_tensorrt_engine(
                workspace_size=1 << 30,
                max_batch_size=1,
                min_subgraph_size=self.args.min_subgraph_size,
                precision_mode=precision_mode,
                use_static=False,
                use_calib_mode=False)

            if use_auto_tune(self.args) and \
                os.path.exists(self.args['auto_tuned_shape_file']):
                logger.info("Use auto tuned dynamic shape")
                allow_build_at_runtime = True
                self.pred_cfg.enable_tuned_tensorrt_dynamic_shape(
                    self.args.auto_tuned_shape_file, allow_build_at_runtime)
            else:
                logger.info("Use manual set dynamic shape")
                min_input_shape = {"x": [1, 3, 100, 100]}
                max_input_shape = {"x": [1, 3, 2000, 3000]}
                opt_input_shape = {"x": [1, 3, 512, 1024]}
                self.pred_cfg.set_trt_dynamic_shape_info(
                    min_input_shape, max_input_shape, opt_input_shape)

    def run(self, img):

        input_names = self.predictor.get_input_names()
        input_handle = self.predictor.get_input_handle(input_names[0])
        output_names = self.predictor.get_output_names()
        output_handle = self.predictor.get_output_handle(output_names[0])

        img = np.array(img)
        self.shape = img.shape[1:3]
        img = self.normalize(img)
        img = np.transpose(img, (0, 3, 1, 2))
        input_handle.reshape(img.shape)
        input_handle.copy_from_cpu(img)

        self.predictor.run()

        results = output_handle.copy_to_cpu()
        results = self.postprocess(results)

        return self.get_line(results)

    def normalize(self, im, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        mean = np.array(mean)[np.newaxis, np.newaxis, :]
        std = np.array(std)[np.newaxis, np.newaxis, :]
        im = im.astype(np.float32, copy=False) / 255.0
        im -= mean
        im /= std
        return im

    def postprocess(self, pred):

        pred = np.argmax(pred, axis=1)
        pred[pred == 3] = 0
        pred[pred > 0] = 255

        return pred

    def get_line(self, results):
        lines = []
        directions = []
        for i in range(results.shape[0]):
            line, direction = self.hough_line(np.uint8(results[i]))
            lines.append(line)
            directions.append(direction)
        return lines, directions

    def get_distance(self, array_1, array_2):
        lon_a = array_1[0]
        lat_a = array_1[1]
        lon_b = array_2[0]
        lat_b = array_2[1]

        s = pow(pow((lat_b - lat_a), 2) + pow((lon_b - lon_a), 2), 0.5)
        return s

    def get_angle(self, array):
        import math
        x1, y1, x2, y2 = array
        a_x = x2 - x1
        a_y = y2 - y1
        angle1 = math.atan2(a_y, a_x)
        angle1 = int(angle1 * 180 / math.pi)
        if angle1 > 90:
            angle1 = 180 - angle1
        return angle1

    def get_proportion(self, lines):

        proportion = 0.0
        h, w = self.shape
        for line in lines:
            x1, y1, x2, y2 = line
            length = abs(y2 - y1) / h + abs(x2 - x1) / w
            proportion = proportion + length

        return proportion

    def line_cluster(self, linesP):

        points = []
        for i in range(0, len(linesP)):
            l = linesP[i]
            x_center = (float(
                (max(l[2], l[0]) - min(l[2], l[0]))) / 2.0 + min(l[2], l[0]))
            y_center = (float(
                (max(l[3], l[1]) - min(l[3], l[1]))) / 2.0 + min(l[3], l[1]))
            points.append([x_center, y_center])

        dbscan = DBSCAN(
            eps=50, min_samples=2, metric=self.get_distance).fit(points)

        labels = dbscan.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        cluster_list = list([] for i in range(n_clusters_))
        if linesP is not None:
            for i in range(0, len(linesP)):
                if labels[i] == -1:
                    continue
                l = linesP[i]
                x1, y1, x2, y2 = l
                if y2 >= y1:
                    cluster_list[labels[i]].append([x1, y1, x2, y2])
                else:
                    ll = [x2, y2, x1, y1]
                    cluster_list[labels[i]].append(ll)

        return cluster_list

    def hough_line(self,
                   binary_img,
                   min_line=50,
                   min_line_points=50,
                   max_line_gap=10):
        linesP = cv2.HoughLinesP(binary_img, 1, np.pi / 180, min_line, None,
                                 min_line_points, max_line_gap)
        if linesP is None:
            return [], None

        coarse_cluster_list = self.line_cluster(linesP[:, 0])
        filter_lines_output, direction = self.filter_lines(coarse_cluster_list)

        return filter_lines_output, direction

    def filter_lines(self, coarse_cluster_list):

        lines = []
        angles = []
        for i in range(len(coarse_cluster_list)):
            if len(coarse_cluster_list[i]) == 0:
                continue
            coarse_cluster_list[i] = np.array(coarse_cluster_list[i])
            distance = abs(coarse_cluster_list[i][:, 3] - coarse_cluster_list[i]
                           [:, 1]) + abs(coarse_cluster_list[i][:, 2] -
                                         coarse_cluster_list[i][:, 0])
            l = coarse_cluster_list[i][np.argmax(distance)]
            angles.append(self.get_angle(l))
            lines.append(l)

        if len(lines) == 0:
            return [], None
        if not self.filter_horizontal_flag:
            return lines, None

        #filter horizontal roads
        angles = np.array(angles)

        max_angle, min_angle = np.max(angles), np.min(angles)

        if (max_angle - min_angle) < self.horizontal_filtration_degree:
            return lines, np.mean(angles)

        thr_angle = (
            max_angle + min_angle) * self.horizontal_filtering_threshold
        lines = np.array(lines)

        min_angle_line = lines[np.where(angles < thr_angle)]
        max_angle_line = lines[np.where(angles >= thr_angle)]

        max_angle_line_pro = self.get_proportion(max_angle_line)
        min_angle_line_pro = self.get_proportion(min_angle_line)

        if max_angle_line_pro >= min_angle_line_pro:
            angle_list = angles[np.where(angles >= thr_angle)]
            return max_angle_line, np.mean(angle_list)
        else:
            angle_list = angles[np.where(angles < thr_angle)]
            return min_angle_line, np.mean(angle_list)
