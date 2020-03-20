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
import argparse
import yaml

import cv2
import numpy as np
import paddle.fluid as fluid
from visualize import visualize_box_mask


def decode_image(im_path, im_info):
    """read rgb image
    Args:
        im_path (str): path of image
        im_info (dict): info of image
    Returns:
        im (np.ndarray):  processed image (np.ndarray)
        im_info (dict): info of processed image
    """
    with open(im_path, 'rb') as f:
        im = f.read()
    data = np.frombuffer(im, dtype='uint8')
    im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_info['origin_shape'] = im.shape[:2]
    im_info['resize_shape'] = im.shape[:2]
    return im, im_info


class Resize(object):
    """resize image by target_size and max_size
    Args:
        arch (str): model type
        target_size (int): the target size of image
        max_size (int): the max size of image
        use_cv2 (bool): whether us cv2
        interp (int): method of resize
    """
    def __init__(self,
                 arch,
                 target_size,
                 max_size=0,
                 use_cv2=True,
                 interp=cv2.INTER_LINEAR):
        self.target_size = target_size
        self.max_size = max_size
        self.interp = interp
        self.arch = arch
        self.scale_set = {'RCNN', 'RetinaNet'}

    def __call__(self, im, im_info):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        im_scale_x, im_scale_y = self.generate_scale(im)
        im = cv2.resize(im,
                        None,
                        None,
                        fx=im_scale_x,
                        fy=im_scale_y,
                        interpolation=self.interp)
        if self.arch in self.scale_set:
            im_info['scale'] = im_scale_x
        im_info['resize_shape'] = im.shape[:2]
        return im, im_info

    def generate_scale(self, im):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
        Returns:
            im_scale_x: the resize ratio of X 
            im_scale_y: the resize ratio of Y 
        """
        origin_shape = im.shape[:2]
        im_c = im.shape[2]
        if self.max_size != 0 and self.arch in self.scale_set:
            im_size_min = np.min(origin_shape[0:2])
            im_size_max = np.max(origin_shape[0:2])
            im_scale = float(self.target_size) / float(im_size_min)
            if np.round(im_scale * im_size_max) > self.max_size:
                im_scale = float(self.max_size) / float(im_size_max)
            im_scale_x = im_scale
            im_scale_y = im_scale
        else:
            im_scale_x = float(self.target_size) / float(origin_shape[1])
            im_scale_y = float(self.target_size) / float(origin_shape[0])
        return im_scale_x, im_scale_y


class Normalize(object):
    """normalize image
    Args:
        mean (list): im - mean
        std (list): im / std
        is_scale (bool): whether need im / 255
        is_channel_first (bool): if True: image shape is CHW, else: HWC
    """
    def __init__(self, mean, std, is_scale=True, is_channel_first=False):
        self.mean = mean
        self.std = std
        self.is_scale = is_scale
        self.is_channel_first = is_channel_first

    def __call__(self, im, im_info):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        im = im.astype(np.float32, copy=False)
        if self.is_channel_first:
            mean = np.array(self.mean)[:, np.newaxis, np.newaxis]
            std = np.array(self.std)[:, np.newaxis, np.newaxis]
        else:
            mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
            std = np.array(self.std)[np.newaxis, np.newaxis, :]
        if self.is_scale:
            im = im / 255.0
        im -= mean
        im /= std
        return im, im_info


class Permute(object):
    """permute image
    Args:
        to_bgr (bool): whether convert RGB to BGR 
        channel_first (bool): whether convert HWC to CHW
    """
    def __init__(self, to_bgr=False, channel_first=True):
        self.to_bgr = to_bgr
        self.channel_first = channel_first

    def __call__(self, im, im_info):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        if self.channel_first:
            im = im.transpose((2, 0, 1)).copy()
        if self.to_bgr:
            im = im[[2, 1, 0], :, :]
        return im, im_info


class PadStride(object):
    """ padding image for model with FPN 
    Args:
        stride (bool): model with FPN need image shape % stride == 0 
    """
    def __init__(self, stride=0):
        self.coarsest_stride = stride

    def __call__(self, im, im_info):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        coarsest_stride = self.coarsest_stride
        if coarsest_stride == 0:
            return im
        im_c, im_h, im_w = im.shape
        pad_h = int(np.ceil(float(im_h) / coarsest_stride) * coarsest_stride)
        pad_w = int(np.ceil(float(im_w) / coarsest_stride) * coarsest_stride)
        padding_im = np.zeros((im_c, pad_h, pad_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = im
        im_info['resize_shape'] = padding_im.shape[1:]
        return padding_im, im_info


def create_inputs(im, im_info, model_arch='YOLO'):
    """generate input for different model type
    Args:
        im (np.ndarray): image (np.ndarray)
        im_info (dict): info of image
        model_arch (str): model type
    Returns:
        inputs (list): input of model
    """
    inputs = []
    inputs.append(im)
    origin_shape = list(im_info['origin_shape'])
    resize_shape = list(im_info['resize_shape'])
    scale = im_info['scale']
    info = []
    if 'YOLO' in model_arch:
        im_size = np.array([origin_shape]).astype('int32')
        info.append(im_size)
    elif 'RetinaNet' in model_arch:
        im_info = np.array([resize_shape + [scale]]).astype('float32')
        info.append(im_info)
    elif 'RCNN' in model_arch:
        im_info = np.array([resize_shape + [scale]]).astype('float32')
        im_shape = np.array([origin_shape + [1.]]).astype('float32')
        info.append(im_info)
        info.append(im_shape)
    inputs += info
    return inputs


class Config():
    """set config of preprocess, postprocess and visualize
    Args:
        model_dir (str): root path of model.yml
    """
    support_models = ['YOLO', 'SSD', 'RetinaNet', 'RCNN']

    def __init__(self, model_dir):
        # parsing Yaml config for Preprocess
        deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        self.check_model(yml_conf)
        self.arch = yml_conf['arch']
        # preprocess info
        self.preprocess_infos = yml_conf['Preprocess']

        self.labels = yml_conf['label_list']
        if 'YOLO' in self.arch:
            self.labels = self.labels[1:]
        self.mask_resolution = None
        if 'mask_resolution' in yml_conf:
            self.mask_resolution = yml_conf['mask_resolution']

    def check_model(self, yml_conf):
        """
        Raises:
            ValueError: loaded model not in supported model type 
        """
        for support_model in self.support_models:
            if support_model in yml_conf['arch']:
                return True
        raise ValueError(
            "Unsupported arch: {}, expect SSD, YOLO, RetinaNet and RCNN".format(
                yml_conf['arch']))


def load_model(model_dir, use_gpu=False):
    """set AnalysisConfig，generate AnalysisPredictor
    Args:
        model_dir (str): root path of __model__ and __params__
        use_gpu (bool): whether use gpu
    Returns:
        predictor (PaddlePredictor): AnalysisPredictor
    """
    config = fluid.core.AnalysisConfig(os.path.join(model_dir, '__model__'),
                                       os.path.join(model_dir, '__params__'))
    if use_gpu:
        # initial GPU memory(M), device ID
        config.enable_use_gpu(100, 0)
        # optimize graph and fuse op
        config.switch_ir_optim(True)
    else:
        config.disable_gpu()

    # disable print log when predict
    config.disable_glog_info()
    # enable shared memory
    config.enable_memory_optim()
    # disable feed, fetch OP，needed by zero_copy_run
    config.switch_use_feed_fetch_ops(False)
    predictor = fluid.core.create_paddle_predictor(config)
    return predictor


class Detector():
    """
    Args:
        model_dir (str): root path of __model__, __params__ and infer_cfg.yml
        use_gpu (bool): whether use gpu
    """
    def __init__(self, model_dir, use_gpu=False, threshold=0.5):
        self.config = Config(model_dir)
        self.predictor = load_model(model_dir, use_gpu=use_gpu)
        self.preprocess_ops = []
        for op_info in self.config.preprocess_infos:
            op_type = op_info.pop('type')
            if op_type == 'Resize':
                op_info['arch'] = self.config.arch
            self.preprocess_ops.append(eval(op_type)(**op_info))

    def preprocess(self, im):
        # process image by preprocess_ops
        im_info = {
            'scale': 1.,
            'origin_shape': None,
            'resize_shape': None,
        }
        im, im_info = decode_image(im, im_info)
        for operator in self.preprocess_ops:
            im, im_info = operator(im, im_info)
        im = np.array((im, )).astype('float32')
        inputs = create_inputs(im, im_info, self.config.arch)
        return inputs, im_info

    def postprocess(self, im_info, threshold=0.5):
        # postprocess output of predictor
        results = {}
        output_names = self.predictor.get_output_names()
        boxes_tensor = self.predictor.get_output_tensor(output_names[0])
        np_boxes = boxes_tensor.copy_to_cpu()

        if 'SSD' in self.config.arch:
            w, h = im_info['origin_shape']
            np_boxes[:, 2] *= h
            np_boxes[:, 3] *= w
            np_boxes[:, 4] *= h
            np_boxes[:, 5] *= w
        expect_boxes = np_boxes[:, 1] > threshold
        np_boxes = np_boxes[expect_boxes, :]
        for box in np_boxes:
            print('class_id:{:d}, confidence:{:.2f},'
                  'left_top:[{:.2f},{:.2f}],'
                  ' right_bottom:[{:.2f},{:.2f}]'.format(
                      int(box[0]), box[1], box[2], box[3], box[4], box[5]))
        results['boxes'] = np_boxes

        if self.config.mask_resolution is not None:
            masks_tensor = self.predictor.get_output_tensor(output_names[1])
            np_masks = masks_tensor.copy_to_cpu()
            np_masks = np_masks[expect_boxes, :, :, :]
            results['masks'] = np_masks
        return results

    def predict(self, image_file, threshold=0.5):
        '''
        Args:
            image_file (str): path of image
            threshold (float): threshold of predicted box' score
        Returns:
            results (dict): include 'boxes': np.ndarray: shape:[N,6], N: number of box，
                            matix element:[class, score, x_min, y_min, x_max, y_max]
                            MaskRCNN's results include 'masks': np.ndarray: 
                            shape:[N, class_num, mask_resolution, mask_resolution]  
        '''
        inputs, im_info = self.preprocess(image_file)
        input_names = self.predictor.get_input_names()
        for i in range(len(inputs)):
            input_tensor = self.predictor.get_input_tensor(input_names[i])
            input_tensor.copy_from_cpu(inputs[i])
        self.predictor.zero_copy_run()
        results = self.postprocess(im_info, threshold)
        return results

    def visualize(self, image_file, results, output_dir):
        # visualize the predict result
        im = visualize_box_mask(image_file,
                                results,
                                self.config.labels,
                                mask_resolution=self.config.mask_resolution)
        img_name = os.path.split(image_file)[-1]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        out_path = os.path.join(output_dir, img_name)
        im.save(out_path, quality=95)
        print("save result to: " + out_path)


def main():
    detector = Detector(FLAGS.model_dir, use_gpu=FLAGS.use_gpu)
    results = detector.predict(FLAGS.image_file, FLAGS.threshold)

    if FLAGS.visualize:
        detector.visualize(FLAGS.image_file, results, FLAGS.output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir",
                        type=str,
                        default=None,
                        help=("Directory include:'__model__', '__params__', "
                              "'infer_cfg.yml', created by export_model."),
                        required=True)
    parser.add_argument("--image_file",
                        type=str,
                        default=None,
                        help="Path of image file.",
                        required=True)
    parser.add_argument("--use_gpu",
                        default=False,
                        help="Whether to predict with GPU.")
    parser.add_argument("--threshold",
                        type=float,
                        default=0.5,
                        help="Threshold of score.")
    parser.add_argument("--visualize",
                        default=False,
                        help="Whether to visualize detection output.")
    parser.add_argument("--output_dir",
                        type=str,
                        default="output",
                        help="Directory of output visualization files.")
    FLAGS = parser.parse_args()
    main()
