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
import yaml
import glob
from functools import reduce

from PIL import Image
import cv2
import numpy as np
import math
import paddle
from paddle.inference import Config
from paddle.inference import create_predictor

from benchmark_utils import PaddleInferBenchmark
from preprocess import preprocess, Resize, NormalizeImage, Permute, PadStride
from visualize import visualize_box_mask
from utils import argsparser, Timer, get_current_memory_mb

# Global dictionary
SUPPORT_MODELS = {
    'YOLO',
    'RCNN',
    'SSD',
    'FCOS',
    'SOLOv2',
    'TTFNet',
    'S2ANet',
}


class Detector(object):
    """
    Args:
        config (object): config of model, defined by `Config(model_dir)`
        model_dir (str): root path of model.pdiparams, model.pdmodel and infer_cfg.yml
        use_gpu (bool): whether use gpu
        run_mode (str): mode of running(fluid/trt_fp32/trt_fp16)
        batch_size (int): size of pre batch in inference
        use_dynamic_shape (bool): use dynamic shape or not
        trt_min_shape (int): min shape for dynamic shape in trt
        trt_max_shape (int): max shape for dynamic shape in trt
        trt_opt_shape (int): opt shape for dynamic shape in trt
        run_mode (str): mode of running(fluid/trt_fp32/trt_fp16)
        threshold (float): threshold to reserve the result for output.
    """

    def __init__(self,
                 pred_config,
                 model_dir,
                 use_gpu=False,
                 run_mode='fluid',
                 batch_size=1,
                 use_dynamic_shape=False,
                 trt_min_shape=1,
                 trt_max_shape=1280,
                 trt_opt_shape=640,
                 trt_calib_mode=False,
                 cpu_threads=1,
                 enable_mkldnn=False):
        self.pred_config = pred_config
        self.predictor, self.config = load_predictor(
            model_dir,
            run_mode=run_mode,
            batch_size=batch_size,
            min_subgraph_size=self.pred_config.min_subgraph_size,
            use_gpu=use_gpu,
            use_dynamic_shape=use_dynamic_shape,
            trt_min_shape=trt_min_shape,
            trt_max_shape=trt_max_shape,
            trt_opt_shape=trt_opt_shape,
            trt_calib_mode=trt_calib_mode,
            cpu_threads=cpu_threads,
            enable_mkldnn=enable_mkldnn)
        self.det_times = Timer()
        self.cpu_mem, self.gpu_mem, self.gpu_util = 0, 0, 0

    def preprocess(self, image_list):
        preprocess_ops = []
        for op_info in self.pred_config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            preprocess_ops.append(eval(op_type)(**new_op_info))

        input_im_lst = []
        input_im_info_lst = []
        for im_path in image_list:
            im, im_info = preprocess(im_path, preprocess_ops)
            input_im_lst.append(im)
            input_im_info_lst.append(im_info)
        inputs = create_inputs(input_im_lst, input_im_info_lst)
        return inputs

    def postprocess(self,
                    np_boxes,
                    np_masks,
                    inputs,
                    np_boxes_num,
                    threshold=0.5):
        # postprocess output of predictor
        results = {}
        if self.pred_config.arch in ['Face']:
            h, w = inputs['im_shape']
            scale_y, scale_x = inputs['scale_factor']
            w, h = float(h) / scale_y, float(w) / scale_x
            np_boxes[:, 2] *= h
            np_boxes[:, 3] *= w
            np_boxes[:, 4] *= h
            np_boxes[:, 5] *= w
        results['boxes'] = np_boxes
        results['boxes_num'] = np_boxes_num
        if np_masks is not None:
            results['masks'] = np_masks
        return results

    def predict(self, image_list, threshold=0.5, warmup=0, repeats=1):
        '''
        Args:
            image_list (list): ,list of image
            threshold (float): threshold of predicted box' score
        Returns:
            results (dict): include 'boxes': np.ndarray: shape:[N,6], N: number of box,
                            matix element:[class, score, x_min, y_min, x_max, y_max]
                            MaskRCNN's results include 'masks': np.ndarray:
                            shape: [N, im_h, im_w]
        '''
        self.det_times.preprocess_time_s.start()
        inputs = self.preprocess(image_list)
        self.det_times.preprocess_time_s.end()
        np_boxes, np_masks = None, None
        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])
        for i in range(warmup):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            boxes_tensor = self.predictor.get_output_handle(output_names[0])
            np_boxes = boxes_tensor.copy_to_cpu()
            if self.pred_config.mask:
                masks_tensor = self.predictor.get_output_handle(output_names[2])
                np_masks = masks_tensor.copy_to_cpu()

        self.det_times.inference_time_s.start()
        for i in range(repeats):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            boxes_tensor = self.predictor.get_output_handle(output_names[0])
            np_boxes = boxes_tensor.copy_to_cpu()
            boxes_num = self.predictor.get_output_handle(output_names[1])
            np_boxes_num = boxes_num.copy_to_cpu()
            if self.pred_config.mask:
                masks_tensor = self.predictor.get_output_handle(output_names[2])
                np_masks = masks_tensor.copy_to_cpu()
        self.det_times.inference_time_s.end(repeats=repeats)

        self.det_times.postprocess_time_s.start()
        results = []
        if reduce(lambda x, y: x * y, np_boxes.shape) < 6:
            print('[WARNNING] No object detected.')
            results = {'boxes': np.array([]), 'boxes_num': [0]}
        else:
            results = self.postprocess(
                np_boxes, np_masks, inputs, np_boxes_num, threshold=threshold)
        self.det_times.postprocess_time_s.end()
        self.det_times.img_num += len(image_list)
        return results


class DetectorSOLOv2(Detector):
    """
    Args:
        config (object): config of model, defined by `Config(model_dir)`
        model_dir (str): root path of model.pdiparams, model.pdmodel and infer_cfg.yml
        use_gpu (bool): whether use gpu
        run_mode (str): mode of running(fluid/trt_fp32/trt_fp16)
        batch_size (int): size of pre batch in inference
        use_dynamic_shape (bool): use dynamic shape or not
        trt_min_shape (int): min shape for dynamic shape in trt
        trt_max_shape (int): max shape for dynamic shape in trt
        trt_opt_shape (int): opt shape for dynamic shape in trt
        threshold (float): threshold to reserve the result for output.
    """

    def __init__(self,
                 pred_config,
                 model_dir,
                 use_gpu=False,
                 run_mode='fluid',
                 batch_size=1,
                 use_dynamic_shape=False,
                 trt_min_shape=1,
                 trt_max_shape=1280,
                 trt_opt_shape=640,
                 trt_calib_mode=False,
                 cpu_threads=1,
                 enable_mkldnn=False):
        self.pred_config = pred_config
        self.predictor, self.config = load_predictor(
            model_dir,
            run_mode=run_mode,
            batch_size=batch_size,
            min_subgraph_size=self.pred_config.min_subgraph_size,
            use_gpu=use_gpu,
            use_dynamic_shape=use_dynamic_shape,
            trt_min_shape=trt_min_shape,
            trt_max_shape=trt_max_shape,
            trt_opt_shape=trt_opt_shape,
            trt_calib_mode=trt_calib_mode,
            cpu_threads=cpu_threads,
            enable_mkldnn=enable_mkldnn)
        self.det_times = Timer()

    def predict(self, image, threshold=0.5, warmup=0, repeats=1):
        '''
        Args:
            image (str/np.ndarray): path of image/ np.ndarray read by cv2
            threshold (float): threshold of predicted box' score
        Returns:
            results (dict): 'segm': np.ndarray,shape:[N, im_h, im_w]
                            'cate_label': label of segm, shape:[N]
                            'cate_score': confidence score of segm, shape:[N]
        '''
        self.det_times.preprocess_time_s.start()
        inputs = self.preprocess(image)
        self.det_times.preprocess_time_s.end()
        np_label, np_score, np_segms = None, None, None
        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])
        for i in range(warmup):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            np_label = self.predictor.get_output_handle(output_names[
                1]).copy_to_cpu()
            np_score = self.predictor.get_output_handle(output_names[
                2]).copy_to_cpu()
            np_segms = self.predictor.get_output_handle(output_names[
                3]).copy_to_cpu()

        self.det_times.inference_time_s.start()
        for i in range(repeats):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            np_label = self.predictor.get_output_handle(output_names[
                1]).copy_to_cpu()
            np_score = self.predictor.get_output_handle(output_names[
                2]).copy_to_cpu()
            np_segms = self.predictor.get_output_handle(output_names[
                3]).copy_to_cpu()
        self.det_times.inference_time_s.end(repeats=repeats)
        self.det_times.img_num += 1

        return dict(segm=np_segms, label=np_label, score=np_score)


def create_inputs(imgs, im_info):
    """generate input for different model type
    Args:
        im (np.ndarray): image (np.ndarray)
        im_info (dict): info of image
    Returns:
        inputs (dict): input of model
    """
    inputs = {}

    im_shape = []
    scale_factor = []
    for e in im_info:
        im_shape.append(np.array((e['im_shape'], )).astype('float32'))
        scale_factor.append(np.array((e['scale_factor'], )).astype('float32'))

    origin_scale_factor = np.concatenate(scale_factor, axis=0)

    imgs_shape = [[e.shape[1], e.shape[2]] for e in imgs]
    max_shape_h = max([e[0] for e in imgs_shape])
    max_shape_w = max([e[1] for e in imgs_shape])
    padding_imgs = []
    padding_imgs_shape = []
    padding_imgs_scale = []
    for img in imgs:
        im_c, im_h, im_w = img.shape[:]
        padding_im = np.zeros(
            (im_c, max_shape_h, max_shape_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = img
        padding_imgs.append(padding_im)
        padding_imgs_shape.append(
            np.array([max_shape_h, max_shape_w]).astype('float32'))
        rescale = [
            float(max_shape_h) / float(im_h), float(max_shape_w) / float(im_w)
        ]
        padding_imgs_scale.append(np.array(rescale).astype('float32'))
    inputs['image'] = np.stack(padding_imgs, axis=0)
    inputs['im_shape'] = np.stack(padding_imgs_shape, axis=0)
    inputs['scale_factor'] = origin_scale_factor
    return inputs


class PredictConfig():
    """set config of preprocess, postprocess and visualize
    Args:
        model_dir (str): root path of model.yml
    """

    def __init__(self, model_dir):
        # parsing Yaml config for Preprocess
        deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        self.check_model(yml_conf)
        self.arch = yml_conf['arch']
        self.preprocess_infos = yml_conf['Preprocess']
        self.min_subgraph_size = yml_conf['min_subgraph_size']
        self.labels = yml_conf['label_list']
        self.mask = False
        if 'mask' in yml_conf:
            self.mask = yml_conf['mask']
        self.print_config()

    def check_model(self, yml_conf):
        """
        Raises:
            ValueError: loaded model not in supported model type 
        """
        for support_model in SUPPORT_MODELS:
            if support_model in yml_conf['arch']:
                return True
        raise ValueError("Unsupported arch: {}, expect {}".format(yml_conf[
            'arch'], SUPPORT_MODELS))

    def print_config(self):
        print('-----------  Model Configuration -----------')
        print('%s: %s' % ('Model Arch', self.arch))
        print('%s: ' % ('Transform Order'))
        for op_info in self.preprocess_infos:
            print('--%s: %s' % ('transform op', op_info['type']))
        print('--------------------------------------------')


def load_predictor(model_dir,
                   run_mode='fluid',
                   batch_size=1,
                   use_gpu=False,
                   min_subgraph_size=3,
                   use_dynamic_shape=False,
                   trt_min_shape=1,
                   trt_max_shape=1280,
                   trt_opt_shape=640,
                   trt_calib_mode=False,
                   cpu_threads=1,
                   enable_mkldnn=False):
    """set AnalysisConfig, generate AnalysisPredictor
    Args:
        model_dir (str): root path of __model__ and __params__
        use_gpu (bool): whether use gpu
        run_mode (str): mode of running(fluid/trt_fp32/trt_fp16/trt_int8)
        use_dynamic_shape (bool): use dynamic shape or not
        trt_min_shape (int): min shape for dynamic shape in trt
        trt_max_shape (int): max shape for dynamic shape in trt
        trt_opt_shape (int): opt shape for dynamic shape in trt
        trt_calib_mode (bool): If the model is produced by TRT offline quantitative
            calibration, trt_calib_mode need to set True
    Returns:
        predictor (PaddlePredictor): AnalysisPredictor
    Raises:
        ValueError: predict by TensorRT need use_gpu == True.
    """
    if not use_gpu and not run_mode == 'fluid':
        raise ValueError(
            "Predict by TensorRT mode: {}, expect use_gpu==True, but use_gpu == {}"
            .format(run_mode, use_gpu))
    config = Config(
        os.path.join(model_dir, 'model.pdmodel'),
        os.path.join(model_dir, 'model.pdiparams'))
    precision_map = {
        'trt_int8': Config.Precision.Int8,
        'trt_fp32': Config.Precision.Float32,
        'trt_fp16': Config.Precision.Half
    }
    if use_gpu:
        # initial GPU memory(M), device ID
        config.enable_use_gpu(200, 0)
        # optimize graph and fuse op
        config.switch_ir_optim(True)
    else:
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(cpu_threads)
        if enable_mkldnn:
            try:
                # cache 10 different shapes for mkldnn to avoid memory leak
                config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()
            except Exception as e:
                print(
                    "The current environment does not support `mkldnn`, so disable mkldnn."
                )
                pass

    if run_mode in precision_map.keys():
        config.enable_tensorrt_engine(
            workspace_size=1 << 10,
            max_batch_size=batch_size,
            min_subgraph_size=min_subgraph_size,
            precision_mode=precision_map[run_mode],
            use_static=False,
            use_calib_mode=trt_calib_mode)

        if use_dynamic_shape:
            min_input_shape = {'image': [1, 3, trt_min_shape, trt_min_shape]}
            max_input_shape = {'image': [1, 3, trt_max_shape, trt_max_shape]}
            opt_input_shape = {'image': [1, 3, trt_opt_shape, trt_opt_shape]}
            config.set_trt_dynamic_shape_info(min_input_shape, max_input_shape,
                                              opt_input_shape)
            print('trt set dynamic shape done!')

    # disable print log when predict
    config.disable_glog_info()
    # enable shared memory
    config.enable_memory_optim()
    # disable feed, fetch OP, needed by zero_copy_run
    config.switch_use_feed_fetch_ops(False)
    predictor = create_predictor(config)
    return predictor, config


def get_test_images(infer_dir, infer_img):
    """
    Get image path list in TEST mode
    """
    assert infer_img is not None or infer_dir is not None, \
        "--infer_img or --infer_dir should be set"
    assert infer_img is None or os.path.isfile(infer_img), \
            "{} is not a file".format(infer_img)
    assert infer_dir is None or os.path.isdir(infer_dir), \
            "{} is not a directory".format(infer_dir)

    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        return [infer_img]

    images = set()
    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), \
        "infer_dir {} is not a directory".format(infer_dir)
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
    images = list(images)

    assert len(images) > 0, "no image found in {}".format(infer_dir)
    print("Found {} inference images in total.".format(len(images)))

    return images


def visualize(image_list, results, labels, output_dir='output/', threshold=0.5):
    # visualize the predict result
    start_idx = 0
    for idx, image_file in enumerate(image_list):
        im_bboxes_num = results['boxes_num'][idx]
        im_results = {}
        if 'boxes' in results:
            im_results['boxes'] = results['boxes'][start_idx:start_idx +
                                                   im_bboxes_num, :]
        if 'masks' in results:
            im_results['masks'] = results['masks'][start_idx:start_idx +
                                                   im_bboxes_num, :]
        if 'segm' in results:
            im_results['segm'] = results['segm'][start_idx:start_idx +
                                                 im_bboxes_num, :]
        start_idx += im_bboxes_num
        im = visualize_box_mask(
            image_file, im_results, labels, threshold=threshold)
        img_name = os.path.split(image_file)[-1]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        out_path = os.path.join(output_dir, img_name)
        im.save(out_path, quality=95)
        print("save result to: " + out_path)


def print_arguments(args):
    print('-----------  Running Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------')


def predict_image(detector, image_list, batch_size=1):
    batch_loop_cnt = math.ceil(float(len(image_list)) / batch_size)
    for i in range(batch_loop_cnt):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(image_list))
        batch_image_list = image_list[start_index:end_index]
        if FLAGS.run_benchmark:
            detector.predict(
                batch_image_list, FLAGS.threshold, warmup=10, repeats=10)
            cm, gm, gu = get_current_memory_mb()
            detector.cpu_mem += cm
            detector.gpu_mem += gm
            detector.gpu_util += gu
            print('Test iter {}'.format(i))
        else:
            results = detector.predict(batch_image_list, FLAGS.threshold)
            visualize(
                batch_image_list,
                results,
                detector.pred_config.labels,
                output_dir=FLAGS.output_dir,
                threshold=FLAGS.threshold)


def predict_video(detector, camera_id):
    if camera_id != -1:
        capture = cv2.VideoCapture(camera_id)
        video_name = 'output.mp4'
    else:
        capture = cv2.VideoCapture(FLAGS.video_file)
        video_name = os.path.split(FLAGS.video_file)[-1]
    fps = 30
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print('frame_count', frame_count)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # yapf: disable
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # yapf: enable
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
    out_path = os.path.join(FLAGS.output_dir, video_name)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    index = 1
    while (1):
        ret, frame = capture.read()
        if not ret:
            break
        print('detect frame:%d' % (index))
        index += 1
        results = detector.predict([frame], FLAGS.threshold)
        im = visualize_box_mask(
            frame,
            results,
            detector.pred_config.labels,
            threshold=FLAGS.threshold)
        im = np.array(im)
        writer.write(im)
        if camera_id != -1:
            cv2.imshow('Mask Detection', im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    writer.release()


def main():
    pred_config = PredictConfig(FLAGS.model_dir)
    detector = Detector(
        pred_config,
        FLAGS.model_dir,
        use_gpu=FLAGS.use_gpu,
        run_mode=FLAGS.run_mode,
        batch_size=FLAGS.batch_size,
        use_dynamic_shape=FLAGS.use_dynamic_shape,
        trt_min_shape=FLAGS.trt_min_shape,
        trt_max_shape=FLAGS.trt_max_shape,
        trt_opt_shape=FLAGS.trt_opt_shape,
        trt_calib_mode=FLAGS.trt_calib_mode,
        cpu_threads=FLAGS.cpu_threads,
        enable_mkldnn=FLAGS.enable_mkldnn)
    if pred_config.arch == 'SOLOv2':
        detector = DetectorSOLOv2(
            pred_config,
            FLAGS.model_dir,
            use_gpu=FLAGS.use_gpu,
            run_mode=FLAGS.run_mode,
            batch_size=FLAGS.batch_size,
            use_dynamic_shape=FLAGS.use_dynamic_shape,
            trt_min_shape=FLAGS.trt_min_shape,
            trt_max_shape=FLAGS.trt_max_shape,
            trt_opt_shape=FLAGS.trt_opt_shape,
            trt_calib_mode=FLAGS.trt_calib_mode,
            cpu_threads=FLAGS.cpu_threads,
            enable_mkldnn=FLAGS.enable_mkldnn)

    # predict from video file or camera video stream
    if FLAGS.video_file is not None or FLAGS.camera_id != -1:
        predict_video(detector, FLAGS.camera_id)
    else:
        # predict from image
        if FLAGS.image_dir is None and FLAGS.image_file is not None:
            assert FLAGS.batch_size == 1, "batch_size should be 1, when image_file is not None"
        img_list = get_test_images(FLAGS.image_dir, FLAGS.image_file)
        predict_image(detector, img_list, FLAGS.batch_size)
        if not FLAGS.run_benchmark:
            detector.det_times.info(average=True)
        else:
            mems = {
                'cpu_rss_mb': detector.cpu_mem / len(img_list),
                'gpu_rss_mb': detector.gpu_mem / len(img_list),
                'gpu_util': detector.gpu_util * 100 / len(img_list)
            }

            perf_info = detector.det_times.report(average=True)
            model_dir = FLAGS.model_dir
            mode = FLAGS.run_mode
            model_info = {
                'model_name': model_dir.strip('/').split('/')[-1],
                'precision': mode.split('_')[-1]
            }
            data_info = {
                'batch_size': FLAGS.batch_size,
                'shape': "dynamic_shape",
                'data_num': perf_info['img_num']
            }
            det_log = PaddleInferBenchmark(detector.config, model_info,
                                           data_info, perf_info, mems)
            det_log('Det')


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()
    print_arguments(FLAGS)

    main()
