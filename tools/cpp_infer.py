import os
import time

import numpy as np
from PIL import Image

import paddle.fluid as fluid

import argparse
from ppdet.utils.visualizer import visualize_results, draw_bbox
from ppdet.utils.eval_utils import eval_results
import ppdet.utils.voc_eval as voc_eval
import ppdet.utils.coco_eval as coco_eval
import cv2
import yaml
import copy

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

eval_clses = {'COCO': coco_eval, 'VOC': voc_eval}

precision_map = {
    'trt_int8': fluid.core.AnalysisConfig.Precision.Int8,
    'trt_fp32': fluid.core.AnalysisConfig.Precision.Float32,
    'trt_fp16': fluid.core.AnalysisConfig.Precision.Half
}


def create_config(model_path, mode='fluid', batch_size=1, min_subgraph_size=3):
    model_file = os.path.join(model_path, '__model__')
    params_file = os.path.join(model_path, '__params__')
    config = fluid.core.AnalysisConfig(model_file, params_file)
    config.enable_use_gpu(100, 0)
    logger.info('min_subgraph_size = %d.' % (min_subgraph_size))

    if mode in precision_map.keys():
        config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=batch_size,
            min_subgraph_size=min_subgraph_size,
            precision_mode=precision_map[mode],
            use_static=False,
            use_calib_mode=mode == 'trt_int8')
        logger.info('Run inference by {}.'.format(mode))
    elif mode == 'fluid':
        logger.info('Run inference by Fluid FP32.')
    else:
        logger.fatal(
            'Wrong mode, only support trt_int8, trt_fp32, trt_fp16, fluid.')
    return config


def offset_to_lengths(lod):
    offset = lod[0]
    lengths = [offset[i + 1] - offset[i] for i in range(len(offset) - 1)]
    return [lengths]


def DecodeImage(im_path):
    with open(im_path, 'rb') as f:
        im = f.read()
    data = np.frombuffer(im, dtype='uint8')
    im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def get_extra_info(im, arch, shape, scale):
    info = []
    input_shape = []
    im_shape = []
    logger.info('The architecture is {}'.format(arch))
    if 'YOLO' in arch:
        im_size = np.array([shape[:2]]).astype('int32')
        logger.info('Extra info: im_size')
        info.append(im_size)
    elif 'SSD' in arch:
        pass
    elif 'RetinaNet' in arch:
        input_shape.extend(im.shape[2:])
        im_info = np.array([input_shape + [scale]]).astype('float32')
        logger.info('Extra info: im_info')
        info.append(im_info)
    elif 'RCNN' in arch:
        input_shape.extend(im.shape[2:])
        im_shape.extend(shape[:2])
        im_info = np.array([input_shape + [scale]]).astype('float32')
        im_shape = np.array([im_shape + [1.]]).astype('float32')
        logger.info('Extra info: im_info, im_shape')
        info.append(im_info)
        info.append(im_shape)
    else:
        logger.error(
            "Unsupported arch: {}, expect YOLO, SSD, RetinaNet and RCNN".format(
                arch))
    return info


class Resize(object):
    def __init__(self, target_size, max_size=0, interp=cv2.INTER_LINEAR):
        super(Resize, self).__init__()
        self.target_size = target_size
        self.max_size = max_size
        self.interp = interp

    def __call__(self, im):
        origin_shape = im.shape[:2]
        im_c = im.shape[2]
        if self.max_size != 0:
            im_size_min = np.min(origin_shape[0:2])
            im_size_max = np.max(origin_shape[0:2])
            im_scale = float(self.target_size) / float(im_size_min)
            if np.round(im_scale * im_size_max) > self.max_size:
                im_scale = float(self.max_size) / float(im_size_max)
            im_scale_x = im_scale
            im_scale_y = im_scale
            resize_w = int(im_scale_x * float(origin_shape[1]))
            resize_h = int(im_scale_y * float(origin_shape[0]))
        else:
            im_scale_x = float(self.target_size) / float(origin_shape[1])
            im_scale_y = float(self.target_size) / float(origin_shape[0])
        im = cv2.resize(
            im,
            None,
            None,
            fx=im_scale_x,
            fy=im_scale_y,
            interpolation=self.interp)
        # padding im
        if self.max_size != 0:
            padding_im = np.zeros(
                (self.max_size, self.max_size, im_c), dtype=np.float32)
            im_h, im_w = im.shape[:2]
            padding_im[:im_h, :im_w, :] = im
            im = padding_im
        return im, im_scale_x


class Normalize(object):
    def __init__(self, mean, std, is_scale=True):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std
        self.is_scale = is_scale

    def __call__(self, im):
        im = im.astype(np.float32, copy=False)
        if self.is_scale:
            im = im / 255.0
        im -= self.mean
        im /= self.std
        return im


class Permute(object):
    def __init__(self, to_bgr=False):
        self.to_bgr = to_bgr

    def __call__(self, im):
        im = im.transpose((2, 0, 1)).copy()
        if self.to_bgr:
            im = im[[2, 1, 0], :, :]
        return im


def Preprocess(img_path, arch, config):
    img = DecodeImage(img_path)
    orig_shape = img.shape
    scale = 1.
    data = []
    data_config = copy.deepcopy(config)
    for data_aug_conf in data_config:
        obj = data_aug_conf.pop('type')
        preprocess = eval(obj)(**data_aug_conf)
        if obj == 'Resize':
            img, scale = preprocess(img)
        else:
            img = preprocess(img)

    img = img[np.newaxis, :]  # N, C, H, W
    data.append(img)
    extra_info = get_extra_info(img, arch, orig_shape, scale)
    data += extra_info
    return data


def infer():
    model_path = FLAGS.model_path
    config_path = FLAGS.config_path
    assert model_path is not None, "Model path: {} does not exist!".format(
        model_path)
    assert config_path is not None, "Config path: {} does not exist!".format(
        config_path)
    with open(config_path) as f:
        conf = yaml.safe_load(f)

    img_data = Preprocess(FLAGS.infer_img, conf['arch'], conf['Preprocess'])

    if conf['use_python_inference']:
        place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        infer_prog, feed_var_names, fetch_targets = fluid.io.load_inference_model(
            dirname=model_path,
            executor=exe,
            model_filename='__model__',
            params_filename='__params__')
        data_dict = {k: v for k, v in zip(feed_var_names, img_data)}
    else:
        inputs = [fluid.core.PaddleTensor(d.copy()) for d in img_data]
        config = create_config(
            model_path,
            mode=conf['mode'],
            min_subgraph_size=conf['min_subgraph_size'])
        predict = fluid.core.create_paddle_predictor(config)

    logger.info('warmup...')
    for i in range(10):
        if conf['use_python_inference']:
            outs = exe.run(infer_prog,
                           feed=data_dict,
                           fetch_list=fetch_targets,
                           return_numpy=False)
        else:
            outs = predict.run(inputs)

    cnt = 100
    logger.info('run benchmark...')
    t1 = time.time()
    for i in range(cnt):
        if conf['use_python_inference']:
            outs = exe.run(infer_prog,
                           feed=data_dict,
                           fetch_list=fetch_targets,
                           return_numpy=False)
        else:
            outs = predict.run(inputs)
    t2 = time.time()

    ms = (t2 - t1) * 1000.0 / float(cnt)

    print("Inference: {} ms per batch image".format(ms))

    if FLAGS.visualize:
        eval_cls = eval_clses[conf['metric']]

        with_background = conf['arch'] != 'YOLO'
        clsid2catid, catid2name = eval_cls.get_category_info(
            None, with_background, True)

        is_bbox_normalized = True if 'SSD' in conf['arch'] else False

        out = outs[-1]
        res = {}
        lod = out.lod() if conf['use_python_inference'] else out.lod
        lengths = offset_to_lengths(lod)
        np_data = np.array(out) if conf[
            'use_python_inference'] else out.as_ndarray()

        res['bbox'] = (np_data, lengths)
        res['im_id'] = np.array([[0]])

        bbox_results = eval_cls.bbox2out([res], clsid2catid, is_bbox_normalized)

        image = Image.open(FLAGS.infer_img).convert('RGB')
        image = draw_bbox(image, 0, catid2name, bbox_results, 0.5)
        image_path = os.path.split(FLAGS.infer_img)[-1]
        if not os.path.exists(FLAGS.output_dir):
            os.makedirs(FLAGS.output_dir)
        out_path = os.path.join(FLAGS.output_dir, image_path)
        image.save(out_path, quality=95)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_path", type=str, default=None, help="model path.")
    parser.add_argument(
        "--config_path", type=str, default=None, help="preprocess config path.")
    parser.add_argument(
        "--infer_img", type=str, default=None, help="Image path")
    parser.add_argument(
        "--visualize",
        action='store_true',
        default=False,
        help="Whether to visualize detection output")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory for storing the output visualization files.")
    FLAGS = parser.parse_args()
    infer()
