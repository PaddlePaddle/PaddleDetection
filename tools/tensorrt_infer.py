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
import pickle as cp

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


def create_tensor(np_data, dtype, debug=False):
    """
    Args:
        np_data (numpy.array): numpy.array data with dtype
        dtype (string): float32, int64 or int32
    """
    dtype_map = {
        'float32': fluid.core.PaddleDType.FLOAT32,
        'int64': fluid.core.PaddleDType.INT64,
        'int32': fluid.core.PaddleDType.INT32
    }
    t = fluid.core.PaddleTensor()
    t.dtype = dtype_map[dtype]
    t.shape = np_data.shape
    t.data = fluid.core.PaddleBuf(np_data.flatten())
    return t


def offset_to_lengths(lod):
    offset = lod[0]
    lengths = [offset[i + 1] - offset[i] for i in range(len(offset) - 1)]
    return [lengths]


def DecodeImage(im_path, to_rgb=True):
    with open(im_path, 'rb') as f:
        im = f.read()
    data = np.frombuffer(im, dtype='uint8')
    im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
    if to_rgb:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def ResizeImage(im, target_shape):
    origin_shape = im.shape[:2]
    im_scale_x = float(target_shape[1]) / float(origin_shape[1])
    im_scale_y = float(target_shape[0]) / float(origin_shape[0])
    im = cv2.resize(
        im, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=2)
    return im, (im_scale_x + im_scale_y) / 2.


def NormalizeImage(im,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225],
                   is_scale=True):
    """Normalize the image.
    Operators:
        1.(optional) Scale the image to [0,1]
        2. Each pixel minus mean and is divided by std
    """
    im = im.astype(np.float32, copy=False)
    mean = np.array(mean)[np.newaxis, np.newaxis, :]
    std = np.array(std)[np.newaxis, np.newaxis, :]

    if is_scale:
        im = im / 255.0
    im -= mean
    im /= std
    return im


def Permute(im):
    im = np.swapaxes(im, 1, 2)
    im = np.swapaxes(im, 1, 0)
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


def Preprocess(img_path, arch):
    data = []
    orig_img = DecodeImage(img_path)
    img, scale = ResizeImage(orig_img, FLAGS.target_shape)
    img = NormalizeImage(img, FLAGS.mean, FLAGS.std)
    img = Permute(img)
    img = img[np.newaxis, :]  # N, C, H, W
    data.append(img)
    extra_info = get_extra_info(img, arch, orig_img.shape, scale)
    data += extra_info
    return data


def benchmark():
    model_path = FLAGS.model_path

    img_data = Preprocess(FLAGS.infer_img, FLAGS.arch)

    if FLAGS.use_python_inference:
        place = fluid.CUDAPlace(0)
        exe = fluid.Executor(place)
        infer_prog, feed_var_names, fetch_targets = fluid.io.load_inference_model(
            dirname=model_path,
            executor=exe,
            model_filename='__model__',
            params_filename='__params__')
        data_dict = {k: v for k, v in zip(feed_var_names, img_data)}
    else:
        inputs = [create_tensor(d, str(d.dtype)) for d in img_data]
        config = create_config(
            model_path,
            mode=FLAGS.mode,
            min_subgraph_size=FLAGS.min_subgraph_size)
        predict = fluid.core.create_paddle_predictor(config)

    logger.info('warmup...')
    for i in range(10):
        if FLAGS.use_python_inference:
            outs = exe.run(infer_prog,
                           feed=[data_dict],
                           fetch_list=fetch_targets,
                           return_numpy=False)
        else:
            outs = predict.run(inputs)

    cnt = 100
    logger.info('run benchmark...')
    #fluid.profiler.start_profiler('GPU')
    t1 = time.time()
    for i in range(cnt):
        if FLAGS.use_python_inference:
            outs = exe.run(infer_prog,
                           feed=[data_dict],
                           fetch_list=fetch_targets,
                           return_numpy=False)
        else:
            outs = predict.run(inputs)
    #print('outs: ', np.array(outs[-1].data.float_data()))
    t2 = time.time()
    #fluid.profiler.stop_profiler('total', 'logs')

    ms = (t2 - t1) * 1000.0 / float(cnt)

    print("Inference: {} ms per batch image".format(ms))

    if FLAGS.visualize:
        eval_cls = eval_clses[FLAGS.metric]

        with_background = FLAGS.arch != 'YOLO'
        clsid2catid, catid2name = eval_cls.get_category_info(
            None, with_background, True)

        is_bbox_normalized = True if 'SSD' in FLAGS.arch else False

        out = outs[-1]
        res = {}
        lod = out.lod() if FLAGS.use_python_inference else out.lod
        lengths = offset_to_lengths(lod)
        np_data = np.array(out) if FLAGS.use_python_inference else np.array(
            out.data.float_data()).reshape(out.shape)

        res['bbox'] = (np_data, lengths)
        res['im_id'] = np.array([[0]])

        bbox_results = eval_cls.bbox2out([res], clsid2catid, is_bbox_normalized)

        image = Image.open(FLAGS.infer_img).convert('RGB')
        image = draw_bbox(image, 0, catid2name, bbox_results, 0.5)
        image_path = os.path.split(FLAGS.infer_img)[-1]
        out_path = os.path.join(FLAGS.output_dir, image_path)
        image.save(out_path, quality=95)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_path", type=str, default=None, help="model path.")
    parser.add_argument(
        "--visualize",
        action='store_true',
        default=False,
        help="Whether to visualize detection output")
    parser.add_argument(
        "--mode",
        type=str,
        default='fluid',
        help="mode can be trt_fp32, trt_int8, fluid.")
    parser.add_argument(
        "--draw_threshold",
        type=float,
        default=0.5,
        help="Threshold to reserve the result for visualization.")
    parser.add_argument(
        "--min_subgraph_size",
        type=int,
        default=3,
        help="min_subgraph_size for TensorRT.")
    parser.add_argument(
        "--arch",
        type=str,
        default='YOLO',
        help="architecture for different preprocessing and input. It can be YOLO, SSD, RCNN, RetinaNet"
    )
    parser.add_argument(
        "--target_shape",
        nargs='+',
        type=int,
        default=[608, 608],
        help="target size for input.")
    parser.add_argument(
        "--mean",
        nargs='+',
        type=float,
        default=[0.485, 0.456, 0.406],
        help="mean for normlized image.")
    parser.add_argument(
        "--std",
        type=float,
        nargs='+',
        default=[0.229, 0.224, 0.225],
        help="std for normlized image.")
    parser.add_argument(
        "--metric",
        type=str,
        default='COCO',
        help="load category info from metric, COCO or VOC")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory for storing the output visualization files.")
    parser.add_argument(
        "--infer_img", type=str, default=None, help="Image path")
    parser.add_argument(
        "--use_python_inference",
        action='store_true',
        default=False,
        help="Whether to python inference")
    FLAGS = parser.parse_args()
    benchmark()
