#opyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import time
import sys
import cv2
import numpy as np

import paddle
from paddle.inference import Config
from paddle.inference import create_predictor
from ppdet.core.workspace import load_config, create
from ppdet.metrics import COCOMetric

from post_process import PPYOLOEPostProcess


def argsparser():
    """
    argsparser func
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, help="inference model filepath")
    parser.add_argument(
        "--image_file",
        type=str,
        default=None,
        help="image path, if set image_file, it will not eval coco.")
    parser.add_argument(
        "--reader_config",
        type=str,
        default=None,
        help="path of datset and reader config.")
    parser.add_argument(
        "--benchmark",
        type=bool,
        default=False,
        help="Whether run benchmark or not.")
    parser.add_argument(
        "--use_trt",
        type=bool,
        default=False,
        help="Whether use TensorRT or not.")
    parser.add_argument(
        "--precision",
        type=str,
        default="paddle",
        help="mode of running(fp32/fp16/int8)")
    parser.add_argument(
        "--device",
        type=str,
        default="GPU",
        help="Choose the device you want to run, it can be: CPU/GPU/XPU, default is GPU",
    )
    parser.add_argument(
        "--use_dynamic_shape",
        type=bool,
        default=True,
        help="Whether use dynamic shape or not.")
    parser.add_argument(
        "--use_mkldnn",
        type=bool,
        default=False,
        help="Whether use mkldnn or not.")
    parser.add_argument(
        "--cpu_threads", type=int, default=10, help="Num of cpu threads.")
    parser.add_argument("--img_shape", type=int, default=640, help="input_size")
    parser.add_argument(
        '--include_nms',
        type=bool,
        default=True,
        help="Whether include nms or not.")

    return parser


CLASS_LABEL = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]


def generate_scale(im, target_shape, keep_ratio=True):
    """
    Args:
        im (np.ndarray): image (np.ndarray)
    Returns:
        im_scale_x: the resize ratio of X
        im_scale_y: the resize ratio of Y
    """
    origin_shape = im.shape[:2]
    if keep_ratio:
        im_size_min = np.min(origin_shape)
        im_size_max = np.max(origin_shape)
        target_size_min = np.min(target_shape)
        target_size_max = np.max(target_shape)
        im_scale = float(target_size_min) / float(im_size_min)
        if np.round(im_scale * im_size_max) > target_size_max:
            im_scale = float(target_size_max) / float(im_size_max)
        im_scale_x = im_scale
        im_scale_y = im_scale
    else:
        resize_h, resize_w = target_shape
        im_scale_y = resize_h / float(origin_shape[0])
        im_scale_x = resize_w / float(origin_shape[1])
    return im_scale_y, im_scale_x


def image_preprocess(img_path, target_shape):
    """
    image_preprocess func
    """
    img = cv2.imread(img_path)
    im_scale_y, im_scale_x = generate_scale(img, target_shape, keep_ratio=False)
    img = cv2.resize(
        img, (target_shape[0], target_shape[0]),
        interpolation=cv2.INTER_LANCZOS4)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, [2, 0, 1]) / 255
    img = np.expand_dims(img, 0)
    img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std
    scale_factor = np.array([[im_scale_y, im_scale_x]])
    return img.astype(np.float32), scale_factor.astype(np.float32)


def get_color_map_list(num_classes):
    """
    get_color_map_list func
    """
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= ((lab >> 0) & 1) << (7 - j)
            color_map[i * 3 + 1] |= ((lab >> 1) & 1) << (7 - j)
            color_map[i * 3 + 2] |= ((lab >> 2) & 1) << (7 - j)
            j += 1
            lab >>= 3
    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    return color_map


def draw_box(image_file, results, class_label, threshold=0.5):
    """
    draw_box func
    """
    srcimg = cv2.imread(image_file, 1)
    for i in range(len(results)):
        color_list = get_color_map_list(len(class_label))
        clsid2color = {}
        classid, conf = int(results[i, 0]), results[i, 1]
        if conf < threshold:
            continue
        xmin, ymin, xmax, ymax = int(results[i, 2]), int(results[i, 3]), int(
            results[i, 4]), int(results[i, 5])

        if classid not in clsid2color:
            clsid2color[classid] = color_list[classid]
        color = tuple(clsid2color[classid])

        cv2.rectangle(srcimg, (xmin, ymin), (xmax, ymax), color, thickness=2)
        print(class_label[classid] + ": " + str(round(conf, 3)))
        cv2.putText(
            srcimg,
            class_label[classid] + ":" + str(round(conf, 3)),
            (xmin, ymin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            thickness=2, )
    return srcimg


def load_predictor(
        model_dir,
        precision="fp32",
        use_trt=False,
        use_mkldnn=False,
        batch_size=1,
        device="CPU",
        min_subgraph_size=3,
        use_dynamic_shape=False,
        trt_min_shape=1,
        trt_max_shape=1280,
        trt_opt_shape=640,
        cpu_threads=1, ):
    """set AnalysisConfig, generate AnalysisPredictor
    Args:
        model_dir (str): root path of __model__ and __params__
        precision (str): mode of running(fp32/fp16/int8)
        use_trt (bool): whether use TensorRT or not.
        use_mkldnn (bool): whether use MKLDNN or not in CPU.
        device (str): Choose the device you want to run, it can be: CPU/GPU, default is CPU
        use_dynamic_shape (bool): use dynamic shape or not
        trt_min_shape (int): min shape for dynamic shape in trt
        trt_max_shape (int): max shape for dynamic shape in trt
        trt_opt_shape (int): opt shape for dynamic shape in trt
    Returns:
        predictor (PaddlePredictor): AnalysisPredictor
    Raises:
        ValueError: predict by TensorRT need device == 'GPU'.
    """
    rerun_flag = False
    if device != "GPU" and use_trt:
        raise ValueError(
            "Predict by TensorRT mode: {}, expect device=='GPU', but device == {}".
            format(precision, device))
    config = Config(
        os.path.join(model_dir, "model.pdmodel"),
        os.path.join(model_dir, "model.pdiparams"))
    if device == "GPU":
        # initial GPU memory(M), device ID
        config.enable_use_gpu(200, 0)
        # optimize graph and fuse op
        config.switch_ir_optim(True)
    else:
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(cpu_threads)
        config.switch_ir_optim()
        if use_mkldnn:
            config.enable_mkldnn()
            if precision == "int8":
                config.enable_mkldnn_int8(
                    {"conv2d", "depthwise_conv2d", "transpose2", "pool2d"})

    precision_map = {
        "int8": Config.Precision.Int8,
        "fp32": Config.Precision.Float32,
        "fp16": Config.Precision.Half,
    }
    if precision in precision_map.keys() and use_trt:
        config.enable_tensorrt_engine(
            workspace_size=(1 << 25) * batch_size,
            max_batch_size=batch_size,
            min_subgraph_size=min_subgraph_size,
            precision_mode=precision_map[precision],
            use_static=True,
            use_calib_mode=False, )

        if use_dynamic_shape:
            dynamic_shape_file = os.path.join(FLAGS.model_path,
                                              "dynamic_shape.txt")
            if os.path.exists(dynamic_shape_file):
                config.enable_tuned_tensorrt_dynamic_shape(dynamic_shape_file,
                                                           True)
                print("trt set dynamic shape done!")
            else:
                config.collect_shape_range_info(dynamic_shape_file)
                print("Start collect dynamic shape...")
                rerun_flag = True

    # enable shared memory
    config.enable_memory_optim()
    predictor = create_predictor(config)
    return predictor, rerun_flag


def get_current_memory_mb():
    """
    It is used to Obtain the memory usage of the CPU and GPU during the running of the program.
    And this function Current program is time-consuming.
    """
    try:
        pkg.require('pynvml')
    except:
        from pip._internal import main
        main(['install', 'pynvml'])
    try:
        pkg.require('psutil')
    except:
        from pip._internal import main
        main(['install', 'psutil'])
    try:
        pkg.require('GPUtil')
    except:
        from pip._internal import main
        main(['install', 'GPUtil'])
    import pynvml
    import psutil
    import GPUtil

    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))

    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    cpu_mem = info.uss / 1024.0 / 1024.0
    gpu_mem = 0
    gpu_percent = 0
    gpus = GPUtil.getGPUs()
    if gpu_id is not None and len(gpus) > 0:
        gpu_percent = gpus[gpu_id].load
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_mem = meminfo.used / 1024.0 / 1024.0
    return round(cpu_mem, 4), round(gpu_mem, 4)


def predict_image(predictor,
                  image_file,
                  image_shape=[640, 640],
                  warmup=1,
                  repeats=1,
                  threshold=0.5):
    """
    predict image main func
    """
    img, scale_factor = image_preprocess(image_file, image_shape)
    inputs = {}
    inputs["image"] = img
    if FLAGS.include_nms:
        inputs['scale_factor'] = scale_factor
    input_names = predictor.get_input_names()
    for i, _ in enumerate(input_names):
        input_tensor = predictor.get_input_handle(input_names[i])
        input_tensor.copy_from_cpu(inputs[input_names[i]])

    for i in range(warmup):
        predictor.run()

    np_boxes, np_boxes_num = None, None
    cpu_mems, gpu_mems = 0, 0
    predict_time = 0.0
    time_min = float("inf")
    time_max = float("-inf")
    for i in range(repeats):
        start_time = time.time()
        predictor.run()
        output_names = predictor.get_output_names()
        boxes_tensor = predictor.get_output_handle(output_names[0])
        np_boxes = boxes_tensor.copy_to_cpu()
        if FLAGS.include_nms:
            boxes_num = predictor.get_output_handle(output_names[1])
            np_boxes_num = boxes_num.copy_to_cpu()
        end_time = time.time()
        timed = end_time - start_time
        time_min = min(time_min, timed)
        time_max = max(time_max, timed)
        predict_time += timed
        cpu_mem, gpu_mem = get_current_memory_mb()
        cpu_mems += cpu_mem
        gpu_mems += gpu_mem

    time_avg = predict_time / repeats
    print("[Benchmark]Avg cpu_mem:{} MB, avg gpu_mem: {} MB".format(
        cpu_mems / repeats, gpu_mems / repeats))
    print("[Benchmark]Inference time(ms): min={}, max={}, avg={}".format(
        round(time_min * 1000, 2),
        round(time_max * 1000, 1), round(time_avg * 1000, 1)))
    if not FLAGS.include_nms:
        postprocess = PPYOLOEPostProcess(score_threshold=0.3, nms_threshold=0.6)
        res = postprocess(np_boxes, scale_factor)
    else:
        res = {'bbox': np_boxes, 'bbox_num': np_boxes_num}
    res_img = draw_box(
        image_file, res["bbox"], CLASS_LABEL, threshold=threshold)
    cv2.imwrite("result.jpg", res_img)


def eval(predictor, val_loader, metric, rerun_flag=False):
    """
    eval main func
    """
    cpu_mems, gpu_mems = 0, 0
    predict_time = 0.0
    time_min = float("inf")
    time_max = float("-inf")
    sample_nums = len(val_loader)
    input_names = predictor.get_input_names()
    output_names = predictor.get_output_names()
    boxes_tensor = predictor.get_output_handle(output_names[0])
    if FLAGS.include_nms:
        boxes_num = predictor.get_output_handle(output_names[1])
    for batch_id, data in enumerate(val_loader):
        data_all = {k: np.array(v) for k, v in data.items()}
        for i, _ in enumerate(input_names):
            input_tensor = predictor.get_input_handle(input_names[i])
            input_tensor.copy_from_cpu(data_all[input_names[i]])
        start_time = time.time()
        predictor.run()
        np_boxes = boxes_tensor.copy_to_cpu()
        if FLAGS.include_nms:
            np_boxes_num = boxes_num.copy_to_cpu()
        if rerun_flag:
            return
        end_time = time.time()
        timed = end_time - start_time
        time_min = min(time_min, timed)
        time_max = max(time_max, timed)
        predict_time += timed
        cpu_mem, gpu_mem = get_current_memory_mb()
        cpu_mems += cpu_mem
        gpu_mems += gpu_mem
        if not FLAGS.include_nms:
            postprocess = PPYOLOEPostProcess(
                score_threshold=0.3, nms_threshold=0.6)
            res = postprocess(np_boxes, data_all['scale_factor'])
        else:
            res = {'bbox': np_boxes, 'bbox_num': np_boxes_num}
        metric.update(data_all, res)
        if batch_id % 100 == 0:
            print("Eval iter:", batch_id)
            sys.stdout.flush()
    metric.accumulate()
    metric.log()
    map_res = metric.get_results()
    metric.reset()
    time_avg = predict_time / sample_nums
    print("[Benchmark]Avg cpu_mem:{} MB, avg gpu_mem: {} MB".format(
        cpu_mems / sample_nums, gpu_mems / sample_nums))
    print("[Benchmark]Inference time(ms): min={}, max={}, avg={}".format(
        round(time_min * 1000, 2),
        round(time_max * 1000, 1), round(time_avg * 1000, 1)))
    print("[Benchmark] COCO mAP: {}".format(map_res["bbox"][0]))
    sys.stdout.flush()


def main():
    """
    main func
    """
    predictor, rerun_flag = load_predictor(
        FLAGS.model_path,
        device=FLAGS.device,
        use_trt=FLAGS.use_trt,
        use_mkldnn=FLAGS.use_mkldnn,
        precision=FLAGS.precision,
        use_dynamic_shape=FLAGS.use_dynamic_shape,
        cpu_threads=FLAGS.cpu_threads)

    if FLAGS.image_file:
        warmup, repeats = 1, 1
        if FLAGS.benchmark:
            warmup, repeats = 50, 100
        predict_image(
            predictor,
            FLAGS.image_file,
            image_shape=[FLAGS.img_shape, FLAGS.img_shape],
            warmup=warmup,
            repeats=repeats)
    else:
        reader_cfg = load_config(FLAGS.reader_config)

        dataset = reader_cfg["EvalDataset"]
        global val_loader
        val_loader = create("EvalReader")(reader_cfg["EvalDataset"],
                                          reader_cfg["worker_num"],
                                          return_list=True)
        clsid2catid = {v: k for k, v in dataset.catid2clsid.items()}
        anno_file = dataset.get_anno()
        metric = COCOMetric(
            anno_file=anno_file, clsid2catid=clsid2catid, IouType="bbox")
        eval(predictor, val_loader, metric, rerun_flag=rerun_flag)

    if rerun_flag:
        print(
            "***** Collect dynamic shape done, Please rerun the program to get correct results. *****"
        )


if __name__ == "__main__":
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()

    # DataLoader need run on cpu
    paddle.set_device("cpu")

    main()
