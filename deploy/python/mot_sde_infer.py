# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import cv2
import numpy as np
import paddle
from benchmark_utils import PaddleInferBenchmark
from preprocess import preprocess
from tracker import DeepSORTTracker
from ppdet.modeling.mot import visualization as mot_vis
from ppdet.modeling.mot.utils import Timer as MOTTimer
from ppdet.modeling.mot.utils import Detection

from paddle.inference import Config
from paddle.inference import create_predictor
from utils import argsparser, Timer, get_current_memory_mb
from infer import get_test_images, print_arguments, PredictConfig, Detector
from mot_jde_infer import write_mot_results
from infer import load_predictor

# Global dictionary
MOT_SUPPORT_MODELS = {'DeepSORT'}


def bench_log(detector, img_list, model_info, batch_size=1, name=None):
    mems = {
        'cpu_rss_mb': detector.cpu_mem / len(img_list),
        'gpu_rss_mb': detector.gpu_mem / len(img_list),
        'gpu_util': detector.gpu_util * 100 / len(img_list)
    }
    perf_info = detector.det_times.report(average=True)
    data_info = {
        'batch_size': batch_size,
        'shape': "dynamic_shape",
        'data_num': perf_info['img_num']
    }
    log = PaddleInferBenchmark(detector.config, model_info, data_info,
                               perf_info, mems)
    log(name)


def scale_coords(coords, input_shape, im_shape, scale_factor):
    im_shape = im_shape[0]
    ratio = scale_factor[0][0]
    pad_w = (input_shape[1] - int(im_shape[1])) / 2
    pad_h = (input_shape[0] - int(im_shape[0])) / 2
    coords[:, 0::2] -= pad_w
    coords[:, 1::2] -= pad_h
    coords[:, 0:4] /= ratio
    coords[:, :4] = np.clip(coords[:, :4], a_min=0, a_max=coords[:, :4].max())
    return coords.round()


def clip_box(xyxy, input_shape, im_shape, scale_factor):
    im_shape = im_shape[0]
    ratio = scale_factor[0][0]
    img0_shape = [int(im_shape[0] / ratio), int(im_shape[1] / ratio)]
    xyxy[:, 0::2] = np.clip(xyxy[:, 0::2], a_min=0, a_max=img0_shape[1])
    xyxy[:, 1::2] = np.clip(xyxy[:, 1::2], a_min=0, a_max=img0_shape[0])
    return xyxy


def preprocess_reid(imgs,
                    w=64,
                    h=192,
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]):
    im_batch = []
    for img in imgs:
        img = cv2.resize(img, (w, h))
        img = img[:, :, ::-1].astype('float32').transpose((2, 0, 1)) / 255
        img_mean = np.array(mean).reshape((3, 1, 1))
        img_std = np.array(std).reshape((3, 1, 1))
        img -= img_mean
        img /= img_std
        img = np.expand_dims(img, axis=0)
        im_batch.append(img)
    im_batch = np.concatenate(im_batch, 0)
    return im_batch


class SDE_Detector(Detector):
    """
    Args:
        pred_config (object): config of model, defined by `Config(model_dir)`
        model_dir (str): root path of model.pdiparams, model.pdmodel and infer_cfg.yml
        device (str): Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU
        run_mode (str): mode of running(fluid/trt_fp32/trt_fp16)
        trt_min_shape (int): min shape for dynamic shape in trt
        trt_max_shape (int): max shape for dynamic shape in trt
        trt_opt_shape (int): opt shape for dynamic shape in trt
        trt_calib_mode (bool): If the model is produced by TRT offline quantitative
            calibration, trt_calib_mode need to set True
        cpu_threads (int): cpu threads
        enable_mkldnn (bool): whether to open MKLDNN
    """

    def __init__(self,
                 pred_config,
                 model_dir,
                 device='CPU',
                 run_mode='fluid',
                 batch_size=1,
                 trt_min_shape=1,
                 trt_max_shape=1088,
                 trt_opt_shape=608,
                 trt_calib_mode=False,
                 cpu_threads=1,
                 enable_mkldnn=False):
        super(SDE_Detector, self).__init__(
            pred_config=pred_config,
            model_dir=model_dir,
            device=device,
            run_mode=run_mode,
            batch_size=batch_size,
            trt_min_shape=trt_min_shape,
            trt_max_shape=trt_max_shape,
            trt_opt_shape=trt_opt_shape,
            trt_calib_mode=trt_calib_mode,
            cpu_threads=cpu_threads,
            enable_mkldnn=enable_mkldnn)
        assert batch_size == 1, "The JDE Detector only supports batch size=1 now"

    def postprocess(self, boxes, input_shape, im_shape, scale_factor,
                    threshold):
        pred_bboxes = scale_coords(boxes[:, 2:], input_shape, im_shape,
                                   scale_factor)
        pred_bboxes = clip_box(pred_bboxes, input_shape, im_shape, scale_factor)
        pred_scores = boxes[:, 1:2]
        keep_mask = pred_scores[:, 0] >= threshold
        return pred_bboxes[keep_mask], pred_scores[keep_mask]

    def predict(self, image, threshold=0.5, warmup=0, repeats=1):
        '''
        Args:
            image (np.ndarray): image numpy data
            threshold (float): threshold of predicted box' score
        Returns:
            pred_bboxes, pred_scores (np.ndarray)
        '''
        self.det_times.preprocess_time_s.start()
        inputs = self.preprocess(image)
        self.det_times.preprocess_time_s.end()

        pred_bboxes, pred_scores = None, None
        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])

        for i in range(warmup):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            boxes_tensor = self.predictor.get_output_handle(output_names[0])
            boxes = boxes_tensor.copy_to_cpu()

        self.det_times.inference_time_s.start()
        for i in range(repeats):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            boxes_tensor = self.predictor.get_output_handle(output_names[0])
            boxes = boxes_tensor.copy_to_cpu()
        self.det_times.inference_time_s.end(repeats=repeats)

        self.det_times.postprocess_time_s.start()
        input_shape = inputs['image'].shape[2:]
        im_shape = inputs['im_shape']
        scale_factor = inputs['scale_factor']
        pred_bboxes, pred_scores = self.postprocess(
            boxes, input_shape, im_shape, scale_factor, threshold)
        self.det_times.postprocess_time_s.end()
        self.det_times.img_num += 1
        return pred_bboxes, pred_scores


class SDE_ReID(object):
    def __init__(self,
                 pred_config,
                 model_dir,
                 device='CPU',
                 run_mode='fluid',
                 batch_size=50,
                 trt_min_shape=1,
                 trt_max_shape=1088,
                 trt_opt_shape=608,
                 trt_calib_mode=False,
                 cpu_threads=1,
                 enable_mkldnn=False):
        self.pred_config = pred_config
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
        assert pred_config.tracker, "Tracking model should have tracker"
        self.tracker = DeepSORTTracker()

    def preprocess(self, crops):
        crops = crops[:self.batch_size]
        inputs = {}
        inputs['crops'] = np.array(crops).astype('float32')
        return inputs

    def postprocess(self, bbox_tlwh, pred_scores, features):
        detections = [
            Detection(tlwh, score, feat)
            for tlwh, score, feat in zip(bbox_tlwh, pred_scores, features)
        ]
        self.tracker.predict()
        online_targets = self.tracker.update(detections)

        online_tlwhs = []
        online_scores = []
        online_ids = []
        for track in online_targets:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            online_tlwhs.append(track.to_tlwh())
            online_scores.append(1.0)
            online_ids.append(track.track_id)
        return online_tlwhs, online_scores, online_ids

    def predict(self, crops, bbox_tlwh, pred_scores, warmup=0, repeats=1):
        self.det_times.preprocess_time_s.start()
        inputs = self.preprocess(crops)
        self.det_times.preprocess_time_s.end()

        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])

        for i in range(warmup):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            feature_tensor = self.predictor.get_output_handle(output_names[0])
            features = feature_tensor.copy_to_cpu()

        self.det_times.inference_time_s.start()
        for i in range(repeats):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            feature_tensor = self.predictor.get_output_handle(output_names[0])
            features = feature_tensor.copy_to_cpu()
        self.det_times.inference_time_s.end(repeats=repeats)

        self.det_times.postprocess_time_s.start()
        online_tlwhs, online_scores, online_ids = self.postprocess(
            bbox_tlwh, pred_scores, features)
        self.det_times.postprocess_time_s.end()
        self.det_times.img_num += 1
        return online_tlwhs, online_scores, online_ids

    def get_crops(self, xyxy, ori_img, pred_scores, w, h):
        self.det_times.preprocess_time_s.start()
        crops = []
        keep_scores = []
        xyxy = xyxy.astype(np.int64)
        ori_img = ori_img.transpose(1, 0, 2)  # [h,w,3]->[w,h,3]
        for i, bbox in enumerate(xyxy):
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            crop = ori_img[bbox[0]:bbox[2], bbox[1]:bbox[3], :]
            crops.append(crop)
            keep_scores.append(pred_scores[i])
        if len(crops) == 0:
            return [], []
        crops = preprocess_reid(crops, w, h)
        self.det_times.preprocess_time_s.end()
        return crops, keep_scores


def predict_image(detector, reid_model, image_list):
    results = []
    image_list.sort()
    for i, img_file in enumerate(image_list):
        frame = cv2.imread(img_file)
        if FLAGS.run_benchmark:
            pred_bboxes, pred_scores = detector.predict(
                [frame], FLAGS.threshold, warmup=10, repeats=10)
            cm, gm, gu = get_current_memory_mb()
            detector.cpu_mem += cm
            detector.gpu_mem += gm
            detector.gpu_util += gu
            print('Test iter {}, file name:{}'.format(i, img_file))
        else:
            pred_bboxes, pred_scores = detector.predict([frame],
                                                        FLAGS.threshold)

        # process
        bbox_tlwh = np.concatenate(
            (pred_bboxes[:, 0:2],
             pred_bboxes[:, 2:4] - pred_bboxes[:, 0:2] + 1),
            axis=1)
        crops, pred_scores = reid_model.get_crops(
            pred_bboxes, frame, pred_scores, w=64, h=192)

        if FLAGS.run_benchmark:
            online_tlwhs, online_scores, online_ids = reid_model.predict(
                crops, bbox_tlwh, pred_scores, warmup=10, repeats=10)
        else:
            online_tlwhs, online_scores, online_ids = reid_model.predict(
                crops, bbox_tlwh, pred_scores)
            online_im = mot_vis.plot_tracking(
                frame, online_tlwhs, online_ids, online_scores, frame_id=i)

            if FLAGS.save_images:
                if not os.path.exists(FLAGS.output_dir):
                    os.makedirs(FLAGS.output_dir)
                img_name = os.path.split(img_file)[-1]
                out_path = os.path.join(FLAGS.output_dir, img_name)
                cv2.imwrite(out_path, online_im)
                print("save result to: " + out_path)


def predict_video(detector, reid_model, camera_id):
    if camera_id != -1:
        capture = cv2.VideoCapture(camera_id)
        video_name = 'mot_output.mp4'
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
    if not FLAGS.save_images:
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    frame_id = 0
    timer = MOTTimer()
    results = []
    while (1):
        ret, frame = capture.read()
        if not ret:
            break
        timer.tic()
        pred_bboxes, pred_scores = detector.predict([frame], FLAGS.threshold)
        timer.toc()
        bbox_tlwh = np.concatenate(
            (pred_bboxes[:, 0:2],
             pred_bboxes[:, 2:4] - pred_bboxes[:, 0:2] + 1),
            axis=1)
        crops, pred_scores = reid_model.get_crops(
            pred_bboxes, frame, pred_scores, w=64, h=192)

        online_tlwhs, online_scores, online_ids = reid_model.predict(
            crops, bbox_tlwh, pred_scores)

        results.append((frame_id + 1, online_tlwhs, online_scores, online_ids))
        fps = 1. / timer.average_time
        im = mot_vis.plot_tracking(
            frame,
            online_tlwhs,
            online_ids,
            online_scores,
            frame_id=frame_id,
            fps=fps)
        if FLAGS.save_images:
            save_dir = os.path.join(FLAGS.output_dir, video_name.split('.')[-2])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite(
                os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), im)
        else:
            writer.write(im)
        frame_id += 1
        print('detect frame:%d' % (frame_id))
        if camera_id != -1:
            cv2.imshow('Tracking Detection', im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    if FLAGS.save_mot_txts:
        result_filename = os.path.join(FLAGS.output_dir,
                                       video_name.split('.')[-2] + '.txt')
        write_mot_results(result_filename, results)

    if FLAGS.save_images:
        save_dir = os.path.join(FLAGS.output_dir, video_name.split('.')[-2])
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg {}'.format(
            save_dir, out_path)
        os.system(cmd_str)
        print('Save video in {}.'.format(out_path))
    else:
        writer.release()


def main():
    pred_config = PredictConfig(FLAGS.model_dir)
    detector = SDE_Detector(
        pred_config,
        FLAGS.model_dir,
        device=FLAGS.device,
        run_mode=FLAGS.run_mode,
        trt_min_shape=FLAGS.trt_min_shape,
        trt_max_shape=FLAGS.trt_max_shape,
        trt_opt_shape=FLAGS.trt_opt_shape,
        trt_calib_mode=FLAGS.trt_calib_mode,
        cpu_threads=FLAGS.cpu_threads,
        enable_mkldnn=FLAGS.enable_mkldnn)

    pred_config = PredictConfig(FLAGS.reid_model_dir)
    reid_model = SDE_ReID(
        pred_config,
        FLAGS.reid_model_dir,
        device=FLAGS.device,
        run_mode=FLAGS.run_mode,
        batch_size=FLAGS.reid_batch_size,
        trt_min_shape=FLAGS.trt_min_shape,
        trt_max_shape=FLAGS.trt_max_shape,
        trt_opt_shape=FLAGS.trt_opt_shape,
        trt_calib_mode=FLAGS.trt_calib_mode,
        cpu_threads=FLAGS.cpu_threads,
        enable_mkldnn=FLAGS.enable_mkldnn)

    # predict from video file or camera video stream
    if FLAGS.video_file is not None or FLAGS.camera_id != -1:
        predict_video(detector, reid_model, FLAGS.camera_id)
    else:
        # predict from image
        img_list = get_test_images(FLAGS.image_dir, FLAGS.image_file)
        predict_image(detector, reid_model, img_list)

        if not FLAGS.run_benchmark:
            detector.det_times.info(average=True)
            reid_model.det_times.info(average=True)
        else:
            mode = FLAGS.run_mode
            det_model_dir = FLAGS.model_dir
            det_model_info = {
                'model_name': det_model_dir.strip('/').split('/')[-1],
                'precision': mode.split('_')[-1]
            }
            bench_log(detector, img_list, det_model_info, name='Det')

            reid_model_dir = FLAGS.reid_model_dir
            reid_model_info = {
                'model_name': reid_model_dir.strip('/').split('/')[-1],
                'precision': mode.split('_')[-1]
            }
            bench_log(reid_model, img_list, reid_model_info, name='ReID')


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()
    print_arguments(FLAGS)
    FLAGS.device = FLAGS.device.upper()
    assert FLAGS.device in ['CPU', 'GPU', 'XPU'
                            ], "device should be CPU, GPU or XPU"

    main()
