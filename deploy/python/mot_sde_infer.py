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
import time
import yaml
import cv2
import numpy as np
from collections import defaultdict

import paddle
from benchmark_utils import PaddleInferBenchmark

from preprocess import preprocess, Resize, NormalizeImage, Permute, PadStride
from utils import argsparser, Timer, get_current_memory_mb
from infer import get_test_images, print_arguments, PredictConfig
from infer import load_predictor, create_inputs
from mot.tracker import JDETracker
from mot.utils import MOTTimer, write_mot_results
from visualize import plot_tracking, plot_tracking_dict


def bench_log(detector, img_list, model_info, batch_size=1, name=None):
    mems = {
        'cpu_rss_mb': detector.cpu_mem / len(img_list),
        'gpu_rss_mb': detector.gpu_mem / len(img_list),
        'gpu_util': detector.gpu_util * 100 / len(img_list)
    }
    perf_info = detector.mot_times.report(average=True) # mot_times
    data_info = {
        'batch_size': batch_size,
        'shape': "dynamic_shape",
        'data_num': perf_info['img_num']
    }
    log = PaddleInferBenchmark(detector.config, model_info, data_info,
                               perf_info, mems)
    log(name)


class MOTDetector(object):
    def __init__(
            self,
            model_dir,
            tracker_config,
            device='CPU',
            run_mode='paddle',
            batch_size=1,
            trt_min_shape=1,
            trt_max_shape=1280,
            trt_opt_shape=640,
            trt_calib_mode=False,
            cpu_threads=1,
            enable_mkldnn=False,
            output_dir='output',
            threshold=0.5, ):
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
        self.cpu_mem, self.gpu_mem, self.gpu_util = 0, 0, 0
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.threshold = threshold

        self.mot_times = Timer(with_tracker=True)
        self.num_classes = len(self.pred_config.labels)

        # tracker config
        cfg = yaml.safe_load(open(tracker_config))['tracker']
        min_box_area = cfg.get('min_box_area', 200)
        vertical_ratio = cfg.get('vertical_ratio', 1.6)
        use_byte = cfg.get('use_byte', True)
        match_thres = cfg.get('match_thres', 0.9)
        conf_thres = cfg.get('conf_thres', 0.6)
        low_conf_thres = cfg.get('low_conf_thres', 0.1)

        self.tracker = JDETracker(
            use_byte = use_byte,
            num_classes=self.num_classes,
            min_box_area=min_box_area,
            vertical_ratio=vertical_ratio,
            match_thres=match_thres,
            conf_thres=conf_thres,
            low_conf_thres=low_conf_thres)

    def set_config(self, model_dir):
        return PredictConfig(model_dir)

    def get_timer(self):
        return self.mot_times

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
        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])

        return inputs

    def postprocess(self, inputs, result):
        # postprocess output of predictor
        np_boxes_num = result['boxes_num']
        if np_boxes_num[0] <= 0:
            print('[WARNNING] No object detected.')
            result = {'boxes': np.zeros([0, 6]), 'boxes_num': [0]}
        result = {k: v for k, v in result.items() if v is not None}
        return result

    def predict(self, repeats=1):
        '''
        Args:
            repeats (int): repeats number for prediction
        Returns:
            result (dict): include 'boxes': np.ndarray: shape:[N,6], N: number of box,
                            matix element:[class, score, x_min, y_min, x_max, y_max]
                            MaskRCNN's result include 'masks': np.ndarray:
                            shape: [N, im_h, im_w]
        '''
        # model prediction
        np_boxes, np_masks = None, None
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
        result = dict(boxes=np_boxes, masks=np_masks, boxes_num=np_boxes_num)

        return result

    def tracking(self, det_results, emb_results=None):
        pred_dets = det_results['boxes']
        pred_embs = emb_results
        pred_dets = np.concatenate((pred_dets[:, 2:], pred_dets[:, 1:2], pred_dets[:, 0:1]), 1)
        # pred_dets should be 'x0, y0, x1, y1, score, cls_id'

        online_targets_dict = self.tracker.update(pred_dets, pred_embs)
        online_tlwhs = defaultdict(list)
        online_scores = defaultdict(list)
        online_ids = defaultdict(list)
        for cls_id in range(self.num_classes):
            online_targets = online_targets_dict[cls_id]
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                tscore = t.score
                if tlwh[2] * tlwh[3] <= self.tracker.min_box_area:
                    continue
                if self.tracker.vertical_ratio > 0 and tlwh[2] / tlwh[
                        3] > self.tracker.vertical_ratio:
                    continue
                online_tlwhs[cls_id].append(tlwh)
                online_ids[cls_id].append(tid)
                online_scores[cls_id].append(tscore)

        return online_tlwhs, online_scores, online_ids

    def predict_image(self,
                      image_list,
                      run_benchmark=False,
                      repeats=1,
                      visual=True):
        num_classes = self.num_classes
        image_list.sort()
        ids2names = self.pred_config.labels
        mot_results = []
        for frame_id, img_file in enumerate(image_list):
            batch_image_list = [img_file]
            print('Test iter {}'.format(frame_id))
            if run_benchmark:
                # preprocess
                inputs = self.preprocess(batch_image_list)  # warmup
                self.mot_times.preprocess_time_s.start()
                inputs = self.preprocess(batch_image_list)
                self.mot_times.preprocess_time_s.end()

                # model prediction
                result = self.predict(repeats=repeats)  # warmup
                self.mot_times.inference_time_s.start()
                result = self.predict(repeats=repeats)
                self.mot_times.inference_time_s.end(repeats=repeats)

                # postprocess
                det_result = self.postprocess(inputs, result)  # warmup
                self.mot_times.postprocess_time_s.start()
                det_result = self.postprocess(inputs, result)
                self.mot_times.postprocess_time_s.end()

                # tracking
                tracking_result = self.tracking(det_result)
                self.mot_times.tracking_time_s.start()
                online_tlwhs, online_scores, online_ids = self.tracking(det_result)
                self.mot_times.tracking_time_s.end()
                self.mot_times.img_num += 1

                cm, gm, gu = get_current_memory_mb()
                self.cpu_mem += cm
                self.gpu_mem += gm
                self.gpu_util += gu
            else:
                self.mot_times.preprocess_time_s.start()
                inputs = self.preprocess(batch_image_list)
                self.mot_times.preprocess_time_s.end()

                self.mot_times.inference_time_s.start()
                result = self.predict()
                self.mot_times.inference_time_s.end()

                self.mot_times.postprocess_time_s.start()
                det_result = self.postprocess(inputs, result)
                self.mot_times.postprocess_time_s.end()

                # tracking process
                self.mot_times.tracking_time_s.start()
                online_tlwhs, online_scores, online_ids = self.tracking(det_result)
                self.mot_times.tracking_time_s.end()
                self.mot_times.img_num += 1

            if visual:
                frame = cv2.imread(img_file)
                im = plot_tracking_dict(
                    frame,
                    num_classes,
                    online_tlwhs,
                    online_ids,
                    online_scores,
                    frame_id=frame_id,
                    ids2names=[])
                seq_name = image_list[0].split('/')[-2]
                save_dir = os.path.join(self.output_dir, seq_name)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                cv2.imwrite(
                    os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), im)


    def predict_video(self, video_file, camera_id):
        video_out_name = 'output.mp4'
        if camera_id != -1:
            capture = cv2.VideoCapture(camera_id)
        else:
            capture = cv2.VideoCapture(video_file)
            video_out_name = os.path.split(video_file)[-1]
        # Get Video info : resolution, fps, frame count
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print("fps: %d, frame_count: %d" % (fps, frame_count))

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        out_path = os.path.join(self.output_dir, video_out_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        frame_id = 1
        timer = MOTTimer()
        results = defaultdict(list)  # support single class and multi classes
        num_classes = self.num_classes
        while (1):
            ret, frame = capture.read()
            if not ret:
                break
            timer.tic()
            print('Tracking frame: %d' % (frame_id))
            frame_id += 1

            batch_image_list = [frame]
            self.mot_times.preprocess_time_s.start()
            inputs = self.preprocess(batch_image_list)
            self.mot_times.preprocess_time_s.end()

            self.mot_times.inference_time_s.start()
            result = self.predict()
            self.mot_times.inference_time_s.end()

            self.mot_times.postprocess_time_s.start()
            det_result = self.postprocess(inputs, result)
            self.mot_times.postprocess_time_s.end()

            # tracking process
            self.mot_times.tracking_time_s.start()
            online_tlwhs, online_scores, online_ids = self.tracking(det_result)
            self.mot_times.tracking_time_s.end()
            self.mot_times.img_num += 1
            timer.toc()

            for cls_id in range(num_classes):
                results[cls_id].append((frame_id + 1, online_tlwhs[cls_id],
                                        online_scores[cls_id], online_ids[cls_id]))

            fps = 1. / timer.duration
            im = plot_tracking_dict(
                frame,
                num_classes,
                online_tlwhs,
                online_ids,
                online_scores,
                frame_id=frame_id,
                fps=fps,
                ids2names=[])

            im = np.array(im)
            writer.write(im)
            if camera_id != -1:
                cv2.imshow('Mask Detection', im)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        writer.release()


def main():
    deploy_file = os.path.join(FLAGS.model_dir, 'infer_cfg.yml')
    with open(deploy_file) as f:
        yml_conf = yaml.safe_load(f)
    arch = yml_conf['arch']
    assert arch in ['YOLO'], 'Only support YOLO series model now.'
    detector_func = 'MOTDetector'
    detector = eval(detector_func)(FLAGS.model_dir,
                                   FLAGS.tracker_config,
                                   device=FLAGS.device,
                                   run_mode=FLAGS.run_mode,
                                   batch_size=FLAGS.batch_size,
                                   trt_min_shape=FLAGS.trt_min_shape,
                                   trt_max_shape=FLAGS.trt_max_shape,
                                   trt_opt_shape=FLAGS.trt_opt_shape,
                                   trt_calib_mode=FLAGS.trt_calib_mode,
                                   cpu_threads=FLAGS.cpu_threads,
                                   enable_mkldnn=FLAGS.enable_mkldnn,
                                   threshold=FLAGS.threshold,
                                   output_dir=FLAGS.output_dir)

    # predict from video file or camera video stream
    if FLAGS.video_file is not None or FLAGS.camera_id != -1:
        detector.predict_video(FLAGS.video_file, FLAGS.camera_id)
    else:
        # predict from image
        if FLAGS.image_dir is None and FLAGS.image_file is not None:
            assert FLAGS.batch_size == 1, "batch_size should be 1, when image_file is not None"
        img_list = get_test_images(FLAGS.image_dir, FLAGS.image_file)
        detector.predict_image(img_list, FLAGS.run_benchmark, repeats=10)

        if not FLAGS.run_benchmark:
            detector.mot_times.info(average=True)
        else:
            mode = FLAGS.run_mode
            det_model_dir = FLAGS.model_dir
            det_model_info = {
                'model_name': det_model_dir.strip('/').split('/')[-1],
                'precision': mode.split('_')[-1]
            }
            bench_log(detector, img_list, det_model_info, name='MOT')


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()
    print_arguments(FLAGS)
    FLAGS.device = FLAGS.device.upper()
    assert FLAGS.device in ['CPU', 'GPU', 'XPU'
                            ], "device should be CPU, GPU or XPU"

    main()
