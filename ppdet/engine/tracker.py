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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import glob
import paddle
import numpy as np

from ppdet.core.workspace import create
from ppdet.utils.checkpoint import load_weight, load_pretrain_weight
from ppdet.modeling.mot.utils import Detection, get_crops, scale_coords, clip_box
from ppdet.modeling.mot.utils import Timer, load_det_results
from ppdet.modeling.mot import visualization as mot_vis

from ppdet.metrics import Metric, MOTMetric, KITTIMOTMetric
import ppdet.utils.stats as stats

from .callbacks import Callback, ComposeCallback

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = ['Tracker']


class Tracker(object):
    def __init__(self, cfg, mode='eval'):
        self.cfg = cfg
        assert mode.lower() in ['test', 'eval'], \
                "mode should be 'test' or 'eval'"
        self.mode = mode.lower()
        self.optimizer = None

        # build MOT data loader
        self.dataset = cfg['{}MOTDataset'.format(self.mode.capitalize())]

        # build model
        self.model = create(cfg.architecture)

        self.status = {}
        self.start_epoch = 0

        # initial default callbacks
        self._init_callbacks()

        # initial default metrics
        self._init_metrics()
        self._reset_metrics()

    def _init_callbacks(self):
        self._callbacks = []
        self._compose_callback = None

    def _init_metrics(self):
        if self.mode in ['test']:
            self._metrics = []
            return

        if self.cfg.metric == 'MOT':
            self._metrics = [MOTMetric(), ]
        elif self.cfg.metric == 'KITTI':
            self._metrics = [KITTIMOTMetric(), ]
        else:
            logger.warning("Metric not support for metric type {}".format(
                self.cfg.metric))
            self._metrics = []

    def _reset_metrics(self):
        for metric in self._metrics:
            metric.reset()

    def register_callbacks(self, callbacks):
        callbacks = [h for h in list(callbacks) if h is not None]
        for c in callbacks:
            assert isinstance(c, Callback), \
                    "metrics shoule be instances of subclass of Metric"
        self._callbacks.extend(callbacks)
        self._compose_callback = ComposeCallback(self._callbacks)

    def register_metrics(self, metrics):
        metrics = [m for m in list(metrics) if m is not None]
        for m in metrics:
            assert isinstance(m, Metric), \
                    "metrics shoule be instances of subclass of Metric"
        self._metrics.extend(metrics)

    def load_weights_jde(self, weights):
        load_weight(self.model, weights, self.optimizer)

    def load_weights_sde(self, det_weights, reid_weights):
        if self.model.detector:
            load_weight(self.model.detector, det_weights)
            load_weight(self.model.reid, reid_weights)
        else:
            load_weight(self.model.reid, reid_weights, self.optimizer)

    def _eval_seq_jde(self,
                      dataloader,
                      save_dir=None,
                      show_image=False,
                      frame_rate=30,
                      draw_threshold=0):
        if save_dir:
            if not os.path.exists(save_dir): os.makedirs(save_dir)
        tracker = self.model.tracker
        tracker.max_time_lost = int(frame_rate / 30.0 * tracker.track_buffer)

        timer = Timer()
        results = []
        frame_id = 0
        self.status['mode'] = 'track'
        self.model.eval()
        for step_id, data in enumerate(dataloader):
            self.status['step_id'] = step_id
            if frame_id % 40 == 0:
                logger.info('Processing frame {} ({:.2f} fps)'.format(
                    frame_id, 1. / max(1e-5, timer.average_time)))

            # forward
            timer.tic()
            pred_dets, pred_embs = self.model(data)
            online_targets = self.model.tracker.update(pred_dets, pred_embs)

            online_tlwhs, online_ids = [], []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                tscore = t.score
                if tscore < draw_threshold: continue
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > tracker.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(tscore)
            timer.toc()

            # save results
            results.append(
                (frame_id + 1, online_tlwhs, online_scores, online_ids))
            self.save_results(data, frame_id, online_ids, online_tlwhs,
                              online_scores, timer.average_time, show_image,
                              save_dir)
            frame_id += 1

        return results, frame_id, timer.average_time, timer.calls

    def _eval_seq_sde(self,
                      dataloader,
                      save_dir=None,
                      show_image=False,
                      frame_rate=30,
                      det_file='',
                      draw_threshold=0):
        if save_dir:
            if not os.path.exists(save_dir): os.makedirs(save_dir)
        tracker = self.model.tracker
        use_detector = False if not self.model.detector else True

        timer = Timer()
        results = []
        frame_id = 0
        self.status['mode'] = 'track'
        self.model.eval()
        self.model.reid.eval()
        if not use_detector:
            dets_list = load_det_results(det_file, len(dataloader))
            logger.info('Finish loading detection results file {}.'.format(
                det_file))

        for step_id, data in enumerate(dataloader):
            self.status['step_id'] = step_id
            if frame_id % 40 == 0:
                logger.info('Processing frame {} ({:.2f} fps)'.format(
                    frame_id, 1. / max(1e-5, timer.average_time)))

            ori_image = data['ori_image']
            input_shape = data['image'].shape[2:]
            im_shape = data['im_shape']
            scale_factor = data['scale_factor']
            timer.tic()
            if not use_detector:
                dets = dets_list[frame_id]
                bbox_tlwh = paddle.to_tensor(dets['bbox'], dtype='float32')
                pred_scores = paddle.to_tensor(dets['score'], dtype='float32')
                if pred_scores < draw_threshold: continue
                if bbox_tlwh.shape[0] > 0:
                    pred_bboxes = paddle.concat(
                        (bbox_tlwh[:, 0:2],
                         bbox_tlwh[:, 2:4] + bbox_tlwh[:, 0:2]),
                        axis=1)
                else:
                    pred_bboxes = []
                    pred_scores = []
            else:
                outs = self.model.detector(data)
                if outs['bbox_num'] > 0:
                    pred_bboxes = scale_coords(outs['bbox'][:, 2:], input_shape,
                                               im_shape, scale_factor)
                    pred_scores = outs['bbox'][:, 1:2]
                else:
                    pred_bboxes = []
                    pred_scores = []

            pred_bboxes = clip_box(pred_bboxes, input_shape, im_shape,
                                   scale_factor)
            bbox_tlwh = paddle.concat(
                (pred_bboxes[:, 0:2],
                 pred_bboxes[:, 2:4] - pred_bboxes[:, 0:2] + 1),
                axis=1)

            crops, pred_scores = get_crops(
                pred_bboxes, ori_image, pred_scores, w=64, h=192)
            crops = paddle.to_tensor(crops)
            pred_scores = paddle.to_tensor(pred_scores)

            data.update({'crops': crops})
            features = self.model(data)
            features = features.numpy()
            detections = [
                Detection(tlwh, score, feat)
                for tlwh, score, feat in zip(bbox_tlwh, pred_scores, features)
            ]
            self.model.tracker.predict()
            online_targets = self.model.tracker.update(detections)

            online_tlwhs = []
            online_scores = []
            online_ids = []
            for track in online_targets:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                online_tlwhs.append(track.to_tlwh())
                online_scores.append(1.0)
                online_ids.append(track.track_id)
            timer.toc()

            # save results
            results.append(
                (frame_id + 1, online_tlwhs, online_scores, online_ids))
            self.save_results(data, frame_id, online_ids, online_tlwhs,
                              online_scores, timer.average_time, show_image,
                              save_dir)
            frame_id += 1

        return results, frame_id, timer.average_time, timer.calls

    def mot_evaluate(self,
                     data_root,
                     seqs,
                     output_dir,
                     data_type='mot',
                     model_type='JDE',
                     save_images=False,
                     save_videos=False,
                     show_image=False,
                     det_results_dir=''):
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        result_root = os.path.join(output_dir, 'mot_results')
        if not os.path.exists(result_root): os.makedirs(result_root)
        assert data_type in ['mot', 'kitti'], \
            "data_type should be 'mot' or 'kitti'"
        assert model_type in ['JDE', 'DeepSORT', 'FairMOT'], \
            "model_type should be 'JDE', 'DeepSORT' or 'FairMOT'"

        # run tracking

        n_frame = 0
        timer_avgs, timer_calls = [], []
        for seq in seqs:
            if not os.path.isdir(os.path.join(data_root, seq)):
                continue
            infer_dir = os.path.join(data_root, seq, 'img1')
            seqinfo = os.path.join(data_root, seq, 'seqinfo.ini')
            if not os.path.exists(seqinfo) or not os.path.exists(
                    infer_dir) or not os.path.isdir(infer_dir):
                continue

            save_dir = os.path.join(output_dir, 'mot_outputs',
                                    seq) if save_images or save_videos else None
            logger.info('start seq: {}'.format(seq))

            images = self.get_infer_images(infer_dir)
            self.dataset.set_images(images)

            dataloader = create('EvalMOTReader')(self.dataset, 0)

            result_filename = os.path.join(result_root, '{}.txt'.format(seq))
            meta_info = open(seqinfo).read()
            frame_rate = int(meta_info[meta_info.find('frameRate') + 10:
                                       meta_info.find('\nseqLength')])
            with paddle.no_grad():
                if model_type in ['JDE', 'FairMOT']:
                    results, nf, ta, tc = self._eval_seq_jde(
                        dataloader,
                        save_dir=save_dir,
                        show_image=show_image,
                        frame_rate=frame_rate)
                elif model_type in ['DeepSORT']:
                    results, nf, ta, tc = self._eval_seq_sde(
                        dataloader,
                        save_dir=save_dir,
                        show_image=show_image,
                        frame_rate=frame_rate,
                        det_file=os.path.join(det_results_dir,
                                              '{}.txt'.format(seq)))
                else:
                    raise ValueError(model_type)

            self.write_mot_results(result_filename, results, data_type)
            n_frame += nf
            timer_avgs.append(ta)
            timer_calls.append(tc)

            if save_videos:
                output_video_path = os.path.join(save_dir, '..',
                                                 '{}_vis.mp4'.format(seq))
                cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg {}'.format(
                    save_dir, output_video_path)
                os.system(cmd_str)
                logger.info('Save video in {}.'.format(output_video_path))

            logger.info('Evaluate seq: {}'.format(seq))
            # update metrics
            for metric in self._metrics:
                metric.update(data_root, seq, data_type, result_root,
                              result_filename)

        timer_avgs = np.asarray(timer_avgs)
        timer_calls = np.asarray(timer_calls)
        all_time = np.dot(timer_avgs, timer_calls)
        avg_time = all_time / np.sum(timer_calls)
        logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(
            all_time, 1.0 / avg_time))

        # accumulate metric to log out
        for metric in self._metrics:
            metric.accumulate()
            metric.log()
        # reset metric states for metric may performed multiple times
        self._reset_metrics()

    def get_infer_images(self, infer_dir):
        assert infer_dir is None or os.path.isdir(infer_dir), \
            "{} is not a directory".format(infer_dir)
        images = set()
        assert os.path.isdir(infer_dir), \
            "infer_dir {} is not a directory".format(infer_dir)
        exts = ['jpg', 'jpeg', 'png', 'bmp']
        exts += [ext.upper() for ext in exts]
        for ext in exts:
            images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
        images = list(images)
        images.sort()
        assert len(images) > 0, "no image found in {}".format(infer_dir)
        logger.info("Found {} inference images in total.".format(len(images)))
        return images

    def mot_predict(self,
                    video_file,
                    frame_rate,
                    image_dir,
                    output_dir,
                    data_type='mot',
                    model_type='JDE',
                    save_images=False,
                    save_videos=True,
                    show_image=False,
                    det_results_dir='',
                    draw_threshold=0.5):
        assert video_file is not None or image_dir is not None, \
            "--video_file or --image_dir should be set."
        assert video_file is None or os.path.isfile(video_file), \
                "{} is not a file".format(video_file)
        assert image_dir is None or os.path.isdir(image_dir), \
                "{} is not a directory".format(image_dir)

        if not os.path.exists(output_dir): os.makedirs(output_dir)
        result_root = os.path.join(output_dir, 'mot_results')
        if not os.path.exists(result_root): os.makedirs(result_root)
        assert data_type in ['mot', 'kitti'], \
            "data_type should be 'mot' or 'kitti'"
        assert model_type in ['JDE', 'DeepSORT', 'FairMOT'], \
            "model_type should be 'JDE', 'DeepSORT' or 'FairMOT'"

        # run tracking        
        if video_file:
            seq = video_file.split('/')[-1].split('.')[0]
            self.dataset.set_video(video_file, frame_rate)
            logger.info('Starting tracking video {}'.format(video_file))
        elif image_dir:
            seq = image_dir.split('/')[-1].split('.')[0]
            images = [
                '{}/{}'.format(image_dir, x) for x in os.listdir(image_dir)
            ]
            images.sort()
            self.dataset.set_images(images)
            logger.info('Starting tracking folder {}, found {} images'.format(
                image_dir, len(images)))
        else:
            raise ValueError('--video_file or --image_dir should be set.')

        save_dir = os.path.join(output_dir, 'mot_outputs',
                                seq) if save_images or save_videos else None

        dataloader = create('TestMOTReader')(self.dataset, 0)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        if frame_rate == -1:
            frame_rate = self.dataset.frame_rate

        with paddle.no_grad():
            if model_type in ['JDE', 'FairMOT']:
                results, nf, ta, tc = self._eval_seq_jde(
                    dataloader,
                    save_dir=save_dir,
                    show_image=show_image,
                    frame_rate=frame_rate,
                    draw_threshold=draw_threshold)
            elif model_type in ['DeepSORT']:
                results, nf, ta, tc = self._eval_seq_sde(
                    dataloader,
                    save_dir=save_dir,
                    show_image=show_image,
                    frame_rate=frame_rate,
                    det_file=os.path.join(det_results_dir,
                                          '{}.txt'.format(seq)),
                    draw_threshold=draw_threshold)
            else:
                raise ValueError(model_type)

        self.write_mot_results(result_filename, results, data_type)

        if save_videos:
            output_video_path = os.path.join(save_dir, '..',
                                             '{}_vis.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg {}'.format(
                save_dir, output_video_path)
            os.system(cmd_str)
            logger.info('Save video in {}'.format(output_video_path))

    def write_mot_results(self, filename, results, data_type='mot'):
        if data_type in ['mot', 'mcmot', 'lab']:
            save_format = '{frame},{id},{x1},{y1},{w},{h},{score},-1,-1,-1\n'
        elif data_type == 'kitti':
            save_format = '{frame} {id} car 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
        else:
            raise ValueError(data_type)

        with open(filename, 'w') as f:
            for frame_id, tlwhs, tscores, track_ids in results:
                if data_type == 'kitti':
                    frame_id -= 1
                for tlwh, score, track_id in zip(tlwhs, tscores, track_ids):
                    if track_id < 0:
                        continue
                    x1, y1, w, h = tlwh
                    x2, y2 = x1 + w, y1 + h
                    line = save_format.format(
                        frame=frame_id,
                        id=track_id,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        w=w,
                        h=h,
                        score=score)
                    f.write(line)
        logger.info('MOT results save in {}'.format(filename))

    def save_results(self, data, frame_id, online_ids, online_tlwhs,
                     online_scores, average_time, show_image, save_dir):
        if show_image or save_dir is not None:
            assert 'ori_image' in data
            img0 = data['ori_image'].numpy()[0]
            online_im = mot_vis.plot_tracking(
                img0,
                online_tlwhs,
                online_ids,
                online_scores,
                frame_id=frame_id,
                fps=1. / average_time)
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir is not None:
            cv2.imwrite(
                os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)),
                online_im)

    def evalBdd100kWithGt(self,
                          evalPath,
                          result_root,
                          id2orginid={1: 2,
                                      2: 3,
                                      3: 4,
                                      4: 9,
                                      5: 10}):
        import argparse
        from scalabel.eval.mot import acc_single_video_mot, evaluate_track
        from scalabel.label.io import group_and_sort, load
        from bdd100k.label.to_scalabel import bdd100k_to_scalabel
        from scalabel.common.parallel import NPROC, pmap
        from bdd100k.common.utils import load_bdd100k_config
        from scalabel.label.io import parse
        import numpy as np
        import glob
        import json
        from functools import partial
        from scalabel.label.typing import Dataset
        import copy
        import os
        import os.path as osp
        import cv2
        import random
        task = 'box_track'
        iou_thr = 0.5
        ignore_iof_thr = 0.5
        bdd100k_config = load_bdd100k_config(task)
        # category
        id2cls = {
            0: 'pedestrian',
            1: 'rider',
            2: 'car',
            3: 'truck',
            4: 'bus',
            5: 'train',
            6: 'motorcycle',
            7: 'bicycle',
            8: 'other person',
            9: 'trailer',
            10: 'other vehicle'
        }
        result_txts = glob.glob(result_root + '/*.txt')
        infer_result = {}
        for result_txt in result_txts:
            seqName = result_txt.split('/')[-1].replace('.txt', '')
            infer_result[seqName] = np.loadtxt(
                result_txt, delimiter=',', dtype=str).tolist()
        raw_frames = []
        seqList = glob.glob(evalPath + '/*')
        gtMap = {}
        global_id = 0
        for seqItem in seqList:
            seqName = seqItem.split('/')[-1]
            gtMap[seqName] = []
            gtTxtPath = osp.join(seqItem, 'gt', 'gt.txt')
            gtAnnos = np.loadtxt(gtTxtPath, delimiter=',', dtype=float)
            frameIds = list(set(np.squeeze(gtAnnos[..., 0]).tolist()))
            frameIndex = 0
            labels = []
            for frameId in frameIds:
                frameId_anno = gtAnnos[gtAnnos[..., 0] == frameId]
                label_map = {}
                label_map['name'] = seqName + '-%07d' % frameId + '.jpg'
                label_map['labels'] = []
                label_map['videoName'] = seqName
                label_map['frameIndex'] = frameIndex
                frameIndex += 1
                for _frameId, _id, _x, _y, _w, _h, _come_into, _category, _visual in frameId_anno:
                    global_id += 1
                    single_label = {}
                    single_label['id'] = '%8d' % global_id
                    single_label['category'] = id2cls[id2orginid[int(
                        _category)]]
                    single_label['attributes'] = {}
                    single_label['attributes']['occluded'] = False
                    single_label['attributes']['truncated'] = False
                    single_label['attributes']['crowd'] = False
                    single_label['box2d'] = {}
                    single_label['box2d']['x1'] = _x
                    single_label['box2d']['y1'] = _y
                    single_label['box2d']['x2'] = _x + _w
                    single_label['box2d']['y2'] = _y + _h
                    label_map['labels'].append(single_label)
                gtMap[seqName].append(label_map)
        bdd_val_seq = copy.deepcopy(gtMap)
        assert len(bdd_val_seq.keys()) == len(infer_result.keys())
        all_idMap = {}
        global_id = 0
        for seqName in infer_result.keys():
            seqName = seqName.split("/")[-1].replace(".txt", "")
            cur_seq_val = bdd_val_seq[seqName]
            labels = np.array(infer_result[seqName], dtype=float)
            labels[:, 4] = labels[:, 2] + labels[:, 4]
            labels[:, 5] = labels[:, 3] + labels[:, 5]
            frameList = np.array(labels[:, 0], dtype=int)
            frameList = list(set(frameList))
            for frameId in frameList:
                frameMap = {}
                name = seqName + "-" + "%07d" % frameId + ".jpg"
                frameMap["name"] = name
                frameMap["labels"] = []
                myLabels = labels[labels[:, 0] == frameId]
                curId = 0
                for myLabel in myLabels:
                    oriId = str(int(myLabel[1]))
                    if oriId in all_idMap.keys():
                        curId = all_idMap[oriId]
                    else:
                        all_idMap[oriId] = str(global_id)
                        curId = str(global_id)
                        global_id += 1
                    box2dMap = {}
                    box2d = {
                        "x1": myLabel[2],
                        "y1": myLabel[3],
                        "x2": myLabel[4],
                        "y2": myLabel[5],
                    }
                    box2dMap["box2d"] = box2d
                    box2dMap["category"] = "car"
                    box2dMap["id"] = str(curId)
                    frameMap["labels"].append(box2dMap)
                for cur_seq_val_jpg in cur_seq_val:
                    if cur_seq_val_jpg["name"] == name:
                        cur_seq_val_jpg["labels"] = frameMap["labels"]
            raw_frames.extend(cur_seq_val)
        parse_ = partial(parse, validate_frames=True)
        gt_frames = []
        for gt_key in gtMap.keys():
            gt_frames.extend(gtMap[gt_key])
        gt_dataset = Dataset(frames=list(map(parse_, gt_frames)), config=None)
        infer_dataset = Dataset(
            frames=list(map(parse_, raw_frames)), config=None)
        results = evaluate_track(
            acc_single_video_mot,
            gts=group_and_sort(
                bdd100k_to_scalabel(gt_dataset.frames, bdd100k_config)),
            results=group_and_sort(
                bdd100k_to_scalabel(infer_dataset.frames, bdd100k_config)),
            config=bdd100k_config.scalabel,
            iou_thr=iou_thr,
            ignore_iof_thr=ignore_iof_thr,
            nproc=1, )
        return results
