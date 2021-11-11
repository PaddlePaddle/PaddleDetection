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
import re
import paddle
import numpy as np
import os.path as osp
from collections import defaultdict

from ppdet.core.workspace import create
from ppdet.utils.checkpoint import load_weight, load_pretrain_weight
from ppdet.modeling.mot.utils import Detection, get_crops, scale_coords, clip_box
from ppdet.modeling.mot.utils import MOTTimer, load_det_results, write_mot_results, save_vis_results
from ppdet.modeling.mot.mtmct_utils import parse_bias, trajectory_fusion, sub_cluster, gen_res, print_mtmct_result

from ppdet.metrics import Metric, MOTMetric, KITTIMOTMetric
from ppdet.metrics import MCMOTMetric
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
        elif self.cfg.metric == 'MCMOT':
            self._metrics = [MCMOTMetric(self.cfg.num_classes), ]
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

        timer = MOTTimer()
        frame_id = 0
        self.status['mode'] = 'track'
        self.model.eval()
        results = defaultdict(list)  # support single class and multi classes

        for step_id, data in enumerate(dataloader):
            self.status['step_id'] = step_id
            if frame_id % 40 == 0:
                logger.info('Processing frame {} ({:.2f} fps)'.format(
                    frame_id, 1. / max(1e-5, timer.average_time)))
            # forward
            timer.tic()
            pred_dets, pred_embs = self.model(data)

            pred_dets, pred_embs = pred_dets.numpy(), pred_embs.numpy()
            online_targets_dict = self.model.tracker.update(pred_dets,
                                                            pred_embs)
            online_tlwhs = defaultdict(list)
            online_scores = defaultdict(list)
            online_ids = defaultdict(list)
            for cls_id in range(self.cfg.num_classes):
                online_targets = online_targets_dict[cls_id]
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    tscore = t.score
                    if tlwh[2] * tlwh[3] <= tracker.min_box_area: continue
                    if tracker.vertical_ratio > 0 and tlwh[2] / tlwh[
                            3] > tracker.vertical_ratio:
                        continue
                    online_tlwhs[cls_id].append(tlwh)
                    online_ids[cls_id].append(tid)
                    online_scores[cls_id].append(tscore)
                # save results
                results[cls_id].append(
                    (frame_id + 1, online_tlwhs[cls_id], online_scores[cls_id],
                     online_ids[cls_id]))

            timer.toc()
            save_vis_results(data, frame_id, online_ids, online_tlwhs,
                             online_scores, timer.average_time, show_image,
                             save_dir, self.cfg.num_classes)
            frame_id += 1

        return results, frame_id, timer.average_time, timer.calls

    def _eval_seq_sde(self,
                      dataloader,
                      save_dir=None,
                      show_image=False,
                      frame_rate=30,
                      seq_name='',
                      scaled=False,
                      det_file='',
                      draw_threshold=0):
        if save_dir:
            if not os.path.exists(save_dir): os.makedirs(save_dir)
        use_detector = False if not self.model.detector else True

        timer = MOTTimer()
        results = defaultdict(list)
        frame_id = 0
        self.status['mode'] = 'track'
        self.model.eval()
        self.model.reid.eval()
        if not use_detector:
            dets_list = load_det_results(det_file, len(dataloader))
            logger.info('Finish loading detection results file {}.'.format(
                det_file))

        ForMTMCT = True if 'cameras_bias' in self.cfg else False
        if ForMTMCT:
            mot_features_dict = {}

        for step_id, data in enumerate(dataloader):
            self.status['step_id'] = step_id
            if frame_id % 40 == 0:
                logger.info('Processing frame {} ({:.2f} fps)'.format(
                    frame_id, 1. / max(1e-5, timer.average_time)))

            ori_image = data['ori_image']
            input_shape = data['image'].shape[2:]
            im_shape = data['im_shape']
            scale_factor = data['scale_factor']

            # forward
            timer.tic()
            if not use_detector:
                dets = dets_list[frame_id]
                bbox_tlwh = paddle.to_tensor(dets['bbox'], dtype='float32')
                if bbox_tlwh.shape[0] > 0:
                    # detector outputs: pred_cls_ids, pred_scores, pred_bboxes
                    pred_cls_ids = paddle.to_tensor(
                        dets['cls_id'], dtype='float32').unsqueeze(1)
                    pred_scores = paddle.to_tensor(
                        dets['score'], dtype='float32').unsqueeze(1)
                    pred_bboxes = paddle.concat(
                        (bbox_tlwh[:, 0:2],
                         bbox_tlwh[:, 2:4] + bbox_tlwh[:, 0:2]),
                        axis=1)
                else:
                    logger.warning(
                        'Frame {} has not object, try to modify score threshold.'.
                        format(frame_id))
                    frame_id += 1
                    continue
            else:
                outs = self.model.detector(data)
                if outs['bbox_num'] > 0:
                    # detector outputs: pred_cls_ids, pred_scores, pred_bboxes
                    pred_cls_ids = outs['bbox'][:, 0:1]
                    pred_scores = outs['bbox'][:, 1:2]
                    if not scaled:
                        # scaled means whether the coords after detector outputs
                        # have been scaled back to the original image, set True 
                        # in general detector, set False in JDE YOLOv3.
                        pred_bboxes = scale_coords(outs['bbox'][:, 2:],
                                                   input_shape, im_shape,
                                                   scale_factor)
                    else:
                        pred_bboxes = outs['bbox'][:, 2:]
                else:
                    logger.warning(
                        'Frame {} has not object, try to modify score threshold.'.
                        format(frame_id))
                    frame_id += 1
                    continue

            pred_xyxys, keep_idx = clip_box(pred_bboxes, input_shape, im_shape,
                                            scale_factor)
            pred_scores = paddle.gather_nd(pred_scores, keep_idx).unsqueeze(1)
            pred_cls_ids = paddle.gather_nd(pred_cls_ids, keep_idx).unsqueeze(1)
            pred_tlwhs = paddle.concat(
                (pred_xyxys[:, 0:2],
                 pred_xyxys[:, 2:4] - pred_xyxys[:, 0:2] + 1),
                axis=1)
            pred_dets = paddle.concat(
                (pred_tlwhs, pred_scores, pred_cls_ids), axis=1)

            tracker = self.model.tracker
            crops = get_crops(
                pred_xyxys,
                ori_image,
                w=tracker.input_size[0],
                h=tracker.input_size[1])
            crops = paddle.to_tensor(crops)

            data.update({'crops': crops})
            pred_embs = self.model(data)
            pred_dets, pred_embs = pred_dets.numpy(), pred_embs.numpy()

            tracker.predict()
            online_targets = tracker.update(pred_dets, pred_embs)

            online_tlwhs, online_scores, online_ids = [], [], []
            if ForMTMCT:
                online_tlbrs, online_feats = [], []

            for t in online_targets:
                if not t.is_confirmed() or t.time_since_update > 1:
                    continue
                tlwh = t.to_tlwh()
                tscore = t.score
                tid = t.track_id
                if tscore < draw_threshold: continue
                if tlwh[2] * tlwh[3] <= tracker.min_box_area: continue
                if tracker.vertical_ratio > 0 and tlwh[2] / tlwh[
                        3] > tracker.vertical_ratio:
                    continue
                online_tlwhs.append(tlwh)
                online_scores.append(tscore)
                online_ids.append(tid)
                if ForMTMCT:
                    online_tlbrs.append(t.to_tlbr())
                    online_feats.append(t.feat)
            timer.toc()

            # save results
            results[0].append(
                (frame_id + 1, online_tlwhs, online_scores, online_ids))
            save_vis_results(data, frame_id, online_ids, online_tlwhs,
                             online_scores, timer.average_time, show_image,
                             save_dir, self.cfg.num_classes)
            if ForMTMCT:
                for _tlbr, _id, _feat in zip(online_tlbrs, online_ids, online_feats):
                    feat_data = {}
                    feat_data['bbox'] = _tlbr
                    feat_data['frame'] = f"{frame_id:06d}"
                    feat_data['id'] = _id
                    feat_data['imgname'] = f'{seq_name}_{_id}_{frame_id}.jpg'
                    feat_data['feat'] = _feat
                    mot_features_dict[feat_data['imgname']] = feat_data
            frame_id += 1

        if ForMTMCT:
            return mot_features_dict, frame_id, timer.average_time, timer.calls
        else:
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
                     scaled=False,
                     det_results_dir=''):
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        result_root = os.path.join(output_dir, 'mot_results')
        if not os.path.exists(result_root): os.makedirs(result_root)
        assert data_type in ['mot', 'mcmot', 'kitti'], \
            "data_type should be 'mot', 'mcmot' or 'kitti'"
        assert model_type in ['JDE', 'DeepSORT', 'FairMOT'], \
            "model_type should be 'JDE', 'DeepSORT' or 'FairMOT'"

        ForMTMCT = True if 'cameras_bias' in self.cfg else False
        if ForMTMCT:
            score_thr = self.cfg.score_thr
            use_ff = self.cfg.use_ff
            use_rerank = self.cfg.use_rerank

            mot_features_list = []
            cid_tid_dict = dict()
            cid_bias = parse_bias(self.cfg.cameras_bias)
            scene_cluster = list(cid_bias.keys())

        # run tracking
        n_frame = 0
        timer_avgs, timer_calls = [], []
        for seq in seqs:
            infer_dir = os.path.join(data_root, seq)
            if not os.path.exists(infer_dir) or not os.path.isdir(infer_dir):
                logger.warning("Seq {} error, {} has no images.".format(
                    seq, infer_dir))
                continue
            if os.path.exists(os.path.join(infer_dir, 'img1')):
                infer_dir = os.path.join(infer_dir, 'img1')

            frame_rate = 30
            seqinfo = os.path.join(data_root, seq, 'seqinfo.ini')
            if os.path.exists(seqinfo):
                meta_info = open(seqinfo).read()
                frame_rate = int(meta_info[meta_info.find('frameRate') + 10:
                                           meta_info.find('\nseqLength')])

            save_dir = os.path.join(output_dir, 'mot_outputs',
                                    seq) if save_images or save_videos else None
            logger.info('start seq: {}'.format(seq))

            self.dataset.set_images(self.get_infer_images(infer_dir))
            dataloader = create('EvalMOTReader')(self.dataset, 0)

            result_filename = os.path.join(result_root, '{}.txt'.format(seq))

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
                        seq_name=seq,
                        scaled=scaled,
                        det_file=os.path.join(det_results_dir,
                                              '{}.txt'.format(seq)))
                else:
                    raise ValueError(model_type)
            
            if not ForMTMCT:
                write_mot_results(result_filename, results, data_type,
                                self.cfg.num_classes)
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
            if ForMTMCT:
                # from mot_feature gen mot_list
                mot_features_list.append(results)
                cid = int(re.sub('[a-z,A-Z]',"",seq))
                tid_data = trajectory_fusion(results, cid, cid_bias)
                
                # single seq process
                for line in tid_data:
                    tracklet = tid_data[line]
                    tid = tracklet['tid']
                    if (cid, tid) not in cid_tid_dict:
                        cid_tid_dict[(cid, tid)] = tracklet
            else:
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

        if ForMTMCT:
            map_tid = sub_cluster(cid_tid_dict, scene_cluster, score_thr, use_ff, use_rerank)
            pred_mtmct_file = osp.join(output_dir, 'mtmct_result.txt')
            gen_res(pred_mtmct_file, scene_cluster, map_tid, mot_features_list)
            data_root_gt = osp.join(str.join('/', data_root.split('/')[0:-1]),'gt','gt.txt')
            print_mtmct_result(data_root_gt, pred_mtmct_file)
        else:
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
                    scaled=False,
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
        assert data_type in ['mot', 'mcmot', 'kitti'], \
            "data_type should be 'mot', 'mcmot' or 'kitti'"
        assert model_type in ['JDE', 'DeepSORT', 'FairMOT'], \
            "model_type should be 'JDE', 'DeepSORT' or 'FairMOT'"

        # run tracking        
        if video_file:
            seq = video_file.split('/')[-1].split('.')[0]
            self.dataset.set_video(video_file, frame_rate)
            logger.info('Starting tracking video {}'.format(video_file))
        elif image_dir:
            seq = image_dir.split('/')[-1].split('.')[0]
            # check if has 'img1' dir, default no
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
                    seq_name=seq,
                    scaled=scaled,
                    det_file=os.path.join(det_results_dir,
                                          '{}.txt'.format(seq)),
                    draw_threshold=draw_threshold)
            else:
                raise ValueError(model_type)

        if save_videos:
            output_video_path = os.path.join(save_dir, '..',
                                             '{}_vis.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg {}'.format(
                save_dir, output_video_path)
            os.system(cmd_str)
            logger.info('Save video in {}'.format(output_video_path))

        ForMTMCT = True if 'cameras_bias' in self.cfg else False
        if not ForMTMCT:
            write_mot_results(result_filename, results, data_type,
                            self.cfg.num_classes)
        else:
            return results

    def mtmct_predict(self,
                      mtmct_dir,
                      frame_rate,
                      output_dir,
                      data_type='mot',
                      model_type='JDE',
                      save_images=False,
                      save_videos=True,
                      show_image=False,
                      scaled=False,
                      det_results_dir='',
                      draw_threshold=0.5):
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        result_root = os.path.join(output_dir, 'mtmct_results')
        if not os.path.exists(result_root): os.makedirs(result_root)

        score_thr = self.cfg.score_thr
        use_ff = self.cfg.use_ff
        use_rerank = self.cfg.use_rerank
        cameras_bias = self.cfg.cameras_bias

        mot_features_list = []
        cid_tid_dict = dict()
        cid_bias = parse_bias(cameras_bias)
        scene_cluster = list(cid_bias.keys())

        seqs = os.listdir(mtmct_dir)
        seqs.sort()
        for seq in seqs:
            fpath = os.path.join(mtmct_dir, seq)
            video_file = fpath if os.path.isfile(fpath) else None
            image_dir = fpath if os.path.isdir(fpath) else None
            if video_file is None and image_dir is None:
                print('{} is not a video or image folder.'.format(fpath))
                continue
            mot_features = self.mot_predict(video_file, frame_rate, image_dir, output_dir,
                                            data_type, model_type, save_images,
                                            save_videos, show_image, scaled,
                                            det_results_dir, draw_threshold)
            # from mot_feature gen mot_list
            mot_features_list.append(mot_features)
            cid = int(re.sub('[a-z,A-Z]', "", seq))
            tid_data = trajectory_fusion(mot_features, cid, cid_bias)
            # single seq process
            for line in tid_data:
                tracklet = tid_data[line]
                tid = tracklet['tid']
                if (cid, tid) not in cid_tid_dict:
                    cid_tid_dict[(cid, tid)] = tracklet

        map_tid = sub_cluster(cid_tid_dict, scene_cluster, score_thr, use_ff, use_rerank)
        pred_mtmct_file = osp.join(output_dir, 'mtmct_result.txt')
        gen_res(pred_mtmct_file, scene_cluster, map_tid, mot_features_list)
