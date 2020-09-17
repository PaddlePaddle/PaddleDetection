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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

from scipy.cluster.vq import kmeans
import random
import numpy as np
from tqdm import tqdm
from ppdet.utils.cli import ArgsParser
from ppdet.utils.check import check_gpu, check_version, check_config
from ppdet.core.workspace import load_config, merge_config, create

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


class BaseAnchorCluster(object):
    def __init__(self, n, cache_path, cache, verbose=True):
        """
        Base Anchor Cluster

        Args:
            n (int): number of clusters
            cache_path (str): cache directory path
            cache (bool): whether using cache
            verbose (bool): whether print results
        """
        super(BaseAnchorCluster, self).__init__()
        self.n = n
        self.cache_path = cache_path
        self.cache = cache
        self.verbose = verbose

    def print_result(self, centers):
        raise NotImplementedError('%s.print_result is not available' %
                                  self.__class__.__name__)

    def get_whs(self):
        whs_cache_path = os.path.join(self.cache_path, 'whs.npy')
        shapes_cache_path = os.path.join(self.cache_path, 'shapes.npy')
        if self.cache and os.path.exists(whs_cache_path) and os.path.exists(
                shapes_cache_path):
            self.whs = np.load(whs_cache_path)
            self.shapes = np.load(shapes_cache_path)
            return self.whs, self.shapes
        whs = np.zeros((0, 2))
        shapes = np.zeros((0, 2))
        roidbs = self.dataset.get_roidb()
        for rec in tqdm(roidbs):
            h, w = rec['h'], rec['w']
            bbox = rec['gt_bbox']
            wh = bbox[:, 2:4] - bbox[:, 0:2] + 1
            wh = wh / np.array([[w, h]])
            shape = np.ones_like(wh) * np.array([[w, h]])
            whs = np.vstack((whs, wh))
            shapes = np.vstack((shapes, shape))

        if self.cache:
            os.makedirs(self.cache_path, exist_ok=True)
            np.save(whs_cache_path, whs)
            np.save(shapes_cache_path, shapes)

        self.whs = whs
        self.shapes = shapes
        return self.whs, self.shapes

    def calc_anchors(self):
        raise NotImplementedError('%s.calc_anchors is not available' %
                                  self.__class__.__name__)

    def __call__(self):
        self.get_whs()
        centers = self.calc_anchors()
        if self.verbose:
            self.print_result(centers)
        return centers


class YOLOv2AnchorCluster(BaseAnchorCluster):
    def __init__(self,
                 n,
                 dataset,
                 size,
                 cache_path,
                 cache,
                 iters=1000,
                 verbose=True):
        super(YOLOv2AnchorCluster, self).__init__(
            n, cache_path, cache, verbose=verbose)
        """
        YOLOv2 Anchor Cluster

        Reference:
            https://github.com/AlexeyAB/darknet/blob/master/scripts/gen_anchors.py

        Args:
            n (int): number of clusters
            dataset (DataSet): DataSet instance, VOC or COCO
            size (list): [w, h]
            cache_path (str): cache directory path
            cache (bool): whether using cache
            iters (int): kmeans algorithm iters
            verbose (bool): whether print results
        """
        self.dataset = dataset
        self.size = size
        self.iters = iters

    def print_result(self, centers):
        logger.info('%d anchor cluster result: [w, h]' % self.n)
        for w, h in centers:
            logger.info('[%d, %d]' % (round(w), round(h)))

    def metric(self, whs, centers):
        wh1 = whs[:, None]
        wh2 = centers[None]
        inter = np.minimum(wh1, wh2).prod(2)
        return inter / (wh1.prod(2) + wh2.prod(2) - inter)

    def kmeans_expectation(self, whs, centers, assignments):
        dist = self.metric(whs, centers)
        new_assignments = dist.argmax(1)
        converged = (new_assignments == assignments).all()
        return converged, new_assignments

    def kmeans_maximizations(self, whs, centers, assignments):
        new_centers = np.zeros_like(centers)
        for i in range(centers.shape[0]):
            mask = (assignments == i)
            if mask.sum():
                new_centers[i, :] = whs[mask].mean(0)
        return new_centers

    def calc_anchors(self):
        self.whs = self.whs * np.array([self.size])
        # random select k centers
        whs, n, iters = self.whs, self.n, self.iters
        logger.info('Running kmeans for %d anchors on %d points...' %
                    (n, len(whs)))
        idx = np.random.choice(whs.shape[0], size=n, replace=False)
        centers = whs[idx]
        assignments = np.zeros(whs.shape[0:1]) * -1
        # kmeans
        if n == 1:
            return self.kmeans_maximizations(whs, centers, assignments)

        pbar = tqdm(range(iters), desc='Cluster anchors with k-means algorithm')
        for _ in pbar:
            # E step
            converged, assignments = self.kmeans_expectation(whs, centers,
                                                             assignments)
            if converged:
                break
            # M step
            centers = self.kmeans_maximizations(whs, centers, assignments)
            ious = self.metric(whs, centers)
            pbar.desc = 'avg_iou: %.4f' % (ious.max(1).mean())

        centers = sorted(centers, key=lambda x: x[0] * x[1])
        return centers


class YOLOv5AnchorCluster(BaseAnchorCluster):
    def __init__(self,
                 n,
                 dataset,
                 size,
                 cache_path,
                 cache,
                 iters=300,
                 gen_iters=1000,
                 thresh=0.25,
                 verbose=True):
        super(YOLOv5AnchorCluster, self).__init__(
            n, cache_path, cache, verbose=verbose)
        """
        YOLOv5 Anchor Cluster

        Reference:
            https://github.com/ultralytics/yolov5/blob/master/utils/general.py

        Args:
            n (int): number of clusters
            dataset (DataSet): DataSet instance, VOC or COCO
            size (list): [w, h]
            cache_path (str): cache directory path
            cache (bool): whether using cache
            iters (int): iters of kmeans algorithm
            gen_iters (int): iters of genetic algorithm
            threshold (float): anchor scale threshold
            verbose (bool): whether print results
        """
        self.dataset = dataset
        self.size = size
        self.iters = iters
        self.gen_iters = gen_iters
        self.thresh = thresh

    def print_result(self, centers):
        whs = self.whs
        centers = centers[np.argsort(centers.prod(1))]
        x, best = self.metric(whs, centers)
        bpr, aat = (
            best > self.thresh).mean(), (x > self.thresh).mean() * self.n
        logger.info(
            'thresh=%.2f: %.4f best possible recall, %.2f anchors past thr' %
            (self.thresh, bpr, aat))
        logger.info(
            'n=%g, img_size=%s, metric_all=%.3f/%.3f-mean/best, past_thresh=%.3f-mean: '
            % (self.n, self.size, x.mean(), best.mean(),
               x[x > self.thresh].mean()))
        logger.info('%d anchor cluster result: [w, h]' % self.n)
        for w, h in centers:
            logger.info('[%d, %d]' % (round(w), round(h)))

    def metric(self, whs, centers):
        r = whs[:, None] / centers[None]
        x = np.minimum(r, 1. / r).min(2)
        return x, x.max(1)

    def fitness(self, whs, centers):
        _, best = self.metric(whs, centers)
        return (best * (best > self.thresh)).mean()

    def calc_anchors(self):
        self.whs = self.whs * self.shapes / self.shapes.max(
            1, keepdims=True) * np.array([self.size])
        wh0 = self.whs
        i = (wh0 < 3.0).any(1).sum()
        if i:
            logger.warn('Extremely small objects found. %d of %d'
                        'labels are < 3 pixels in width or height' %
                        (i, len(wh0)))

        wh = wh0[(wh0 >= 2.0).any(1)]
        logger.info('Running kmeans for %g anchors on %g points...' %
                    (self.n, len(wh)))
        s = wh.std(0)
        centers, dist = kmeans(wh / s, self.n, iter=self.iters)
        centers *= s

        f, sh, mp, s = self.fitness(wh, centers), centers.shape, 0.9, 0.1
        pbar = tqdm(
            range(self.gen_iters),
            desc='Evolving anchors with Genetic Algorithm')
        for _ in pbar:
            v = np.ones(sh)
            while (v == 1).all():
                v = ((np.random.random(sh) < mp) * np.random.random() *
                     np.random.randn(*sh) * s + 1).clip(0.3, 3.0)
            new_centers = (centers.copy() * v).clip(min=2.0)
            new_f = self.fitness(wh, new_centers)
            if new_f > f:
                f, centers = new_f, new_centers.copy()
                pbar.desc = 'Evolving anchors with Genetic Algorithm: fitness = %.4f' % f

        return centers


def main():
    parser = ArgsParser()
    parser.add_argument(
        '--n', '-n', default=9, type=int, help='num of clusters')
    parser.add_argument(
        '--iters',
        '-i',
        default=1000,
        type=int,
        help='num of iterations for kmeans')
    parser.add_argument(
        '--gen_iters',
        '-gi',
        default=1000,
        type=int,
        help='num of iterations for genetic algorithm')
    parser.add_argument(
        '--thresh',
        '-t',
        default=0.25,
        type=float,
        help='anchor scale threshold')
    parser.add_argument(
        '--verbose', '-v', default=True, type=bool, help='whether print result')
    parser.add_argument(
        '--size',
        '-s',
        default=None,
        type=str,
        help='image size: w,h, using comma as delimiter')
    parser.add_argument(
        '--method',
        '-m',
        default='v2',
        type=str,
        help='cluster method, [v2, v5] are supported now')
    parser.add_argument(
        '--cache_path', default='cache', type=str, help='cache path')
    parser.add_argument(
        '--cache', action='store_true', help='whether use cache')
    FLAGS = parser.parse_args()

    cfg = load_config(FLAGS.config)
    merge_config(FLAGS.opt)
    check_config(cfg)
    # check if set use_gpu=True in paddlepaddle cpu version
    check_gpu(cfg.use_gpu)
    # check if paddlepaddle version is satisfied
    check_version()

    # get dataset
    dataset = cfg['TrainReader']['dataset']
    if FLAGS.size:
        if ',' in FLAGS.size:
            size = list(map(int, FLAGS.size.split(',')))
            assert len(size) == 2, "the format of size is incorrect"
        else:
            size = int(FLAGS.size)
            size = [size, size]

    elif 'image_shape' in cfg['TrainReader']['inputs_def']:
        size = cfg['TrainReader']['inputs_def']['image_shape'][1:]
    else:
        raise ValueError('size is not specified')

    if FLAGS.method == 'v2':
        cluster = YOLOv2AnchorCluster(FLAGS.n, dataset, size, FLAGS.cache_path,
                                      FLAGS.cache, FLAGS.iters, FLAGS.verbose)
    elif FLAGS.method == 'v5':
        cluster = YOLOv5AnchorCluster(FLAGS.n, dataset, size, FLAGS.cache_path,
                                      FLAGS.cache, FLAGS.iters, FLAGS.gen_iters,
                                      FLAGS.thresh, FLAGS.verbose)
    else:
        raise ValueError('cluster method: %s is not supported' % FLAGS.method)

    anchors = cluster()


if __name__ == "__main__":
    main()
