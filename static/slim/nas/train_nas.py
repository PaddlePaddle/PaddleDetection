# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

import time
import numpy as np
import datetime
from collections import deque

from paddle import fluid

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

try:
    from ppdet.experimental import mixed_precision_context
    from ppdet.core.workspace import load_config, merge_config, create, register
    from ppdet.data.reader import create_reader

    from ppdet.utils import dist_utils
    from ppdet.utils.eval_utils import parse_fetches, eval_run
    from ppdet.utils.stats import TrainingStats
    from ppdet.utils.cli import ArgsParser
    from ppdet.utils.check import check_gpu, check_version, check_config, enable_static_mode
    import ppdet.utils.checkpoint as checkpoint
except ImportError as e:
    if sys.argv[0].find('static') >= 0:
        logger.error("Importing ppdet failed when running static model "
                     "with error: {}\n"
                     "please try:\n"
                     "\t1. run static model under PaddleDetection/static "
                     "directory\n"
                     "\t2. run 'pip uninstall ppdet' to uninstall ppdet "
                     "dynamic version firstly.".format(e))
        sys.exit(-1)
    else:
        raise e

from paddleslim.analysis import flops, TableLatencyEvaluator
from paddleslim.nas import SANAS
### register search space to paddleslim
import search_space


@register
class Constraint(object):
    """
    Constraint for nas
    """

    def __init__(self,
                 ctype,
                 max_constraint=None,
                 min_constraint=None,
                 table_file=None):
        super(Constraint, self).__init__()
        self.ctype = ctype
        self.max_constraint = max_constraint
        self.min_constraint = min_constraint
        self.table_file = table_file

    def compute_constraint(self, program):
        if self.ctype == 'flops':
            model_status = flops(program)
        elif self.ctype == 'latency':
            assert os.path.exists(
                self.table_file
            ), "latency constraint must have latency table, please check whether table file exist!"
            model_latency = TableLatencyEvaluator(self.table_file)
            model_status = model_latency.latency(program, only_conv=True)
        else:
            raise NotImplementedError(
                "{} constraint is NOT support!!! Now PaddleSlim support flops constraint and latency constraint".
                format(self.ctype))

        return model_status


def get_bboxes_scores(result):
    bboxes = result['bbox'][0]
    gt_bbox = result['gt_bbox'][0]
    bbox_lengths = result['bbox'][1][0]
    gt_lengths = result['gt_bbox'][1][0]
    bbox_list = []
    gt_box_list = []
    for i in range(len(bbox_lengths)):
        num = bbox_lengths[i]
        for j in range(num):
            dt = bboxes[j]
            clsid, score, xmin, ymin, xmax, ymax = dt.tolist()
            im_shape = result['im_shape'][0][i].tolist()
            im_height, im_width = int(im_shape[0]), int(im_shape[1])
            xmin *= im_width
            ymin *= im_height
            xmax *= im_width
            ymax *= im_height
            bbox_list.append([xmin, ymin, xmax, ymax, score])
    faces_num_gt = 0
    for i in range(len(gt_lengths)):
        num = gt_lengths[i]
        for j in range(num):
            gt = gt_bbox[j]
            xmin, ymin, xmax, ymax = gt.tolist()
            im_shape = result['im_shape'][0][i].tolist()
            im_height, im_width = int(im_shape[0]), int(im_shape[1])
            xmin *= im_width
            ymin *= im_height
            xmax *= im_width
            ymax *= im_height
            gt_box_list.append([xmin, ymin, xmax, ymax])
            faces_num_gt += 1
    return gt_box_list, bbox_list, faces_num_gt


def calculate_ap_py(results):
    def cal_iou(rect1, rect2):
        lt_x = max(rect1[0], rect2[0])
        lt_y = max(rect1[1], rect2[1])
        rb_x = min(rect1[2], rect2[2])
        rb_y = min(rect1[3], rect2[3])
        if (rb_x > lt_x) and (rb_y > lt_y):
            intersection = (rb_x - lt_x) * (rb_y - lt_y)
        else:
            return 0

        area1 = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
        area2 = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])

        intersection = min(intersection, area1, area2)
        union = area1 + area2 - intersection
        return float(intersection) / union

    def is_same_face(face_gt, face_pred):
        iou = cal_iou(face_gt, face_pred)
        return iou >= 0.5

    def eval_single_image(faces_gt, faces_pred):
        pred_is_true = [False] * len(faces_pred)
        gt_been_pred = [False] * len(faces_gt)
        for i in range(len(faces_pred)):
            isface = False
            for j in range(len(faces_gt)):
                if gt_been_pred[j] == 0:
                    isface = is_same_face(faces_gt[j], faces_pred[i])
                    if isface == 1:
                        gt_been_pred[j] = True
                        break
            pred_is_true[i] = isface
        return pred_is_true

    score_res_pair = {}
    faces_num_gt = 0
    for t in results:
        gt_box_list, bbox_list, face_num_gt = get_bboxes_scores(t)
        faces_num_gt += face_num_gt
        pred_is_true = eval_single_image(gt_box_list, bbox_list)

        for i in range(0, len(pred_is_true)):
            now_score = bbox_list[i][-1]
            if now_score in score_res_pair:
                score_res_pair[now_score].append(int(pred_is_true[i]))
            else:
                score_res_pair[now_score] = [int(pred_is_true[i])]
    keys = score_res_pair.keys()
    keys = sorted(keys, reverse=True)
    tp_num = 0
    predict_num = 0
    precision_list = []
    recall_list = []
    for i in range(len(keys)):
        k = keys[i]
        v = score_res_pair[k]
        predict_num += len(v)
        tp_num += sum(v)
        recall = float(tp_num) / faces_num_gt
        precision_list.append(float(tp_num) / predict_num)
        recall_list.append(recall)
    ap = precision_list[0] * recall_list[0]
    for i in range(1, len(precision_list)):
        ap += precision_list[i] * (recall_list[i] - recall_list[i - 1])
    return ap


def main():
    env = os.environ
    FLAGS.dist = 'PADDLE_TRAINER_ID' in env and 'PADDLE_TRAINERS_NUM' in env
    if FLAGS.dist:
        trainer_id = int(env['PADDLE_TRAINER_ID'])
        import random
        local_seed = (99 + trainer_id)
        random.seed(local_seed)
        np.random.seed(local_seed)

    cfg = load_config(FLAGS.config)
    merge_config(FLAGS.opt)
    check_config(cfg)
    # check if set use_gpu=True in paddlepaddle cpu version
    check_gpu(cfg.use_gpu)
    # check if paddlepaddle version is satisfied
    check_version()

    main_arch = cfg.architecture

    if cfg.use_gpu:
        devices_num = fluid.core.get_cuda_device_count()
    else:
        devices_num = int(os.environ.get('CPU_NUM', 1))

    if 'FLAGS_selected_gpus' in env:
        device_id = int(env['FLAGS_selected_gpus'])
    else:
        device_id = 0
    place = fluid.CUDAPlace(device_id) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    lr_builder = create('LearningRate')
    optim_builder = create('OptimizerBuilder')

    # add NAS
    config = ([(cfg.search_space)])
    server_address = (cfg.server_ip, cfg.server_port)
    load_checkpoint = FLAGS.resume_checkpoint if FLAGS.resume_checkpoint else None
    sa_nas = SANAS(
        config,
        server_addr=server_address,
        init_temperature=cfg.init_temperature,
        reduce_rate=cfg.reduce_rate,
        search_steps=cfg.search_steps,
        save_checkpoint=cfg.save_dir,
        load_checkpoint=load_checkpoint,
        is_server=cfg.is_server)
    start_iter = 0
    train_reader = create_reader(cfg.TrainReader, (cfg.max_iters - start_iter) *
                                 devices_num, cfg)
    eval_reader = create_reader(cfg.EvalReader)

    constraint = create('Constraint')
    for step in range(cfg.search_steps):
        logger.info('----->>> search step: {} <<<------'.format(step))
        archs = sa_nas.next_archs()[0]

        # build program
        startup_prog = fluid.Program()
        train_prog = fluid.Program()
        with fluid.program_guard(train_prog, startup_prog):
            with fluid.unique_name.guard():
                model = create(main_arch)
                if FLAGS.fp16:
                    assert (getattr(model.backbone, 'norm_type', None)
                            != 'affine_channel'), \
                        '--fp16 currently does not support affine channel, ' \
                        ' please modify backbone settings to use batch norm'

                with mixed_precision_context(FLAGS.loss_scale,
                                             FLAGS.fp16) as ctx:
                    inputs_def = cfg['TrainReader']['inputs_def']
                    feed_vars, train_loader = model.build_inputs(**inputs_def)
                    train_fetches = archs(feed_vars, 'train', cfg)
                    loss = train_fetches['loss']
                    if FLAGS.fp16:
                        loss *= ctx.get_loss_scale_var()
                    lr = lr_builder()
                    optimizer = optim_builder(lr)
                    optimizer.minimize(loss)
                    if FLAGS.fp16:
                        loss /= ctx.get_loss_scale_var()

        current_constraint = constraint.compute_constraint(train_prog)
        logger.info('current steps: {}, constraint {}'.format(
            step, current_constraint))

        if (constraint.max_constraint != None and
                current_constraint > constraint.max_constraint) or (
                    constraint.min_constraint != None and
                    current_constraint < constraint.min_constraint):
            continue

        # parse train fetches
        train_keys, train_values, _ = parse_fetches(train_fetches)
        train_values.append(lr)

        if FLAGS.eval:
            eval_prog = fluid.Program()
            with fluid.program_guard(eval_prog, startup_prog):
                with fluid.unique_name.guard():
                    model = create(main_arch)
                    inputs_def = cfg['EvalReader']['inputs_def']
                    feed_vars, eval_loader = model.build_inputs(**inputs_def)
                    fetches = archs(feed_vars, 'eval', cfg)
            eval_prog = eval_prog.clone(True)

            # When iterable mode, set set_sample_list_generator(eval_reader, place)
            eval_loader.set_sample_list_generator(eval_reader)
            extra_keys = ['im_id', 'im_shape', 'gt_bbox']
            eval_keys, eval_values, eval_cls = parse_fetches(fetches, eval_prog,
                                                             extra_keys)

        # compile program for multi-devices
        build_strategy = fluid.BuildStrategy()
        build_strategy.fuse_all_optimizer_ops = False
        build_strategy.fuse_elewise_add_act_ops = True

        exec_strategy = fluid.ExecutionStrategy()
        # iteration number when CompiledProgram tries to drop local execution scopes.
        # Set it to be 1 to save memory usages, so that unused variables in
        # local execution scopes can be deleted after each iteration.
        exec_strategy.num_iteration_per_drop_scope = 1
        if FLAGS.dist:
            dist_utils.prepare_for_multi_process(exe, build_strategy,
                                                 startup_prog, train_prog)
            exec_strategy.num_threads = 1

        exe.run(startup_prog)
        compiled_train_prog = fluid.CompiledProgram(
            train_prog).with_data_parallel(
                loss_name=loss.name,
                build_strategy=build_strategy,
                exec_strategy=exec_strategy)
        if FLAGS.eval:
            compiled_eval_prog = fluid.CompiledProgram(eval_prog)
        # When iterable mode, set set_sample_list_generator(train_reader, place)
        train_loader.set_sample_list_generator(train_reader)

        train_stats = TrainingStats(cfg.log_iter, train_keys)
        train_loader.start()
        end_time = time.time()

        cfg_name = os.path.basename(FLAGS.config).split('.')[0]
        save_dir = os.path.join(cfg.save_dir, cfg_name)
        time_stat = deque(maxlen=cfg.log_iter)
        ap = 0
        for it in range(start_iter, cfg.max_iters):
            start_time = end_time
            end_time = time.time()
            time_stat.append(end_time - start_time)
            time_cost = np.mean(time_stat)
            eta_sec = (cfg.max_iters - it) * time_cost
            eta = str(datetime.timedelta(seconds=int(eta_sec)))
            outs = exe.run(compiled_train_prog, fetch_list=train_values)
            stats = {
                k: np.array(v).mean()
                for k, v in zip(train_keys, outs[:-1])
            }

            train_stats.update(stats)
            logs = train_stats.log()
            if it % cfg.log_iter == 0 and (not FLAGS.dist or trainer_id == 0):
                strs = 'iter: {}, lr: {:.6f}, {}, time: {:.3f}, eta: {}'.format(
                    it, np.mean(outs[-1]), logs, time_cost, eta)
                logger.info(strs)

            if (it > 0 and it == cfg.max_iters - 1) and (not FLAGS.dist or
                                                         trainer_id == 0):
                save_name = str(
                    it) if it != cfg.max_iters - 1 else "model_final"
                checkpoint.save(exe, train_prog,
                                os.path.join(save_dir, save_name))
                if FLAGS.eval:
                    # evaluation
                    results = eval_run(exe, compiled_eval_prog, eval_loader,
                                       eval_keys, eval_values, eval_cls)
                    ap = calculate_ap_py(results)

        train_loader.reset()
        eval_loader.reset()
        logger.info('rewards: ap is {}'.format(ap))
        sa_nas.reward(float(ap))
    current_best_tokens = sa_nas.current_info()['best_tokens']
    logger.info("All steps end, the best BlazeFace-NAS structure  is: ")
    sa_nas.tokens2arch(current_best_tokens)


if __name__ == '__main__':
    enable_static_mode()
    parser = ArgsParser()
    parser.add_argument(
        "-r",
        "--resume_checkpoint",
        default=None,
        type=str,
        help="Checkpoint path for resuming training.")
    parser.add_argument(
        "--fp16",
        action='store_true',
        default=False,
        help="Enable mixed precision training.")
    parser.add_argument(
        "--loss_scale",
        default=8.,
        type=float,
        help="Mixed precision training loss scale.")
    parser.add_argument(
        "--eval",
        action='store_true',
        default=True,
        help="Whether to perform evaluation in train")
    FLAGS = parser.parse_args()
    main()
