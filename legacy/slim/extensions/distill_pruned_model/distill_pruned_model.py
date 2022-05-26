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
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 4)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

from collections import OrderedDict
from paddleslim.dist.single_distiller import merge
from paddleslim.prune import Pruner
from paddleslim.analysis import flops

from paddle import fluid

try:
    from ppdet.core.workspace import load_config, merge_config, create
    from ppdet.data.reader import create_reader
    from ppdet.utils.eval_utils import parse_fetches, eval_results, eval_run
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


def split_distill(split_output_names, weight, target_number):
    """
    Add fine grained distillation losses.
    Each loss is composed by distill_reg_loss, distill_cls_loss and
    distill_obj_loss
    """
    student_var = []
    for name in split_output_names:
        student_var.append(fluid.default_main_program().global_block().var(
            name))
    s_x, s_y, s_w, s_h, s_obj, s_cls = [], [], [], [], [], []
    for i in range(target_number):
        s_x.append(student_var[i * 6])
        s_y.append(student_var[i * 6 + 1])
        s_w.append(student_var[i * 6 + 2])
        s_h.append(student_var[i * 6 + 3])
        s_obj.append(student_var[i * 6 + 4])
        s_cls.append(student_var[i * 6 + 5])

    teacher_var = []
    for name in split_output_names:
        teacher_var.append(fluid.default_main_program().global_block().var(
            'teacher_' + name))
    t_x, t_y, t_w, t_h, t_obj, t_cls = [], [], [], [], [], []
    for i in range(target_number):
        t_x.append(teacher_var[i * 6])
        t_y.append(teacher_var[i * 6 + 1])
        t_w.append(teacher_var[i * 6 + 2])
        t_h.append(teacher_var[i * 6 + 3])
        t_obj.append(teacher_var[i * 6 + 4])
        t_cls.append(teacher_var[i * 6 + 5])

    def obj_weighted_reg(sx, sy, sw, sh, tx, ty, tw, th, tobj):
        loss_x = fluid.layers.sigmoid_cross_entropy_with_logits(
            sx, fluid.layers.sigmoid(tx))
        loss_y = fluid.layers.sigmoid_cross_entropy_with_logits(
            sy, fluid.layers.sigmoid(ty))
        loss_w = fluid.layers.abs(sw - tw)
        loss_h = fluid.layers.abs(sh - th)
        loss = fluid.layers.sum([loss_x, loss_y, loss_w, loss_h])
        weighted_loss = fluid.layers.reduce_mean(loss *
                                                 fluid.layers.sigmoid(tobj))
        return weighted_loss

    def obj_weighted_cls(scls, tcls, tobj):
        loss = fluid.layers.sigmoid_cross_entropy_with_logits(
            scls, fluid.layers.sigmoid(tcls))
        weighted_loss = fluid.layers.reduce_mean(
            fluid.layers.elementwise_mul(
                loss, fluid.layers.sigmoid(tobj), axis=0))
        return weighted_loss

    def obj_loss(sobj, tobj):
        obj_mask = fluid.layers.cast(tobj > 0., dtype="float32")
        obj_mask.stop_gradient = True
        loss = fluid.layers.reduce_mean(
            fluid.layers.sigmoid_cross_entropy_with_logits(sobj, obj_mask))
        return loss

    distill_reg_losses = []
    for i in range(target_number):
        distill_reg_losses.append(
            obj_weighted_reg(s_x[i], s_y[i], s_w[i], s_h[i], t_x[i], t_y[i],
                             t_w[i], t_h[i], t_obj[i]))
    distill_reg_loss = fluid.layers.sum(distill_reg_losses)

    distill_cls_losses = []
    for i in range(target_number):
        distill_cls_losses.append(
            obj_weighted_cls(s_cls[i], t_cls[i], t_obj[i]))
    distill_cls_loss = fluid.layers.sum(distill_cls_losses)

    distill_obj_losses = []
    for i in range(target_number):
        distill_obj_losses.append(obj_loss(s_obj[i], t_obj[i]))
    distill_obj_loss = fluid.layers.sum(distill_obj_losses)

    loss = (distill_reg_loss + distill_cls_loss + distill_obj_loss) * weight
    return loss


def main():
    env = os.environ
    cfg = load_config(FLAGS.config)
    merge_config(FLAGS.opt)
    check_config(cfg)
    # check if set use_gpu=True in paddlepaddle cpu version
    check_gpu(cfg.use_gpu)
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

    # build program
    model = create(main_arch)
    inputs_def = cfg['TrainReader']['inputs_def']
    train_feed_vars, train_loader = model.build_inputs(**inputs_def)
    train_fetches = model.train(train_feed_vars)
    loss = train_fetches['loss']

    start_iter = 0
    train_reader = create_reader(cfg.TrainReader, (cfg.max_iters - start_iter) *
                                 devices_num, cfg)
    # When iterable mode, set set_sample_list_generator(train_reader, place)
    train_loader.set_sample_list_generator(train_reader)

    eval_prog = fluid.Program()
    with fluid.program_guard(eval_prog, fluid.default_startup_program()):
        with fluid.unique_name.guard():
            model = create(main_arch)
            inputs_def = cfg['EvalReader']['inputs_def']
            test_feed_vars, eval_loader = model.build_inputs(**inputs_def)
            fetches = model.eval(test_feed_vars)
    eval_prog = eval_prog.clone(True)

    eval_reader = create_reader(cfg.EvalReader)
    # When iterable mode, set set_sample_list_generator(eval_reader, place)
    eval_loader.set_sample_list_generator(eval_reader)

    teacher_cfg = load_config(FLAGS.teacher_config)
    merge_config(FLAGS.opt)
    teacher_arch = teacher_cfg.architecture
    teacher_program = fluid.Program()
    teacher_startup_program = fluid.Program()

    with fluid.program_guard(teacher_program, teacher_startup_program):
        with fluid.unique_name.guard():
            teacher_feed_vars = OrderedDict()
            for name, var in train_feed_vars.items():
                teacher_feed_vars[name] = teacher_program.global_block(
                )._clone_variable(
                    var, force_persistable=False)
            model = create(teacher_arch)
            train_fetches = model.train(teacher_feed_vars)
            teacher_loss = train_fetches['loss']

    exe.run(teacher_startup_program)
    assert FLAGS.teacher_pretrained, "teacher_pretrained should be set"
    checkpoint.load_params(exe, teacher_program, FLAGS.teacher_pretrained)
    teacher_program = teacher_program.clone(for_test=True)

    target_number = len(model.yolo_head.anchor_masks)

    data_name_map = {
        'image': 'image',
        'gt_bbox': 'gt_bbox',
        'gt_class': 'gt_class',
        'gt_score': 'gt_score'
    }
    for i in range(target_number):
        data_name_map['target{}'.format(i)] = 'target{}'.format(i)

    merge(teacher_program, fluid.default_main_program(), data_name_map, place)

    output_names = [
        [
            'strided_slice_0.tmp_0', 'strided_slice_1.tmp_0',
            'strided_slice_2.tmp_0', 'strided_slice_3.tmp_0',
            'strided_slice_4.tmp_0', 'transpose_0.tmp_0'
        ],
        [
            'strided_slice_5.tmp_0', 'strided_slice_6.tmp_0',
            'strided_slice_7.tmp_0', 'strided_slice_8.tmp_0',
            'strided_slice_9.tmp_0', 'transpose_2.tmp_0'
        ],
        [
            'strided_slice_10.tmp_0', 'strided_slice_11.tmp_0',
            'strided_slice_12.tmp_0', 'strided_slice_13.tmp_0',
            'strided_slice_14.tmp_0', 'transpose_4.tmp_0'
        ],
    ]

    yolo_output_names = []
    for i in range(target_number):
        yolo_output_names.extend(output_names[i])

    assert cfg.use_fine_grained_loss, \
        "Only support use_fine_grained_loss=True, Please set it in config file or '-o use_fine_grained_loss=true'"
    distill_loss = split_distill(yolo_output_names, 1000, target_number)
    loss = distill_loss + loss
    lr_builder = create('LearningRate')
    optim_builder = create('OptimizerBuilder')
    lr = lr_builder()
    opt = optim_builder(lr)
    opt.minimize(loss)

    exe.run(fluid.default_startup_program())
    checkpoint.load_params(exe,
                           fluid.default_main_program(), cfg.pretrain_weights)


    assert FLAGS.pruned_params is not None, \
        "FLAGS.pruned_params is empty!!! Please set it by '--pruned_params' option."
    pruned_params = FLAGS.pruned_params.strip().split(",")
    logger.info("pruned params: {}".format(pruned_params))
    pruned_ratios = [float(n) for n in FLAGS.pruned_ratios.strip().split(",")]
    logger.info("pruned ratios: {}".format(pruned_ratios))
    assert len(pruned_params) == len(pruned_ratios), \
        "The length of pruned params and pruned ratios should be equal."
    assert pruned_ratios > [0] * len(pruned_ratios) and pruned_ratios < [1] * len(pruned_ratios), \
        "The elements of pruned ratios should be in range (0, 1)."

    assert FLAGS.prune_criterion in ['l1_norm', 'geometry_median'], \
            "unsupported prune criterion {}".format(FLAGS.prune_criterion)
    pruner = Pruner(criterion=FLAGS.prune_criterion)
    distill_prog = pruner.prune(
        fluid.default_main_program(),
        fluid.global_scope(),
        params=pruned_params,
        ratios=pruned_ratios,
        place=place,
        only_graph=False)[0]

    base_flops = flops(eval_prog)
    eval_prog = pruner.prune(
        eval_prog,
        fluid.global_scope(),
        params=pruned_params,
        ratios=pruned_ratios,
        place=place,
        only_graph=True)[0]
    pruned_flops = flops(eval_prog)
    logger.info("FLOPs -{}; total FLOPs: {}; pruned FLOPs: {}".format(
        float(base_flops - pruned_flops) / base_flops, base_flops,
        pruned_flops))

    build_strategy = fluid.BuildStrategy()
    build_strategy.fuse_all_reduce_ops = False
    build_strategy.fuse_all_optimizer_ops = False
    build_strategy.fuse_elewise_add_act_ops = True
    # only enable sync_bn in multi GPU devices
    sync_bn = getattr(model.backbone, 'norm_type', None) == 'sync_bn'
    build_strategy.sync_batch_norm = sync_bn and devices_num > 1 \
        and cfg.use_gpu

    exec_strategy = fluid.ExecutionStrategy()
    # iteration number when CompiledProgram tries to drop local execution scopes.
    # Set it to be 1 to save memory usages, so that unused variables in
    # local execution scopes can be deleted after each iteration.
    exec_strategy.num_iteration_per_drop_scope = 1

    parallel_main = fluid.CompiledProgram(distill_prog).with_data_parallel(
        loss_name=loss.name,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy)
    compiled_eval_prog = fluid.CompiledProgram(eval_prog)

    # parse eval fetches
    extra_keys = []
    if cfg.metric == 'COCO':
        extra_keys = ['im_info', 'im_id', 'im_shape']
    if cfg.metric == 'VOC':
        extra_keys = ['gt_bbox', 'gt_class', 'is_difficult']
    eval_keys, eval_values, eval_cls = parse_fetches(fetches, eval_prog,
                                                     extra_keys)

    # whether output bbox is normalized in model output layer
    is_bbox_normalized = False
    if hasattr(model, 'is_bbox_normalized') and \
            callable(model.is_bbox_normalized):
        is_bbox_normalized = model.is_bbox_normalized()
    map_type = cfg.map_type if 'map_type' in cfg else '11point'
    best_box_ap_list = [0.0, 0]  #[map, iter]
    cfg_name = os.path.basename(FLAGS.config).split('.')[0]
    save_dir = os.path.join(cfg.save_dir, cfg_name)

    train_loader.start()
    for step_id in range(start_iter, cfg.max_iters):
        teacher_loss_np, distill_loss_np, loss_np, lr_np = exe.run(
            parallel_main,
            fetch_list=[
                'teacher_' + teacher_loss.name, distill_loss.name, loss.name,
                lr.name
            ])
        if step_id % cfg.log_iter == 0:
            logger.info(
                "step {} lr {:.6f}, loss {:.6f}, distill_loss {:.6f}, teacher_loss {:.6f}".
                format(step_id, lr_np[0], loss_np[0], distill_loss_np[0],
                       teacher_loss_np[0]))
        if step_id % cfg.snapshot_iter == 0 and step_id != 0 or step_id == cfg.max_iters - 1:
            save_name = str(
                step_id) if step_id != cfg.max_iters - 1 else "model_final"
            checkpoint.save(exe, distill_prog,
                            os.path.join(save_dir, save_name))
            # eval
            results = eval_run(exe, compiled_eval_prog, eval_loader, eval_keys,
                               eval_values, eval_cls, cfg)
            resolution = None
            box_ap_stats = eval_results(results, cfg.metric, cfg.num_classes,
                                        resolution, is_bbox_normalized,
                                        FLAGS.output_eval, map_type,
                                        cfg['EvalReader']['dataset'])

            if box_ap_stats[0] > best_box_ap_list[0]:
                best_box_ap_list[0] = box_ap_stats[0]
                best_box_ap_list[1] = step_id
                checkpoint.save(exe, distill_prog,
                                os.path.join("./", "best_model"))
            logger.info("Best test box ap: {}, in step: {}".format(
                best_box_ap_list[0], best_box_ap_list[1]))
    train_loader.reset()


if __name__ == '__main__':
    enable_static_mode()
    parser = ArgsParser()
    parser.add_argument(
        "-t",
        "--teacher_config",
        default=None,
        type=str,
        help="Config file of teacher architecture.")
    parser.add_argument(
        "--teacher_pretrained",
        default=None,
        type=str,
        help="Whether to use pretrained model.")
    parser.add_argument(
        "--output_eval",
        default=None,
        type=str,
        help="Evaluation directory, default is current directory.")

    parser.add_argument(
        "-p",
        "--pruned_params",
        default=None,
        type=str,
        help="The parameters to be pruned when calculating sensitivities.")
    parser.add_argument(
        "--pruned_ratios",
        default=None,
        type=str,
        help="The ratios pruned iteratively for each parameter when calculating sensitivities."
    )
    parser.add_argument(
        "--prune_criterion",
        default='l1_norm',
        type=str,
        help="criterion function type for channels sorting in pruning, can be set " \
             "as 'l1_norm' or 'geometry_median' currently, default 'l1_norm'")
    FLAGS = parser.parse_args()
    main()
