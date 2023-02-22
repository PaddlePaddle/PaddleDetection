from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
# add python path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

# ignore warning log
import warnings
warnings.filterwarnings('ignore')
from ppdet.utils.cli import ArgsParser, merge_args
from ppdet.core.workspace import load_config, merge_config
from ppdet.utils.check import check_gpu, check_npu, check_xpu, check_version, check_config
from ppdet.utils.cam_utils import BBoxCAM
import paddle



def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        "--infer_img",
        type=str,
        default='demo/000000014439.jpg',    # hxw: 404x640
        help="Image path, has higher priority over --infer_dir")
    parser.add_argument("--weights",
                        type=str,
                        default='output/faster_rcnn_r50_vd_fpn_2x_coco_paddlejob/best_model.pdparams'
                        )
    parser.add_argument("--cam_out",
                        type=str,
                        default='cam_faster_rcnn'
                        )
    parser.add_argument("--use_gpu",
                        type=bool,
                        default=True)
    parser.add_argument(
        "--infer_dir",
        type=str,
        default=None,
        help="Directory for images to perform inference on.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory for storing the output visualization files.")
    parser.add_argument(
        "--draw_threshold",
        type=float,
        default=0.8,
        help="Threshold to reserve the result for visualization.")
    parser.add_argument(
        "--save_results",
        type=bool,
        default=False,
        help="Whether to save inference results to output_dir.")
    parser.add_argument(
        "--target_feature_layer_name",
        type=str,
        default='model.backbone', # define the featuremap to show grad cam, such as model.backbone, model.bbox_head.roi_extractor
        help="Whether to save inference results to output_dir.")
    args = parser.parse_args()

    return args

def run(FLAGS, cfg):
    assert cfg.architecture in ['FasterRCNN', 'MaskRCNN', 'YOLOv3', 'PPYOLOE',
                                'PPYOLOEWithAuxHead', 'BlazeFace', 'SSD', 'RetinaNet'],  \
        'Only supported cam for faster_rcnn based and yolov3 based architecture for now,  ' \
        'the others are not supported temporarily!'

    bbox_cam = BBoxCAM(FLAGS, cfg)
    bbox_cam.get_bboxes_cams()

    print('finish')



def main():
    FLAGS = parse_args()
    cfg = load_config(FLAGS.config)
    merge_args(cfg, FLAGS)
    merge_config(FLAGS.opt)

    # disable npu in config by default
    if 'use_npu' not in cfg:
        cfg.use_npu = False

    # disable xpu in config by default
    if 'use_xpu' not in cfg:
        cfg.use_xpu = False

    if cfg.use_gpu:
        place = paddle.set_device('gpu')
    elif cfg.use_npu:
        place = paddle.set_device('npu')
    elif cfg.use_xpu:
        place = paddle.set_device('xpu')
    else:
        place = paddle.set_device('cpu')

    check_config(cfg)
    check_gpu(cfg.use_gpu)
    check_npu(cfg.use_npu)
    check_xpu(cfg.use_xpu)
    check_version()

    run(FLAGS, cfg)


if __name__ == '__main__':
    main()
