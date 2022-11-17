import ast
import yaml
import copy
import argparse
from argparse import ArgumentParser, RawDescriptionHelpFormatter


class ArgsParser(ArgumentParser):
    def __init__(self):
        super(ArgsParser, self).__init__(
            formatter_class=RawDescriptionHelpFormatter)
        self.add_argument(
            "-o", "--opt", nargs='*', help="set configuration options")

    def parse_args(self, argv=None):
        args = super(ArgsParser, self).parse_args(argv)
        assert args.config is not None, \
            "Please specify --config=configure_file_path."
        args.opt = self._parse_opt(args.opt)
        return args

    def _parse_opt(self, opts):
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            k, v = s.split('=', 1)
            if '.' not in k:
                config[k] = yaml.load(v, Loader=yaml.Loader)
            else:
                keys = k.split('.')
                if keys[0] not in config:
                    config[keys[0]] = {}
                cur = config[keys[0]]
                for idx, key in enumerate(keys[1:]):
                    if idx == len(keys) - 2:
                        cur[key] = yaml.load(v, Loader=yaml.Loader)
                    else:
                        cur[key] = {}
                        cur = cur[key]
        return config


def argsparser():
    parser = ArgsParser()

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=("Path of configure"),
        required=True)
    parser.add_argument(
        "--image_file", type=str, default=None, help="Path of image file.")
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Dir of image file, `image_file` has a higher priority.")
    parser.add_argument(
        "--video_file",
        type=str,
        default=None,
        help="Path of video file, `video_file` or `camera_id` has a highest priority."
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="Dir of video file, `video_file` has a higher priority.")
    parser.add_argument(
        "--rtsp",
        type=str,
        nargs='+',
        default=None,
        help="list of rtsp inputs, for one or multiple rtsp input.")
    parser.add_argument(
        "--camera_id",
        type=int,
        default=-1,
        help="device id of camera to predict.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory of output visualization files.")
    parser.add_argument(
        "--pushurl",
        type=str,
        default="",
        help="url of output visualization stream.")
    parser.add_argument(
        "--run_mode",
        type=str,
        default='paddle',
        help="mode of running(paddle/trt_fp32/trt_fp16/trt_int8)")
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU."
    )
    parser.add_argument(
        "--enable_mkldnn",
        type=ast.literal_eval,
        default=False,
        help="Whether use mkldnn with CPU.")
    parser.add_argument(
        "--cpu_threads", type=int, default=1, help="Num of threads with CPU.")
    parser.add_argument(
        "--trt_min_shape", type=int, default=1, help="min_shape for TensorRT.")
    parser.add_argument(
        "--trt_max_shape",
        type=int,
        default=1280,
        help="max_shape for TensorRT.")
    parser.add_argument(
        "--trt_opt_shape",
        type=int,
        default=640,
        help="opt_shape for TensorRT.")
    parser.add_argument(
        "--trt_calib_mode",
        type=bool,
        default=False,
        help="If the model is produced by TRT offline quantitative "
        "calibration, trt_calib_mode need to set True.")
    parser.add_argument(
        "--do_entrance_counting",
        action='store_true',
        help="Whether counting the numbers of identifiers entering "
        "or getting out from the entrance. Note that only support single-class MOT."
    )
    parser.add_argument(
        "--do_break_in_counting",
        action='store_true',
        help="Whether counting the numbers of identifiers break in "
        "the area. Note that only support single-class MOT and "
        "the video should be taken by a static camera.")
    parser.add_argument(
        "--illegal_parking_time",
        type=int,
        default=-1,
        help="illegal parking time which units are seconds, default is -1 which means not recognition illegal parking"
    )
    parser.add_argument(
        "--region_type",
        type=str,
        default='horizontal',
        help="Area type for entrance counting or break in counting, 'horizontal' and "
        "'vertical' used when do entrance counting. 'custom' used when do break in counting. "
        "Note that only support single-class MOT, and the video should be taken by a static camera."
    )
    parser.add_argument(
        '--region_polygon',
        nargs='+',
        type=int,
        default=[],
        help="Clockwise point coords (x0,y0,x1,y1...) of polygon of area when "
        "do_break_in_counting. Note that only support single-class MOT and "
        "the video should be taken by a static camera.")
    parser.add_argument(
        "--secs_interval",
        type=int,
        default=2,
        help="The seconds interval to count after tracking")
    parser.add_argument(
        "--draw_center_traj",
        action='store_true',
        help="Whether drawing the trajectory of center")

    return parser


def merge_cfg(args):
    # load config
    with open(args.config) as f:
        pred_config = yaml.safe_load(f)

    def merge(cfg, arg):
        # update cfg from arg directly
        merge_cfg = copy.deepcopy(cfg)
        for k, v in cfg.items():
            if k in arg:
                merge_cfg[k] = arg[k]
            else:
                if isinstance(v, dict):
                    merge_cfg[k] = merge(v, arg)

        return merge_cfg

    def merge_opt(cfg, arg):
        merge_cfg = copy.deepcopy(cfg)
        # merge opt
        if 'opt' in arg.keys() and arg['opt']:
            for name, value in arg['opt'].items(
            ):  # example: {'MOT': {'batch_size': 3}}
                if name not in merge_cfg.keys():
                    print("No", name, "in config file!")
                    continue
                for sub_k, sub_v in value.items():
                    if sub_k not in merge_cfg[name].keys():
                        print("No", sub_k, "in config file of", name, "!")
                        continue
                    merge_cfg[name][sub_k] = sub_v

        return merge_cfg

    args_dict = vars(args)
    pred_config = merge(pred_config, args_dict)
    pred_config = merge_opt(pred_config, args_dict)

    return pred_config


def print_arguments(cfg):
    print('-----------  Running Arguments -----------')
    buffer = yaml.dump(cfg)
    print(buffer)
    print('------------------------------------------')
