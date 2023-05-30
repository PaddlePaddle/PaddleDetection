import fastdeploy as fd
import cv2
import os


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tinypose_model_dir",
        required=True,
        help="path of paddletinypose model directory")
    parser.add_argument(
        "--det_model_dir", help="path of paddledetection model directory")
    parser.add_argument(
        "--image_file", required=True, help="path of test image file.")
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="type of inference device, support 'cpu' or 'gpu'.")
    parser.add_argument(
        "--use_trt",
        type=ast.literal_eval,
        default=False,
        help="wether to use tensorrt.")
    return parser.parse_args()


def build_picodet_option(args):
    option = fd.RuntimeOption()

    if args.device.lower() == "gpu":
        option.use_gpu()

    if args.use_trt:
        option.use_paddle_infer_backend()
        # If use original Tensorrt, not Paddle-TensorRT,
        # please try `option.use_trt_backend()`
        option.paddle_infer_option.enable_trt = True
        option.paddle_infer_option.collect_trt_shape = True
        option.trt_option.set_shape("image", [1, 3, 320, 320], [1, 3, 320, 320],
                                             [1, 3, 320, 320])
        option.trt_option.set_shape("scale_factor", [1, 2], [1, 2], [1, 2])                                     
    return option


def build_tinypose_option(args):
    option = fd.RuntimeOption()

    if args.device.lower() == "gpu":
        option.use_gpu()

    if args.use_trt:
        option.use_paddle_infer_backend()
        # If use original Tensorrt, not Paddle-TensorRT,
        # please try `option.use_trt_backend()`
        option.paddle_infer_option.enable_trt = True
        option.paddle_infer_option.collect_trt_shape = True
        option.trt_option.set_shape("image", [1, 3, 256, 192], [1, 3, 256, 192],
                                             [1, 3, 256, 192])
    return option


args = parse_arguments()
picodet_model_file = os.path.join(args.det_model_dir, "model.pdmodel")
picodet_params_file = os.path.join(args.det_model_dir, "model.pdiparams")
picodet_config_file = os.path.join(args.det_model_dir, "infer_cfg.yml")

# setup runtime
runtime_option = build_picodet_option(args)
det_model = fd.vision.detection.PicoDet(
    picodet_model_file,
    picodet_params_file,
    picodet_config_file,
    runtime_option=runtime_option)

tinypose_model_file = os.path.join(args.tinypose_model_dir, "model.pdmodel")
tinypose_params_file = os.path.join(args.tinypose_model_dir, "model.pdiparams")
tinypose_config_file = os.path.join(args.tinypose_model_dir, "infer_cfg.yml")
# setup runtime
runtime_option = build_tinypose_option(args)
tinypose_model = fd.vision.keypointdetection.PPTinyPose(
    tinypose_model_file,
    tinypose_params_file,
    tinypose_config_file,
    runtime_option=runtime_option)

# predict
im = cv2.imread(args.image_file)
pipeline = fd.pipeline.PPTinyPose(det_model, tinypose_model)
pipeline.detection_model_score_threshold = 0.5
pipeline_result = pipeline.predict(im)
print("Paddle TinyPose Result:\n", pipeline_result)

# visualize
vis_im = fd.vision.vis_keypoint_detection(
    im, pipeline_result, conf_threshold=0.2)
cv2.imwrite("visualized_result.jpg", vis_im)
print("TinyPose visualized result save in ./visualized_result.jpg")
