import fastdeploy as fd
import cv2
import os


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        required=True,
        help="path of PP-TinyPose model directory")
    parser.add_argument(
        "--image_file", required=True, help="path of test image file.")
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="type of inference device, support 'cpu', or 'gpu'.")
    parser.add_argument(
        "--use_trt",
        type=ast.literal_eval,
        default=False,
        help="wether to use tensorrt.")
    return parser.parse_args()


def build_option(args):
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

tinypose_model_file = os.path.join(args.model_dir, "model.pdmodel")
tinypose_params_file = os.path.join(args.model_dir, "model.pdiparams")
tinypose_config_file = os.path.join(args.model_dir, "infer_cfg.yml")
# setup runtime 
runtime_option = build_option(args)
tinypose_model = fd.vision.keypointdetection.PPTinyPose(
    tinypose_model_file,
    tinypose_params_file,
    tinypose_config_file,
    runtime_option=runtime_option)
# predict
im = cv2.imread(args.image_file)
tinypose_result = tinypose_model.predict(im)
print("Paddle TinyPose Result:\n", tinypose_result)

# visualize
vis_im = fd.vision.vis_keypoint_detection(
    im, tinypose_result, conf_threshold=0.5)
cv2.imwrite("visualized_result.jpg", vis_im)
print("TinyPose visualized result save in ./visualized_result.jpg")
