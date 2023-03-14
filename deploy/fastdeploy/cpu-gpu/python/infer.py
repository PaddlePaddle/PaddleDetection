import cv2
import os

import fastdeploy as fd


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="Path of PaddleSeg model.")
    parser.add_argument(
        "--image", type=str, required=True, help="Path of test image file.")
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Type of inference device, support 'kunlunxin', 'cpu' or 'gpu'.")
    parser.add_argument(
        "--use_trt",
        type=ast.literal_eval,
        default=False,
        help="Wether to use tensorrt.")
    return parser.parse_args()


def build_option(args):
    option = fd.RuntimeOption()

    if args.device.lower() == "gpu":
        option.use_gpu()

    if args.use_trt:
        option.use_trt_backend()
        # If use original Tensorrt, not Paddle-TensorRT,
        # comment the following two lines
        option.paddle_infer_option.enable_trt = True
        option.paddle_infer_option.collect_trt_shape = True
        option.trt_option.set_shape("image", [1, 3, 640, 640], [1, 3, 640, 640],
                                             [1, 3, 640, 640])
        option.set_trt_option.set_shape("scale_factor", [1, 2], [1, 2], [1, 2])
    return option


args = parse_arguments()

if args.model_dir is None:
    model_dir = fd.download_model(name='ppyoloe_crn_l_300e_coco')
else:
    model_dir = args.model_dir

model_file = os.path.join(model_dir, "model.pdmodel")
params_file = os.path.join(model_dir, "model.pdiparams")
config_file = os.path.join(model_dir, "infer_cfg.yml")

# settting for runtime
runtime_option = build_option(args)
model = fd.vision.detection.PPYOLOE(
    model_file, params_file, config_file, runtime_option=runtime_option)

# predict
if args.image is None:
    image = fd.utils.get_detection_test_image()
else:
    image = args.image
im = cv2.imread(image)
result = model.predict(im)
print(result)

# visualize
vis_im = fd.vision.vis_detection(im, result, score_threshold=0.5)
cv2.imwrite("visualized_result.jpg", vis_im)
print("Visualized result save in ./visualized_result.jpg")
