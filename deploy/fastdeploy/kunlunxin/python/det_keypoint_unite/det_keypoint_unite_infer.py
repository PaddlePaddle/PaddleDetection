import fastdeploy as fd
import cv2
import os


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tinypose_model_dir",
        required=True,
        help="path of paddletinypose model directory")
    parser.add_argument(
        "--det_model_dir", help="path of paddledetection model directory")
    parser.add_argument(
        "--image_file", required=True, help="path of test image file.")
    return parser.parse_args()


def build_picodet_option(args):
    option = fd.RuntimeOption()
    option.use_kunlunxin()                                  
    return option


def build_tinypose_option(args):
    option = fd.RuntimeOption()
    option.use_kunlunxin()
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
