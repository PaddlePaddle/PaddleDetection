import fastdeploy as fd
import cv2
import os


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        required=True,
        help="path of PP-TinyPose model directory")
    parser.add_argument(
        "--image_file", required=True, help="path of test image file.")
    return parser.parse_args()


args = parse_arguments()

runtime_option = fd.RuntimeOption()
runtime_option.use_kunlunxin()

tinypose_model_file = os.path.join(args.model_dir, "model.pdmodel")
tinypose_params_file = os.path.join(args.model_dir, "model.pdiparams")
tinypose_config_file = os.path.join(args.model_dir, "infer_cfg.yml")
# setup runtime
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
