import cv2
import os

import fastdeploy as fd


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", required=True, help="Path of PaddleDetection model.")
    parser.add_argument(
        "--image_file", type=str, required=True, help="Path of test image file.")
    return parser.parse_args()

args = parse_arguments()

runtime_option = fd.RuntimeOption()
runtime_option.use_ascend()

if args.model_dir is None:
    model_dir = fd.download_model(name='ppyoloe_crn_l_300e_coco')
else:
    model_dir = args.model_dir

model_file = os.path.join(model_dir, "model.pdmodel")
params_file = os.path.join(model_dir, "model.pdiparams")
config_file = os.path.join(model_dir, "infer_cfg.yml")

# settting for runtime
model = fd.vision.detection.PPYOLOE(
    model_file, params_file, config_file, runtime_option=runtime_option)

# predict
if args.image_file is None:
    image_file = fd.utils.get_detection_test_image()
else:
    image_file = args.image_file
im = cv2.imread(image_file)
result = model.predict(im)
print(result)

# visualize
vis_im = fd.vision.vis_detection(im, result, score_threshold=0.5)
cv2.imwrite("visualized_result.jpg", vis_im)
print("Visualized result save in ./visualized_result.jpg")
