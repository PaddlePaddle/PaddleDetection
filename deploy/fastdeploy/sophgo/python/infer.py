# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import fastdeploy as fd
import cv2
import os


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file", required=True, help="Path of sophgo model.")
    parser.add_argument("--config_file", required=True, help="Path of config.")
    parser.add_argument(
        "--image_file", type=str, required=True, help="Path of test image file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    model_file = args.model_file
    params_file = ""
    config_file = args.config_file

    # setup runtime
    runtime_option = fd.RuntimeOption()
    runtime_option.use_sophgo()

    model = fd.vision.detection.PPYOLOE(
        model_file,
        params_file,
        config_file,
        runtime_option=runtime_option,
        model_format=fd.ModelFormat.SOPHGO)

    model.postprocessor.apply_nms()

    # predict
    im = cv2.imread(args.image_file)
    result = model.predict(im)
    print(result)

    # visualize
    vis_im = fd.vision.vis_detection(im, result, score_threshold=0.5)
    cv2.imwrite("sophgo_result.jpg", vis_im)
    print("Visualized result save in ./sophgo_result.jpg")
