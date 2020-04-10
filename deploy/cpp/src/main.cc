//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <glog/logging.h>

#include <iostream>
#include <string>
#include <vector>

#include "include/object_detector.h"


DEFINE_string(model_dir, "", "Path of inference model");
DEFINE_string(image_path, "", "Path of input image");
DEFINE_string(video_path, "", "Path of input video");
DEFINE_bool(use_gpu, false, "Infering with GPU or CPU");

int main(int argc, char** argv) {
  // Parsing command-line
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_model_dir.empty()
      || (FLAGS_image_path.empty() && FLAGS_video_path.empty())) {
      std::cout << "Usage: ./main --model_dir=/PATH/TO/INFERENCE_MODEL/ "
                << "--image_path=/PATH/TO/INPUT/IMAGE/" << std::endl;
      return -1;
  }
  // Load model and create a object detector
  PaddleDetection::ObjectDetector det(FLAGS_model_dir, FLAGS_use_gpu);
  // Open input image as an opencv cv::Mat object
  cv::Mat im = cv::imread(FLAGS_image_path, 1);
  // Store all detected result
  std::vector<PaddleDetection::ObjectResult> result;
  det.Predict(im, &result);
  return 0;
}
