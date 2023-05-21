// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "fastdeploy/vision.h"

#ifdef WIN32
const char sep = '\\';
#else
const char sep = '/';
#endif

void KunlunXinInfer(const std::string& tinypose_model_dir,
              const std::string& image_file) {
  auto tinypose_model_file = tinypose_model_dir + sep + "model.pdmodel";
  auto tinypose_params_file = tinypose_model_dir + sep + "model.pdiparams";
  auto tinypose_config_file = tinypose_model_dir + sep + "infer_cfg.yml";
  auto option = fastdeploy::RuntimeOption();
  option.UseKunlunXin();
  auto tinypose_model = fastdeploy::vision::keypointdetection::PPTinyPose(
      tinypose_model_file, tinypose_params_file, tinypose_config_file, option);
  if (!tinypose_model.Initialized()) {
    std::cerr << "TinyPose Model Failed to initialize." << std::endl;
    return;
  }

  auto im = cv::imread(image_file);
  fastdeploy::vision::KeyPointDetectionResult res;
  if (!tinypose_model.Predict(&im, &res)) {
    std::cerr << "TinyPose Prediction Failed." << std::endl;
    return;
  } else {
    std::cout << "TinyPose Prediction Done!" << std::endl;
  }
  
  std::cout << res.Str() << std::endl;

  auto tinypose_vis_im =
      fastdeploy::vision::VisKeypointDetection(im, res, 0.5);
  cv::imwrite("tinypose_vis_result.jpg", tinypose_vis_im);
  std::cout << "TinyPose visualized result saved in ./tinypose_vis_result.jpg"
            << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout
        << "Usage: infer_demo path/to/model_dir path/to/image, "
           "e.g ./infer_demo ./model_dir ./test.jpeg"
        << std::endl;
    return -1;
  }
  KunlunXinInfer(argv[1], argv[2]);
  return 0;
}
