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

void CpuInfer(const std::string& tinypose_model_dir,
              const std::string& image_file) {
  auto tinypose_model_file = tinypose_model_dir + sep + "model.pdmodel";
  auto tinypose_params_file = tinypose_model_dir + sep + "model.pdiparams";
  auto tinypose_config_file = tinypose_model_dir + sep + "infer_cfg.yml";
  auto option = fastdeploy::RuntimeOption();
  option.UseCpu();
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

void GpuInfer(const std::string& tinypose_model_dir,
              const std::string& image_file) {
  auto option = fastdeploy::RuntimeOption();
  option.UseGpu();

  auto tinypose_model_file = tinypose_model_dir + sep + "model.pdmodel";
  auto tinypose_params_file = tinypose_model_dir + sep + "model.pdiparams";
  auto tinypose_config_file = tinypose_model_dir + sep + "infer_cfg.yml";
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

void TrtInfer(const std::string& tinypose_model_dir,
              const std::string& image_file) {
  auto tinypose_model_file = tinypose_model_dir + sep + "model.pdmodel";
  auto tinypose_params_file = tinypose_model_dir + sep + "model.pdiparams";
  auto tinypose_config_file = tinypose_model_dir + sep + "infer_cfg.yml";
  auto tinypose_option = fastdeploy::RuntimeOption();
  tinypose_option.UseGpu();
  tinypose_option.UsePaddleInferBackend();
  // If use original Tensorrt, not Paddle-TensorRT,
  // please try `option.UseTrtBackend()`
  tinypose_option.paddle_infer_option.enable_trt = true;
  tinypose_option.paddle_infer_option.collect_trt_shape = true;
  tinypose_option.trt_option.SetShape("image", {1, 3, 256, 192}, {1, 3, 256, 192}, 
                                      {1, 3, 256, 192});

  auto tinypose_model = fastdeploy::vision::keypointdetection::PPTinyPose(
      tinypose_model_file, tinypose_params_file, tinypose_config_file,
      tinypose_option);
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
  if (argc < 4) {
    std::cout << "Usage: infer_demo path/to/pptinypose_model_dir path/to/image "
                 "run_option, "
                 "e.g ./infer_demo ./pptinypose_model_dir ./test.jpeg 0"
              << std::endl;
    std::cout << "The data type of run_option is int, 0: run with cpu; 1: run "
                 "with gpu; 2: run with gpu and use tensorrt backend;"
              << std::endl;
    return -1;
  }

  if (std::atoi(argv[3]) == 0) {
    CpuInfer(argv[1], argv[2]);
  } else if (std::atoi(argv[3]) == 1) {
    GpuInfer(argv[1], argv[2]);
  } else if (std::atoi(argv[3]) == 2) {
    TrtInfer(argv[1], argv[2]);
  }
  return 0;
}
