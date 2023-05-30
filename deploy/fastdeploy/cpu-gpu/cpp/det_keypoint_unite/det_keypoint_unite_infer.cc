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
#include "fastdeploy/pipeline.h"

#ifdef WIN32
const char sep = '\\';
#else
const char sep = '/';
#endif

void CpuInfer(const std::string& det_model_dir,
              const std::string& tinypose_model_dir,
              const std::string& image_file) {
  auto det_model_file = det_model_dir + sep + "model.pdmodel";
  auto det_params_file = det_model_dir + sep + "model.pdiparams";
  auto det_config_file = det_model_dir + sep + "infer_cfg.yml";
  auto det_model = fastdeploy::vision::detection::PicoDet(
      det_model_file, det_params_file, det_config_file);
  if (!det_model.Initialized()) {
    std::cerr << "Detection Model Failed to initialize." << std::endl;
    return;
  }

  auto tinypose_model_file = tinypose_model_dir + sep + "model.pdmodel";
  auto tinypose_params_file = tinypose_model_dir + sep + "model.pdiparams";
  auto tinypose_config_file = tinypose_model_dir + sep + "infer_cfg.yml";
  auto tinypose_model = fastdeploy::vision::keypointdetection::PPTinyPose(
      tinypose_model_file, tinypose_params_file, tinypose_config_file);
  if (!tinypose_model.Initialized()) {
    std::cerr << "TinyPose Model Failed to initialize." << std::endl;
    return;
  }

  auto im = cv::imread(image_file);
  fastdeploy::vision::KeyPointDetectionResult res;

  auto pipeline =fastdeploy::pipeline::PPTinyPose(&det_model, &tinypose_model);
  pipeline.detection_model_score_threshold = 0.5;
  if (!pipeline.Predict(&im, &res)) {
    std::cerr << "TinyPose Prediction Failed." << std::endl;
    return;
  } else {
    std::cout << "TinyPose Prediction Done!" << std::endl;
  }
  
  std::cout << res.Str() << std::endl;

  auto vis_im =
      fastdeploy::vision::VisKeypointDetection(im, res, 0.2);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "TinyPose visualized result saved in ./vis_result.jpg"
            << std::endl;
}

void GpuInfer(const std::string& det_model_dir,
              const std::string& tinypose_model_dir,
              const std::string& image_file) {
  auto option = fastdeploy::RuntimeOption();
  option.UseGpu();
  auto det_model_file = det_model_dir + sep + "model.pdmodel";
  auto det_params_file = det_model_dir + sep + "model.pdiparams";
  auto det_config_file = det_model_dir + sep + "infer_cfg.yml";
  auto det_model = fastdeploy::vision::detection::PicoDet(
      det_model_file, det_params_file, det_config_file, option);
  if (!det_model.Initialized()) {
    std::cerr << "Detection Model Failed to initialize." << std::endl;
    return;
  }

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

  auto pipeline =
      fastdeploy::pipeline::PPTinyPose(
          &det_model, &tinypose_model);
  pipeline.detection_model_score_threshold = 0.5;
  if (!pipeline.Predict(&im, &res)) {
    std::cerr << "TinyPose Prediction Failed." << std::endl;
    return;
  } else {
    std::cout << "TinyPose Prediction Done!" << std::endl;
  }
  
  std::cout << res.Str() << std::endl;

  auto vis_im =
      fastdeploy::vision::VisKeypointDetection(im, res, 0.2);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "TinyPose visualized result saved in ./vis_result.jpg"
            << std::endl;
}

void TrtInfer(const std::string& det_model_dir,
              const std::string& tinypose_model_dir,
              const std::string& image_file) {
  auto det_model_file = det_model_dir + sep + "model.pdmodel";
  auto det_params_file = det_model_dir + sep + "model.pdiparams";
  auto det_config_file = det_model_dir + sep + "infer_cfg.yml";

  auto det_option = fastdeploy::RuntimeOption();
  det_option.UseGpu();
  det_option.UsePaddleInferBackend();
  // If use original Tensorrt, not Paddle-TensorRT,
  // please try `option.UseTrtBackend()`
  det_option.paddle_infer_option.enable_trt = true;
  det_option.paddle_infer_option.collect_trt_shape = true;
  det_option.trt_option.SetShape("image", {1, 3, 320, 320}, {1, 3, 320, 320}, 
                                 {1, 3, 320, 320});
  det_option.trt_option.SetShape("scale_factor", {1, 2}, {1, 2}, {1, 2});    
  auto det_model = fastdeploy::vision::detection::PicoDet(
      det_model_file, det_params_file, det_config_file, det_option);
  if (!det_model.Initialized()) {
    std::cerr << "Detection Model Failed to initialize." << std::endl;
    return;
  }

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

  auto pipeline =
      fastdeploy::pipeline::PPTinyPose(
          &det_model, &tinypose_model);
  pipeline.detection_model_score_threshold = 0.5;
  if (!pipeline.Predict(&im, &res)) {
    std::cerr << "TinyPose Prediction Failed." << std::endl;
    return;
  } else {
    std::cout << "TinyPose Prediction Done!" << std::endl;
  }
  
  std::cout << res.Str() << std::endl;

  auto vis_im =
      fastdeploy::vision::VisKeypointDetection(im, res, 0.2);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "TinyPose visualized result saved in ./vis_result.jpg"
            << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc < 5) {
    std::cout << "Usage: infer_demo path/to/detection_model_dir "
                 "path/to/pptinypose_model_dir path/to/image run_option, "
                 "e.g ./infer_model ./picodet_model_dir ./pptinypose_model_dir "
                 "./test.jpeg 0"
              << std::endl;
    std::cout << "The data type of run_option is int, 0: run with cpu; 1: run "
                 "with gpu; 2: run with gpu and use tensorrt backend;"
              << std::endl;
    return -1;
  }

  if (std::atoi(argv[4]) == 0) {
    CpuInfer(argv[1], argv[2], argv[3]);
  } else if (std::atoi(argv[4]) == 1) {
    GpuInfer(argv[1], argv[2], argv[3]);
  } else if (std::atoi(argv[4]) == 2) {
    TrtInfer(argv[1], argv[2], argv[3]);
  }
  return 0;
}
