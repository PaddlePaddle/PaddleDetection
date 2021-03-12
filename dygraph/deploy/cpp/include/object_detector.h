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

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <ctime>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "paddle_inference_api.h" // NOLINT

#include "include/preprocess_op.h"
#include "include/config_parser.h"

using namespace paddle_infer;

namespace PaddleDetection {
// Object Detection Result
struct ObjectResult {
  // Rectangle coordinates of detected object: left, right, top, down
  std::vector<int> rect;
  // Class id of detected object
  int class_id;
  // Confidence of detected object
  float confidence;
};


// Generate visualization colormap for each class
std::vector<int> GenerateColorMap(int num_class);


// Visualiztion Detection Result
cv::Mat VisualizeResult(const cv::Mat& img,
                     const std::vector<ObjectResult>& results,
                     const std::vector<std::string>& lable_list,
                     const std::vector<int>& colormap);


class ObjectDetector {
 public:
  explicit ObjectDetector(const std::string& model_dir, 
                          bool use_gpu=false,
                          const std::string& run_mode="fluid",
                          const int gpu_id=0,
                          bool use_dynamic_shape=false,
                          const int trt_min_shape=1,
                          const int trt_max_shape=1280,
                          const int trt_opt_shape=640) {
    config_.load_config(model_dir);
    threshold_ = config_.draw_threshold_;
    image_shape_ = config_.image_shape_;
    preprocessor_.Init(config_.preprocess_info_, image_shape_);
    LoadModel(model_dir, use_gpu, config_.min_subgraph_size_, 1, run_mode, gpu_id,
    use_dynamic_shape, trt_min_shape, trt_max_shape, trt_opt_shape);
  }

  // Load Paddle inference model
  void LoadModel(
    const std::string& model_dir,
    bool use_gpu,
    const int min_subgraph_size,
    const int batch_size = 1,
    const std::string& run_mode = "fluid",
    const int gpu_id=0,
    bool use_dynamic_shape=false,
    const int trt_min_shape=1,
    const int trt_max_shape=1280,
    const int trt_opt_shape=640);

  // Run predictor
  void Predict(const cv::Mat& im,
      const double threshold = 0.5,
      const int warmup = 0,
      const int repeats = 1,
      const bool run_benchmark = false,
      std::vector<ObjectResult>* result = nullptr);

  // Get Model Label list
  const std::vector<std::string>& GetLabelList() const {
    return config_.label_list_;
  }

 private:
  // Preprocess image and copy data to input buffer
  void Preprocess(const cv::Mat& image_mat);
  // Postprocess result
  void Postprocess(
      const cv::Mat& raw_mat,
      std::vector<ObjectResult>* result);

  std::shared_ptr<Predictor> predictor_;
  Preprocessor preprocessor_;
  ImageBlob inputs_;
  std::vector<float> output_data_;
  float threshold_;
  ConfigPaser config_;
  std::vector<int> image_shape_;
};

}  // namespace PaddleDetection
