//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle_api.h"  // NOLINT

#include "include/preprocess_op.h"
#include "include/config_parser.h"
#include "include/keypoint_postprocess.h"

using namespace paddle::lite_api;  // NOLINT

namespace PaddleDetection {
// Object KeyPoint Result
struct KeyPointResult {
  // Keypoints: shape(N x 3); N: number of Joints; 3: x,y,conf
  std::vector<float> keypoints;
  int num_joints = -1;
};

// Visualiztion KeyPoint Result
cv::Mat VisualizeKptsResult(const cv::Mat& img,
                     const std::vector<KeyPointResult>& results,
                     const std::vector<int>& colormap);


class KeyPointDetector {
 public:
  explicit KeyPointDetector(const std::string& model_dir, 
                          int cpu_threads=1,
                          const int batch_size=1) {
    config_.load_config(model_dir);
    threshold_ = config_.draw_threshold_;
    preprocessor_.Init(config_.preprocess_info_);
    printf("before keypoint detector\n");
    LoadModel(model_dir, cpu_threads);
    printf("create keypoint detector\n");
  }

  // Load Paddle inference model
  void LoadModel(std::string model_file,
               int num_theads);

  // Run predictor
  void Predict(const std::vector<cv::Mat> imgs,
      std::vector<std::vector<float>> &center,
      std::vector<std::vector<float>> &scale,
      const double threshold = 0.5,
      const int warmup = 0,
      const int repeats = 1,
      std::vector<KeyPointResult>* result = nullptr,
      std::vector<double>* times = nullptr);

  // Get Model Label list
  const std::vector<std::string>& GetLabelList() const {
    return config_.label_list_;
  }

 private:
  // Preprocess image and copy data to input buffer
  void Preprocess(const cv::Mat& image_mat);
  // Postprocess result
  void Postprocess(
      const std::vector<float> output,
      const std::vector<int64_t> output_shape,
      const std::vector<int64_t> idxout,
      const std::vector<int64_t> idx_shape,
      std::vector<KeyPointResult>* result,
      std::vector<std::vector<float>> &center,
      std::vector<std::vector<float>> &scale);

  std::shared_ptr<PaddlePredictor> predictor_;
  Preprocessor preprocessor_;
  ImageBlob inputs_;
  std::vector<float> output_data_;
  std::vector<int64_t> idx_data_;
  float threshold_;
  ConfigPaser config_;
};

}  // namespace PaddleDetection
