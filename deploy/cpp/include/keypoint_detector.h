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

#include <ctime>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "paddle_inference_api.h"  // NOLINT

#include "include/config_parser.h"
#include "include/keypoint_postprocess.h"
#include "include/preprocess_op.h"

using namespace paddle_infer;

namespace PaddleDetection {

// Visualiztion KeyPoint Result
cv::Mat VisualizeKptsResult(const cv::Mat& img,
                            const std::vector<KeyPointResult>& results,
                            const std::vector<int>& colormap);

class KeyPointDetector {
 public:
  explicit KeyPointDetector(const std::string& model_dir,
                            const std::string& device = "CPU",
                            bool use_mkldnn = false,
                            int cpu_threads = 1,
                            const std::string& run_mode = "paddle",
                            const int batch_size = 1,
                            const int gpu_id = 0,
                            const int trt_min_shape = 1,
                            const int trt_max_shape = 1280,
                            const int trt_opt_shape = 640,
                            bool trt_calib_mode = false,
                            bool use_dark = true) {
    this->device_ = device;
    this->gpu_id_ = gpu_id;
    this->cpu_math_library_num_threads_ = cpu_threads;
    this->use_mkldnn_ = use_mkldnn;
    this->use_dark = use_dark;

    this->trt_min_shape_ = trt_min_shape;
    this->trt_max_shape_ = trt_max_shape;
    this->trt_opt_shape_ = trt_opt_shape;
    this->trt_calib_mode_ = trt_calib_mode;
    config_.load_config(model_dir);
    this->use_dynamic_shape_ = config_.use_dynamic_shape_;
    this->min_subgraph_size_ = config_.min_subgraph_size_;
    threshold_ = config_.draw_threshold_;
    preprocessor_.Init(config_.preprocess_info_);
    LoadModel(model_dir, batch_size, run_mode);
  }

  // Load Paddle inference model
  void LoadModel(const std::string& model_dir,
                 const int batch_size = 1,
                 const std::string& run_mode = "paddle");

  // Run predictor
  void Predict(const std::vector<cv::Mat> imgs,
               std::vector<std::vector<float>>& center,
               std::vector<std::vector<float>>& scale,
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
  std::string device_ = "CPU";
  int gpu_id_ = 0;
  int cpu_math_library_num_threads_ = 1;
  bool use_dark = true;
  bool use_mkldnn_ = false;
  int min_subgraph_size_ = 3;
  bool use_dynamic_shape_ = false;
  int trt_min_shape_ = 1;
  int trt_max_shape_ = 1280;
  int trt_opt_shape_ = 640;
  bool trt_calib_mode_ = false;
  // Preprocess image and copy data to input buffer
  void Preprocess(const cv::Mat& image_mat);
  // Postprocess result
  void Postprocess(std::vector<float>& output,
                   std::vector<int> output_shape,
                   std::vector<int64_t>& idxout,
                   std::vector<int> idx_shape,
                   std::vector<KeyPointResult>* result,
                   std::vector<std::vector<float>>& center,
                   std::vector<std::vector<float>>& scale);

  std::shared_ptr<Predictor> predictor_;
  Preprocessor preprocessor_;
  ImageBlob inputs_;
  std::vector<float> output_data_;
  std::vector<int64_t> idx_data_;
  float threshold_;
  ConfigPaser config_;
};

}  // namespace PaddleDetection
