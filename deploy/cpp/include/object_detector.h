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

#include <ctime>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "paddle_inference_api.h"  // NOLINT

#include "include/config_parser.h"
#include "include/picodet_postprocess.h"
#include "include/preprocess_op.h"
#include "include/utils.h"

using namespace paddle_infer;

namespace PaddleDetection {

// Generate visualization colormap for each class
std::vector<int> GenerateColorMap(int num_class);

// Visualiztion Detection Result
cv::Mat VisualizeResult(
    const cv::Mat& img,
    const std::vector<PaddleDetection::ObjectResult>& results,
    const std::vector<std::string>& lables,
    const std::vector<int>& colormap,
    const bool is_rbox);

class ObjectDetector {
 public:
  explicit ObjectDetector(const std::string& model_dir,
                          const std::string& device = "CPU",
                          bool use_mkldnn = false,
                          int cpu_threads = 1,
                          const std::string& run_mode = "paddle",
                          const int batch_size = 1,
                          const int gpu_id = 0,
                          const int trt_min_shape = 1,
                          const int trt_max_shape = 1280,
                          const int trt_opt_shape = 640,
                          bool trt_calib_mode = false) {
    this->device_ = device;
    this->gpu_id_ = gpu_id;
    this->cpu_math_library_num_threads_ = cpu_threads;
    this->use_mkldnn_ = use_mkldnn;

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
               const double threshold = 0.5,
               const int warmup = 0,
               const int repeats = 1,
               std::vector<PaddleDetection::ObjectResult>* result = nullptr,
               std::vector<int>* bbox_num = nullptr,
               std::vector<double>* times = nullptr);

  // Get Model Label list
  const std::vector<std::string>& GetLabelList() const {
    return config_.label_list_;
  }

 private:
  std::string device_ = "CPU";
  int gpu_id_ = 0;
  int cpu_math_library_num_threads_ = 1;
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
  void Postprocess(const std::vector<cv::Mat> mats,
                   std::vector<PaddleDetection::ObjectResult>* result,
                   std::vector<int> bbox_num,
                   std::vector<float> output_data_,
                   std::vector<int> output_mask_data_,
                   bool is_rbox);

  std::shared_ptr<Predictor> predictor_;
  Preprocessor preprocessor_;
  ImageBlob inputs_;
  float threshold_;
  ConfigPaser config_;
};

}  // namespace PaddleDetection
