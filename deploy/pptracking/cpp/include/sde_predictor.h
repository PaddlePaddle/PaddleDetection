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
#include "include/preprocess_op.h"
#include "include/utils.h"

using namespace paddle_infer;  // NOLINT

namespace PaddleDetection {

class SDEPredictor {
 public:
  explicit SDEPredictor(const std::string& device,
                        const std::string& det_model_dir = "",
                        const std::string& reid_model_dir = "",
                        const double threshold = -1.,
                        const std::string& run_mode = "fluid",
                        const int gpu_id = 0,
                        const bool use_mkldnn = false,
                        const int cpu_threads = 1,
                        bool trt_calib_mode = false,
                        const int min_box_area = 200) {
    this->device_ = device;
    this->gpu_id_ = gpu_id;
    this->use_mkldnn_ = use_mkldnn;
    this->cpu_math_library_num_threads_ = cpu_threads;
    this->trt_calib_mode_ = trt_calib_mode;
    this->min_box_area_ = min_box_area;

    det_config_.load_config(det_model_dir);
    this->min_subgraph_size_ = det_config_.min_subgraph_size_;
    det_preprocessor_.Init(det_config_.preprocess_info_);

    reid_config_.load_config(reid_model_dir);
    reid_preprocessor_.Init(reid_config_.preprocess_info_);

    LoadModel(det_model_dir, reid_model_dir, run_mode);
    this->conf_thresh_ = det_config_.conf_thresh_;
  }

  // Load Paddle inference model
  void LoadModel(const std::string& det_model_dir,
                 const std::string& reid_model_dir,
                 const std::string& run_mode = "fluid");

  // Run predictor
  void Predict(const std::vector<cv::Mat> imgs,
               const double threshold = 0.5,
               MOTResult* result = nullptr,
               std::vector<double>* times = nullptr);

 private:
  std::string device_ = "CPU";
  float threhold = 0.5;
  int gpu_id_ = 0;
  bool use_mkldnn_ = false;
  int cpu_math_library_num_threads_ = 1;
  int min_subgraph_size_ = 3;
  bool trt_calib_mode_ = false;

  // Preprocess image and copy data to input buffer
  void Preprocess(const cv::Mat& image_mat);
  // Postprocess result
  void Postprocess(const cv::Mat dets, const cv::Mat emb, MOTResult* result);

  std::shared_ptr<Predictor> det_predictor_;
  std::shared_ptr<Predictor> reid_predictor_;
  Preprocessor det_preprocessor_;
  Preprocessor reid_preprocessor_;
  ImageBlob inputs_;
  std::vector<float> bbox_data_;
  std::vector<float> emb_data_;
  double threshold_;
  ConfigPaser det_config_;
  ConfigPaser reid_config_;
  float min_box_area_ = 200;
  float conf_thresh_;
};

}  // namespace PaddleDetection
