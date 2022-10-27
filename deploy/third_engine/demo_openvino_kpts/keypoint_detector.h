//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <inference_engine.hpp>

#include "keypoint_postprocess.h"

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
                            const std::vector<int>& colormap,
                            float threshold = 0.2);

class KeyPointDetector {
 public:
  explicit KeyPointDetector(const std::string& model_path,
                            int input_height = 256,
                            int input_width = 192,
                            float score_threshold = 0.3,
                            const int batch_size = 1,
                            bool use_dark = true) {
    use_dark_ = use_dark;

    in_w = input_width;
    in_h = input_height;
    threshold_ = score_threshold;

    InferenceEngine::Core ie;
    auto model = ie.ReadNetwork(model_path);
    // prepare input settings
    InferenceEngine::InputsDataMap inputs_map(model.getInputsInfo());
    input_name_ = inputs_map.begin()->first;
    InferenceEngine::InputInfo::Ptr input_info = inputs_map.begin()->second;
    // prepare output settings
    InferenceEngine::OutputsDataMap outputs_map(model.getOutputsInfo());
    int idx = 0;
    for (auto& output_info : outputs_map) {
      if (idx == 0) {
        output_info.second->setPrecision(InferenceEngine::Precision::FP32);
      } else {
        output_info.second->setPrecision(InferenceEngine::Precision::FP32);
      }
      idx++;
    }

    // get network
    network_ = ie.LoadNetwork(model, "CPU");
    infer_request_ = network_.CreateInferRequest();
  }

  // Load Paddle inference model
  void LoadModel(std::string model_file, int num_theads);

  // Run predictor
  void Predict(const std::vector<cv::Mat> imgs,
               std::vector<std::vector<float>>& center,
               std::vector<std::vector<float>>& scale,
               std::vector<KeyPointResult>* result = nullptr);

  bool use_dark() { return this->use_dark_; }

  inline float get_threshold() { return threshold_; };

  int in_w = 128;
  int in_h = 256;

 private:
  // Postprocess result
  void Postprocess(std::vector<float>& output,
                   std::vector<uint64_t>& output_shape,
                   std::vector<float>& idxout,
                   std::vector<uint64_t>& idx_shape,
                   std::vector<KeyPointResult>* result,
                   std::vector<std::vector<float>>& center,
                   std::vector<std::vector<float>>& scale);

  std::vector<float> output_data_;
  std::vector<float> idx_data_;
  float threshold_;
  bool use_dark_;

  InferenceEngine::ExecutableNetwork network_;
  InferenceEngine::InferRequest infer_request_;
  std::string input_name_;
};

}  // namespace PaddleDetection
