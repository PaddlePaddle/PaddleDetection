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

#include "Interpreter.hpp"

#include "ImageProcess.hpp"
#include "MNNDefine.h"
#include "Tensor.hpp"

#include "keypoint_postprocess.h"

using namespace MNN;

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
                            int num_thread = 4,
                            int input_height = 256,
                            int input_width = 192,
                            float score_threshold = 0.3,
                            const int batch_size = 1,
                            bool use_dark = true) {
    printf("config path: %s",
           model_path.substr(0, model_path.find_last_of('/') + 1).c_str());
    use_dark_ = use_dark;

    in_w = input_width;
    in_h = input_height;
    threshold_ = score_threshold;

    KeyPointDet_interpreter = std::shared_ptr<MNN::Interpreter>(
        MNN::Interpreter::createFromFile(model_path.c_str()));
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    /*modeNum means gpuMode for GPU usage, Or means numThread for CPU usage.*/
    config.numThread = num_thread;
    // If type not fount, let it failed
    config.backupType = MNN_FORWARD_CPU;
    BackendConfig backendConfig;
    backendConfig.precision = static_cast<MNN::BackendConfig::PrecisionMode>(1);
    config.backendConfig = &backendConfig;

    KeyPointDet_session = KeyPointDet_interpreter->createSession(config);

    input_tensor =
        KeyPointDet_interpreter->getSessionInput(KeyPointDet_session, nullptr);
  }

  ~KeyPointDetector() {
    KeyPointDet_interpreter->releaseModel();
    KeyPointDet_interpreter->releaseSession(KeyPointDet_session);
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

  // const float mean_vals[3] = { 103.53f, 116.28f, 123.675f };
  // const float norm_vals[3] = { 0.017429f, 0.017507f, 0.017125f };
  const float mean_vals[3] = {0.f, 0.f, 0.f};
  const float norm_vals[3] = {1.f, 1.f, 1.f};
  int in_w = 128;
  int in_h = 256;

 private:
  // Postprocess result
  void Postprocess(std::vector<float>& output,
                   std::vector<int>& output_shape,
                   std::vector<int>& idxout,
                   std::vector<int>& idx_shape,
                   std::vector<KeyPointResult>* result,
                   std::vector<std::vector<float>>& center,
                   std::vector<std::vector<float>>& scale);

  std::vector<float> output_data_;
  std::vector<int> idx_data_;
  float threshold_;
  bool use_dark_;

  std::shared_ptr<MNN::Interpreter> KeyPointDet_interpreter;
  MNN::Session* KeyPointDet_session = nullptr;
  MNN::Tensor* input_tensor = nullptr;
};

}  // namespace PaddleDetection
