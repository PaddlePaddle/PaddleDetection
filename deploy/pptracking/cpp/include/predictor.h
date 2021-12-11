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
#include "include/jde_predictor.h"
#include "include/preprocess_op.h"
#include "include/sde_predictor.h"

using namespace paddle_infer;  // NOLINT

namespace PaddleDetection {

class Predictor {
 public:
  explicit Predictor(const std::string& device = "CPU",
                     const std::string& track_model_dir = "",
                     const std::string& det_model_dir = "",
                     const std::string& reid_model_dir = "",
                     const double threshold = -1.,
                     const std::string& run_mode = "paddle",
                     const int gpu_id = 0,
                     const bool use_mkldnn = false,
                     const int cpu_threads = 1,
                     bool trt_calib_mode = false,
                     const int min_box_area = 200) {
    if (track_model_dir.empty() && det_model_dir.empty()) {
      throw "Predictor must receive track_model or det_model!";
    }

    if (!track_model_dir.empty() && !det_model_dir.empty()) {
      throw "Predictor only receive one of track_model or det_model!";
    }

    if (!track_model_dir.empty()) {
      jde_sct_ =
          std::make_shared<PaddleDetection::JDEPredictor>(device,
                                                          track_model_dir,
                                                          threshold,
                                                          run_mode,
                                                          gpu_id,
                                                          use_mkldnn,
                                                          cpu_threads,
                                                          trt_calib_mode,
                                                          min_box_area);
      use_jde_ = true;
    }
    if (!det_model_dir.empty()) {
      sde_sct_ = std::make_shared<PaddleDetection::SDEPredictor>(device,
                                                                 det_model_dir,
                                                                 reid_model_dir,
                                                                 threshold,
                                                                 run_mode,
                                                                 gpu_id,
                                                                 use_mkldnn,
                                                                 cpu_threads,
                                                                 trt_calib_mode,
                                                                 min_box_area);
      use_jde_ = false;
    }
  }

  // Run predictor
  void Predict(const std::vector<cv::Mat> imgs,
               const double threshold = 0.5,
               MOTResult* result = nullptr,
               std::vector<double>* times = nullptr);

 private:
  std::shared_ptr<PaddleDetection::JDEPredictor> jde_sct_;
  std::shared_ptr<PaddleDetection::SDEPredictor> sde_sct_;
  bool use_jde_ = true;
};

}  // namespace PaddleDetection
