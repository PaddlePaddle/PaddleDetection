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
#include <sstream>
// for setprecision
#include <chrono>
#include <iomanip>
#include "include/predictor.h"

using namespace paddle_infer;  // NOLINT

namespace PaddleDetection {

void Predictor::Predict(const std::vector<cv::Mat> imgs,
                        const double threshold,
                        MOTResult* result,
                        std::vector<double>* times) {
  if (use_jde_) {
    jde_sct_->Predict(imgs, threshold, result, times);
  } else {
    sde_sct_->Predict(imgs, threshold, result, times);
  }
}

}  // namespace PaddleDetection
