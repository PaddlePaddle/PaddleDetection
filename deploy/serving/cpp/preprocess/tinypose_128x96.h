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

#pragma once
#include "core/general-server/general_model_service.pb.h"
#include "core/general-server/op/general_infer_helper.h"
#include "paddle_inference_api.h" // NOLINT
#include <string>
#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include <cstring>
#include <fstream>
#include <numeric>

namespace baidu {
namespace paddle_serving {
namespace serving {

class tinypose_128x96
    : public baidu::paddle_serving::predictor::OpWithChannel<GeneralBlob> {
public:
  typedef std::vector<paddle::PaddleTensor> TensorVector;

  DECLARE_OP(tinypose_128x96);

  int inference();

private:
  // preprocess
  std::vector<float> mean_ = {0.485f, 0.456f, 0.406f};
  std::vector<float> scale_ = {0.229f, 0.224f, 0.225f};
  bool is_scale_ = true;
  int im_shape_h = 128;
  int im_shape_w = 96;
  float scale_factor_h = 1.0f;
  float scale_factor_w = 1.0f;
  void preprocess_det(const cv::Mat &img, float *data, float &scale_factor_h,
                      float &scale_factor_w, int im_shape_h, int im_shape_w,
                      const std::vector<float> &mean,
                      const std::vector<float> &scale, const bool is_scale);

  // read pics
  cv::Mat Base2Mat(std::string &base64_data);
  std::string base64Decode(const char *Data, int DataByte);
};

} // namespace serving
} // namespace paddle_serving
} // namespace baidu
