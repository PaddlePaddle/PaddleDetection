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

#include <cmath>
#include <ctime>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "include/utils.h"

namespace PaddleDetection {

void PicoDetPostProcess(std::vector<PaddleDetection::ObjectResult> *results,
                        std::vector<const float *> outs,
                        std::vector<int> fpn_stride,
                        std::vector<float> im_shape,
                        std::vector<float> scale_factor,
                        float score_threshold = 0.3, float nms_threshold = 0.5,
                        int num_class = 80, int reg_max = 7);

} // namespace PaddleDetection
