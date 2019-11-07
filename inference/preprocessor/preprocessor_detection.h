// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "preprocessor.h"

namespace PaddleSolution {

class DetectionPreProcessor : public ImagePreProcessor {
 public:
    DetectionPreProcessor() : _config(nullptr) {
    }

    bool init(std::shared_ptr<PaddleSolution::PaddleModelConfigPaser> config);

    bool single_process(const std::string& fname, std::vector<float> &data,
                        int* ori_w, int* ori_h, int* resize_w,
                        int* resize_h, float* scale_ratio);

    bool batch_process(const std::vector<std::string>& imgs,
                       std::vector<std::vector<float>> &data,
                       int* ori_w, int* ori_h, int* resize_w,
                       int* resize_h, float* scale_ratio);
 private:
    std::shared_ptr<PaddleSolution::PaddleModelConfigPaser> _config;
};

}  // namespace PaddleSolution
