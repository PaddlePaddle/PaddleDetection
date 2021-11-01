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

#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <ctime>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "include/utils.h"

namespace PaddleDetection {

// Generate visualization color
cv::Scalar GetColor(int idx);

// Visualize Tracking Results
cv::Mat VisualizeTrackResult(const cv::Mat& img,
                     const MOTResult& results,
                     const float fps, const int frame_id);

// Pedestrian/Vehicle Counting
void FlowStatistic(const MOTResult& results, const int frame_id,
                   std::vector<int>* count_list, 
                   std::vector<int>* in_count_list, 
                   std::vector<int>* out_count_list);

// Save Tracking Results
void SaveResult(const MOTResult& results, const std::string& output_dir);

} // namespace PaddleDetection
