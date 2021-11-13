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

#include <glog/logging.h>

#include <ctime>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "include/utils.h"

namespace PaddleDetection {

// Generate visualization color
cv::Scalar GetColor(int idx);

// Visualize Tracking Results
cv::Mat VisualizeTrackResult(const cv::Mat& img,
                             const MOTResult& results,
                             const float fps,
                             const int frame_id);

// Pedestrian/Vehicle Counting
void FlowStatistic(const MOTResult& results,
                   const int frame_id,
                   const int secs_interval,
                   const bool do_entrance_counting,
                   const int video_fps,
                   const Rect entrance,
                   std::set<int>* id_set,
                   std::set<int>* interval_id_set,
                   std::vector<int>* in_id_list,
                   std::vector<int>* out_id_list,
                   std::map<int, std::vector<float>>* prev_center,
                   std::vector<std::string>* records);

// Save Tracking Results
void SaveMOTResult(const MOTResult& results,
                   const int frame_id,
                   std::vector<std::string>* records);

}  // namespace PaddleDetection
