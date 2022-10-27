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

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

std::vector<float> get_3rd_point(std::vector<float>& a, std::vector<float>& b);
std::vector<float> get_dir(float src_point_x, float src_point_y, float rot_rad);
void affine_tranform(float pt_x,
                     float pt_y,
                     cv::Mat& trans,
                     std::vector<float>& x,
                     int p,
                     int num);
cv::Mat get_affine_transform(std::vector<float>& center,
                             std::vector<float>& scale,
                             float rot,
                             std::vector<int>& output_size,
                             int inv);
void transform_preds(std::vector<float>& coords,
                     std::vector<float>& center,
                     std::vector<float>& scale,
                     std::vector<uint64_t>& output_size,
                     std::vector<int>& dim,
                     std::vector<float>& target_coords,
                     bool affine);
void box_to_center_scale(std::vector<int>& box,
                         int width,
                         int height,
                         std::vector<float>& center,
                         std::vector<float>& scale);
void get_max_preds(std::vector<float>& heatmap,
                   std::vector<int>& dim,
                   std::vector<float>& preds,
                   std::vector<float>& maxvals,
                   int batchid,
                   int joint_idx);
void get_final_preds(std::vector<float>& heatmap,
                     std::vector<uint64_t>& dim,
                     std::vector<float>& idxout,
                     std::vector<uint64_t>& idxdim,
                     std::vector<float>& center,
                     std::vector<float> scale,
                     std::vector<float>& preds,
                     int batchid,
                     bool DARK = true);

void CropImg(cv::Mat& img,
             cv::Mat& crop_img,
             std::vector<int>& area,
             std::vector<float>& center,
             std::vector<float>& scale,
             float expandratio = 0.25);
