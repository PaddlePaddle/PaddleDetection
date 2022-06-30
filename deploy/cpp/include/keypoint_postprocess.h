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

#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

namespace PaddleDetection {

std::vector<float> get_3rd_point(std::vector<float>& a, std::vector<float>& b);

std::vector<float> get_dir(float src_point_x, float src_point_y, float rot_rad);

void affine_tranform(
    float pt_x, float pt_y, cv::Mat& trans, std::vector<float>& preds, int p);

cv::Mat get_affine_transform(std::vector<float>& center,
                             std::vector<float>& scale,
                             float rot,
                             std::vector<int>& output_size,
                             int inv);

void transform_preds(std::vector<float>& coords,
                     std::vector<float>& center,
                     std::vector<float>& scale,
                     std::vector<int>& output_size,
                     std::vector<int>& dim,
                     std::vector<float>& target_coords,
                     bool affine = false);

void box_to_center_scale(std::vector<int>& box,
                         int width,
                         int height,
                         std::vector<float>& center,
                         std::vector<float>& scale);

void get_max_preds(float* heatmap,
                   std::vector<int>& dim,
                   std::vector<float>& preds,
                   float* maxvals,
                   int batchid,
                   int joint_idx);

void get_final_preds(std::vector<float>& heatmap,
                     std::vector<int>& dim,
                     std::vector<int64_t>& idxout,
                     std::vector<int>& idxdim,
                     std::vector<float>& center,
                     std::vector<float> scale,
                     std::vector<float>& preds,
                     int batchid,
                     bool DARK = true);

// Object KeyPoint Result
struct KeyPointResult {
  // Keypoints: shape(N x 3); N: number of Joints; 3: x,y,conf
  std::vector<float> keypoints;
  int num_joints = -1;
};

class PoseSmooth {
 public:
  explicit PoseSmooth(const int width,
                      const int height,
                      std::string filter_type = "OneEuro",
                      float alpha = 0.5,
                      float fc_d = 0.1,
                      float fc_min = 0.1,
                      float beta = 0.1,
                      float thres_mult = 0.3)
      : width(width),
        height(height),
        alpha(alpha),
        fc_d(fc_d),
        fc_min(fc_min),
        beta(beta),
        filter_type(filter_type),
        thres_mult(thres_mult){};

  // Run predictor
  KeyPointResult smooth_process(KeyPointResult* result);
  void PointSmooth(KeyPointResult* result,
                   KeyPointResult* keypoint_smoothed,
                   std::vector<float> thresholds,
                   int index);
  float OneEuroFilter(float x_cur, float x_pre, int loc);
  float smoothing_factor(float te, float fc);
  float ExpSmoothing(float x_cur, float x_pre, int loc = 0);

 private:
  int width = 0;
  int height = 0;
  float alpha = 0.;
  float fc_d = 1.;
  float fc_min = 0.;
  float beta = 1.;
  float thres_mult = 1.;
  std::string filter_type = "OneEuro";
  std::vector<float> thresholds = {0.005,
                                   0.005,
                                   0.005,
                                   0.005,
                                   0.005,
                                   0.01,
                                   0.01,
                                   0.01,
                                   0.01,
                                   0.01,
                                   0.01,
                                   0.01,
                                   0.01,
                                   0.01,
                                   0.01,
                                   0.01,
                                   0.01};
  KeyPointResult x_prev_hat;
  KeyPointResult dx_prev_hat;
};
}  // namespace PaddleDetection
