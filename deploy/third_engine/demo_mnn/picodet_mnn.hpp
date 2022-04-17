// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#ifndef __PicoDet_H__
#define __PicoDet_H__

#pragma once

#include "Interpreter.hpp"

#include "ImageProcess.hpp"
#include "MNNDefine.h"
#include "Tensor.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

typedef struct NonPostProcessHeadInfo_ {
  std::string cls_layer;
  std::string dis_layer;
  int stride;
} NonPostProcessHeadInfo;

typedef struct BoxInfo_ {
  float x1;
  float y1;
  float x2;
  float y2;
  float score;
  int label;
} BoxInfo;

class PicoDet {
public:
  PicoDet(const std::string &mnn_path, int input_width, int input_length,
          int num_thread_ = 4, float score_threshold_ = 0.5,
          float nms_threshold_ = 0.3);

  ~PicoDet();

  int detect(cv::Mat &img, std::vector<BoxInfo> &result_list,
             bool has_postprocess);

private:
  void decode_infer(MNN::Tensor *cls_pred, MNN::Tensor *dis_pred, int stride,
                    float threshold,
                    std::vector<std::vector<BoxInfo>> &results);
  BoxInfo disPred2Bbox(const float *&dfl_det, int label, float score, int x,
                       int y, int stride);
  void nms(std::vector<BoxInfo> &input_boxes, float NMS_THRESH);

private:
  std::shared_ptr<MNN::Interpreter> PicoDet_interpreter;
  MNN::Session *PicoDet_session = nullptr;
  MNN::Tensor *input_tensor = nullptr;

  int num_thread;
  int image_w;
  int image_h;

  int in_w = 320;
  int in_h = 320;

  float score_threshold;
  float nms_threshold;

  const float mean_vals[3] = {103.53f, 116.28f, 123.675f};
  const float norm_vals[3] = {0.017429f, 0.017507f, 0.017125f};

  const int num_class = 80;
  const int reg_max = 7;

  std::vector<float> bbox_output_data_;
  std::vector<float> class_output_data_;

  std::vector<std::string> nms_heads_info{"tmp_16", "concat_4.tmp_0"};
  // If not export post-process, will use non_postprocess_heads_info
  std::vector<NonPostProcessHeadInfo> non_postprocess_heads_info{
      // cls_pred|dis_pred|stride
      {"transpose_0.tmp_0", "transpose_1.tmp_0", 8},
      {"transpose_2.tmp_0", "transpose_3.tmp_0", 16},
      {"transpose_4.tmp_0", "transpose_5.tmp_0", 32},
      {"transpose_6.tmp_0", "transpose_7.tmp_0", 64},
  };
};

template <typename _Tp>
int activation_function_softmax(const _Tp *src, _Tp *dst, int length);

inline float fast_exp(float x);
inline float sigmoid(float x);

#endif
