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
// reference from https://github.com/RangiLyu/nanodet/tree/main/demo_mnn

#ifndef __PicoDet_H__
#define __PicoDet_H__

#pragma once

#include "Interpreter.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "ImageProcess.hpp"
#include "MNNDefine.h"
#include "Tensor.hpp"

typedef struct HeadInfo_ {
  std::string cls_layer;
  std::string dis_layer;
  int stride;
} HeadInfo;

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
  PicoDet(const std::string &mnn_path,
          int input_width,
          int input_length,
          int num_thread_ = 4,
          float score_threshold_ = 0.5,
          float nms_threshold_ = 0.3);

  ~PicoDet();

  int detect(cv::Mat &img, std::vector<BoxInfo> &result_list);
  std::string get_label_str(int label);

 private:
  void decode_infer(MNN::Tensor *cls_pred,
                    MNN::Tensor *dis_pred,
                    int stride,
                    float threshold,
                    std::vector<std::vector<BoxInfo>> &results);
  BoxInfo disPred2Bbox(
      const float *&dfl_det, int label, float score, int x, int y, int stride);
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

  std::vector<HeadInfo> heads_info{
      // cls_pred|dis_pred|stride
      {"save_infer_model/scale_0.tmp_1", "save_infer_model/scale_4.tmp_1", 8},
      {"save_infer_model/scale_1.tmp_1", "save_infer_model/scale_5.tmp_1", 16},
      {"save_infer_model/scale_2.tmp_1", "save_infer_model/scale_6.tmp_1", 32},
      {"save_infer_model/scale_3.tmp_1", "save_infer_model/scale_7.tmp_1", 64},
  };

  std::vector<std::string> labels{
      "person",        "bicycle",      "car",
      "motorcycle",    "airplane",     "bus",
      "train",         "truck",        "boat",
      "traffic light", "fire hydrant", "stop sign",
      "parking meter", "bench",        "bird",
      "cat",           "dog",          "horse",
      "sheep",         "cow",          "elephant",
      "bear",          "zebra",        "giraffe",
      "backpack",      "umbrella",     "handbag",
      "tie",           "suitcase",     "frisbee",
      "skis",          "snowboard",    "sports ball",
      "kite",          "baseball bat", "baseball glove",
      "skateboard",    "surfboard",    "tennis racket",
      "bottle",        "wine glass",   "cup",
      "fork",          "knife",        "spoon",
      "bowl",          "banana",       "apple",
      "sandwich",      "orange",       "broccoli",
      "carrot",        "hot dog",      "pizza",
      "donut",         "cake",         "chair",
      "couch",         "potted plant", "bed",
      "dining table",  "toilet",       "tv",
      "laptop",        "mouse",        "remote",
      "keyboard",      "cell phone",   "microwave",
      "oven",          "toaster",      "sink",
      "refrigerator",  "book",         "clock",
      "vase",          "scissors",     "teddy bear",
      "hair drier",    "toothbrush"};
};

template <typename _Tp>
int activation_function_softmax(const _Tp *src, _Tp *dst, int length);

inline float fast_exp(float x);
inline float sigmoid(float x);

#endif
