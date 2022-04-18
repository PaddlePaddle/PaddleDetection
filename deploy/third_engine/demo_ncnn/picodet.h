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
// reference from https://github.com/RangiLyu/nanodet/tree/main/demo_ncnn

#ifndef PICODET_H
#define PICODET_H

#include <net.h>
#include <opencv2/core/core.hpp>

typedef struct NonPostProcessHeadInfo {
  std::string cls_layer;
  std::string dis_layer;
  int stride;
} NonPostProcessHeadInfo;

typedef struct BoxInfo {
  float x1;
  float y1;
  float x2;
  float y2;
  float score;
  int label;
} BoxInfo;

class PicoDet {
public:
  PicoDet(const char *param, const char *bin, int input_width, int input_hight,
          bool useGPU, float score_threshold_, float nms_threshold_);

  ~PicoDet();

  static PicoDet *detector;
  ncnn::Net *Net;
  static bool hasGPU;

  int detect(cv::Mat image, std::vector<BoxInfo> &result_list,
             bool has_postprocess);

private:
  void preprocess(cv::Mat &image, ncnn::Mat &in);
  void decode_infer(ncnn::Mat &cls_pred, ncnn::Mat &dis_pred, int stride,
                    float threshold,
                    std::vector<std::vector<BoxInfo>> &results);
  BoxInfo disPred2Bbox(const float *&dfl_det, int label, float score, int x,
                       int y, int stride);
  static void nms(std::vector<BoxInfo> &result, float nms_threshold);
  void nms_boxes(ncnn::Mat &cls_pred, ncnn::Mat &dis_pred,
                 float score_threshold,
                 std::vector<std::vector<BoxInfo>> &result_list);

  int image_w;
  int image_h;
  int in_w = 320;
  int in_h = 320;
  int num_class = 80;
  int reg_max = 7;

  float score_threshold;
  float nms_threshold;

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

#endif
