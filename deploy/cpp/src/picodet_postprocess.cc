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
//
// The code is based on:
// https://github.com/RangiLyu/nanodet/blob/main/demo_mnn/nanodet_mnn.cpp

#include "include/picodet_postprocess.h"

namespace PaddleDetection {

float fast_exp(float x) {
  union {
    uint32_t i;
    float f;
  } v{};
  v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
  return v.f;
}

template <typename _Tp>
int activation_function_softmax(const _Tp *src, _Tp *dst, int length) {
  const _Tp alpha = *std::max_element(src, src + length);
  _Tp denominator{0};

  for (int i = 0; i < length; ++i) {
    dst[i] = fast_exp(src[i] - alpha);
    denominator += dst[i];
  }

  for (int i = 0; i < length; ++i) {
    dst[i] /= denominator;
  }

  return 0;
}

// PicoDet decode
PaddleDetection::ObjectResult
disPred2Bbox(const float *&dfl_det, int label, float score, int x, int y,
             int stride, std::vector<float> im_shape, int reg_max) {
  float ct_x = (x + 0.5) * stride;
  float ct_y = (y + 0.5) * stride;
  std::vector<float> dis_pred;
  dis_pred.resize(4);
  for (int i = 0; i < 4; i++) {
    float dis = 0;
    float *dis_after_sm = new float[reg_max + 1];
    activation_function_softmax(dfl_det + i * (reg_max + 1), dis_after_sm,
                                reg_max + 1);
    for (int j = 0; j < reg_max + 1; j++) {
      dis += j * dis_after_sm[j];
    }
    dis *= stride;
    dis_pred[i] = dis;
    delete[] dis_after_sm;
  }
  int xmin = (int)(std::max)(ct_x - dis_pred[0], .0f);
  int ymin = (int)(std::max)(ct_y - dis_pred[1], .0f);
  int xmax = (int)(std::min)(ct_x + dis_pred[2], (float)im_shape[0]);
  int ymax = (int)(std::min)(ct_y + dis_pred[3], (float)im_shape[1]);

  PaddleDetection::ObjectResult result_item;
  result_item.rect = {xmin, ymin, xmax, ymax};
  result_item.class_id = label;
  result_item.confidence = score;

  return result_item;
}

void PicoDetPostProcess(std::vector<PaddleDetection::ObjectResult> *results,
                        std::vector<const float *> outs,
                        std::vector<int> fpn_stride,
                        std::vector<float> im_shape,
                        std::vector<float> scale_factor, float score_threshold,
                        float nms_threshold, int num_class, int reg_max) {
  std::vector<std::vector<PaddleDetection::ObjectResult>> bbox_results;
  bbox_results.resize(num_class);
  int in_h = im_shape[0], in_w = im_shape[1];
  for (int i = 0; i < fpn_stride.size(); ++i) {
    int feature_h = std::ceil((float)in_h / fpn_stride[i]);
    int feature_w = std::ceil((float)in_w / fpn_stride[i]);
    for (int idx = 0; idx < feature_h * feature_w; idx++) {
      const float *scores = outs[i] + (idx * num_class);

      int row = idx / feature_w;
      int col = idx % feature_w;
      float score = 0;
      int cur_label = 0;
      for (int label = 0; label < num_class; label++) {
        if (scores[label] > score) {
          score = scores[label];
          cur_label = label;
        }
      }
      if (score > score_threshold) {
        const float *bbox_pred =
            outs[i + fpn_stride.size()] + (idx * 4 * (reg_max + 1));
        bbox_results[cur_label].push_back(
            disPred2Bbox(bbox_pred, cur_label, score, col, row, fpn_stride[i],
                         im_shape, reg_max));
      }
    }
  }
  for (int i = 0; i < (int)bbox_results.size(); i++) {
    PaddleDetection::nms(bbox_results[i], nms_threshold);

    for (auto box : bbox_results[i]) {
      box.rect[0] = box.rect[0] / scale_factor[1];
      box.rect[2] = box.rect[2] / scale_factor[1];
      box.rect[1] = box.rect[1] / scale_factor[0];
      box.rect[3] = box.rect[3] / scale_factor[0];
      results->push_back(box);
    }
  }
}

} // namespace PaddleDetection
