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

#include "picodet.h"
#include <benchmark.h>
#include <iostream>

inline float fast_exp(float x) {
  union {
    uint32_t i;
    float f;
  } v{};
  v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
  return v.f;
}

inline float sigmoid(float x) { return 1.0f / (1.0f + fast_exp(-x)); }

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

bool PicoDet::hasGPU = false;
PicoDet *PicoDet::detector = nullptr;

PicoDet::PicoDet(const char *param, const char *bin, int input_width,
                 int input_hight, bool useGPU, float score_threshold_ = 0.5,
                 float nms_threshold_ = 0.3) {
  this->Net = new ncnn::Net();
#if NCNN_VULKAN
  this->hasGPU = ncnn::get_gpu_count() > 0;
#endif
  this->Net->opt.use_vulkan_compute = this->hasGPU && useGPU;
  this->Net->opt.use_fp16_arithmetic = true;
  this->Net->load_param(param);
  this->Net->load_model(bin);
  this->in_w = input_width;
  this->in_h = input_hight;
  this->score_threshold = score_threshold_;
  this->nms_threshold = nms_threshold_;
}

PicoDet::~PicoDet() { delete this->Net; }

void PicoDet::preprocess(cv::Mat &image, ncnn::Mat &in) {
  // cv::resize(image, image, cv::Size(this->in_w, this->in_h), 0.f, 0.f);
  int img_w = image.cols;
  int img_h = image.rows;
  in = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR, img_w,
                                     img_h, this->in_w, this->in_h);
  const float mean_vals[3] = {103.53f, 116.28f, 123.675f};
  const float norm_vals[3] = {0.017429f, 0.017507f, 0.017125f};
  in.substract_mean_normalize(mean_vals, norm_vals);
}

int PicoDet::detect(cv::Mat image, std::vector<BoxInfo> &result_list,
                    bool has_postprocess) {

  ncnn::Mat input;
  preprocess(image, input);
  auto ex = this->Net->create_extractor();
  ex.set_light_mode(false);
  ex.set_num_threads(4);
#if NCNN_VULKAN
  ex.set_vulkan_compute(this->hasGPU);
#endif
  ex.input("image", input); // picodet

  this->image_h = image.rows;
  this->image_w = image.cols;

  std::vector<std::vector<BoxInfo>> results;
  results.resize(this->num_class);

  if (has_postprocess) {
    ncnn::Mat dis_pred;
    ncnn::Mat cls_pred;
    ex.extract(this->nms_heads_info[0].c_str(), dis_pred);
    ex.extract(this->nms_heads_info[1].c_str(), cls_pred);
    std::cout << dis_pred.h << "  " << dis_pred.w << std::endl;
    std::cout << cls_pred.h << "  " << cls_pred.w << std::endl;
    this->nms_boxes(cls_pred, dis_pred, this->score_threshold, results);
  } else {
    for (const auto &head_info : this->non_postprocess_heads_info) {
      ncnn::Mat dis_pred;
      ncnn::Mat cls_pred;
      ex.extract(head_info.dis_layer.c_str(), dis_pred);
      ex.extract(head_info.cls_layer.c_str(), cls_pred);
      this->decode_infer(cls_pred, dis_pred, head_info.stride,
                         this->score_threshold, results);
    }
  }

  for (int i = 0; i < (int)results.size(); i++) {
    this->nms(results[i], this->nms_threshold);

    for (auto box : results[i]) {
      box.x1 = box.x1 / this->in_w * this->image_w;
      box.x2 = box.x2 / this->in_w * this->image_w;
      box.y1 = box.y1 / this->in_h * this->image_h;
      box.y2 = box.y2 / this->in_h * this->image_h;
      result_list.push_back(box);
    }
  }
  return 0;
}

void PicoDet::nms_boxes(ncnn::Mat &cls_pred, ncnn::Mat &dis_pred,
                        float score_threshold,
                        std::vector<std::vector<BoxInfo>> &result_list) {
  BoxInfo bbox;
  int i, j;
  for (i = 0; i < dis_pred.h; i++) {
    bbox.x1 = dis_pred.row(i)[0];
    bbox.y1 = dis_pred.row(i)[1];
    bbox.x2 = dis_pred.row(i)[2];
    bbox.y2 = dis_pred.row(i)[3];
    const float *scores = cls_pred.row(i);
    float score = 0;
    int cur_label = 0;
    for (int label = 0; label < this->num_class; label++) {
      float score_ = cls_pred.row(label)[i];
      if (score_ > score) {
        score = score_;
        cur_label = label;
      }
    }
    bbox.score = score;
    bbox.label = cur_label;
    result_list[cur_label].push_back(bbox);
  }
}

void PicoDet::decode_infer(ncnn::Mat &cls_pred, ncnn::Mat &dis_pred, int stride,
                           float threshold,
                           std::vector<std::vector<BoxInfo>> &results) {
  int feature_h = ceil((float)this->in_w / stride);
  int feature_w = ceil((float)this->in_h / stride);

  for (int idx = 0; idx < feature_h * feature_w; idx++) {
    const float *scores = cls_pred.row(idx);
    int row = idx / feature_w;
    int col = idx % feature_w;
    float score = 0;
    int cur_label = 0;
    for (int label = 0; label < this->num_class; label++) {
      if (scores[label] > score) {
        score = scores[label];
        cur_label = label;
      }
    }
    if (score > threshold) {
      const float *bbox_pred = dis_pred.row(idx);
      results[cur_label].push_back(
          this->disPred2Bbox(bbox_pred, cur_label, score, col, row, stride));
    }
  }
}

BoxInfo PicoDet::disPred2Bbox(const float *&dfl_det, int label, float score,
                              int x, int y, int stride) {
  float ct_x = (x + 0.5) * stride;
  float ct_y = (y + 0.5) * stride;
  std::vector<float> dis_pred;
  dis_pred.resize(4);
  for (int i = 0; i < 4; i++) {
    float dis = 0;
    float *dis_after_sm = new float[this->reg_max + 1];
    activation_function_softmax(dfl_det + i * (this->reg_max + 1), dis_after_sm,
                                this->reg_max + 1);
    for (int j = 0; j < this->reg_max + 1; j++) {
      dis += j * dis_after_sm[j];
    }
    dis *= stride;
    dis_pred[i] = dis;
    delete[] dis_after_sm;
  }
  float xmin = (std::max)(ct_x - dis_pred[0], .0f);
  float ymin = (std::max)(ct_y - dis_pred[1], .0f);
  float xmax = (std::min)(ct_x + dis_pred[2], (float)this->in_w);
  float ymax = (std::min)(ct_y + dis_pred[3], (float)this->in_w);
  return BoxInfo{xmin, ymin, xmax, ymax, score, label};
}

void PicoDet::nms(std::vector<BoxInfo> &input_boxes, float NMS_THRESH) {
  std::sort(input_boxes.begin(), input_boxes.end(),
            [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
  std::vector<float> vArea(input_boxes.size());
  for (int i = 0; i < int(input_boxes.size()); ++i) {
    vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1) *
               (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
  }
  for (int i = 0; i < int(input_boxes.size()); ++i) {
    for (int j = i + 1; j < int(input_boxes.size());) {
      float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
      float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
      float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
      float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);
      float w = (std::max)(float(0), xx2 - xx1 + 1);
      float h = (std::max)(float(0), yy2 - yy1 + 1);
      float inter = w * h;
      float ovr = inter / (vArea[i] + vArea[j] - inter);
      if (ovr >= NMS_THRESH) {
        input_boxes.erase(input_boxes.begin() + j);
        vArea.erase(vArea.begin() + j);
      } else {
        j++;
      }
    }
  }
}
