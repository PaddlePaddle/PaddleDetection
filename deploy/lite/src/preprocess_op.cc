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

#include <string>
#include <thread>
#include <vector>

#include "include/preprocess_op.h"

namespace PaddleDetection {

void InitInfo::Run(cv::Mat* im, ImageBlob* data) {
  data->im_shape_ = {static_cast<float>(im->rows),
                     static_cast<float>(im->cols)};
  data->scale_factor_ = {1., 1.};
  data->in_net_shape_ = {static_cast<float>(im->rows),
                         static_cast<float>(im->cols)};
}

void NormalizeImage::Run(cv::Mat* im, ImageBlob* data) {
  double e = 1.0;
  if (is_scale_) {
    e *= 1./255.0;
  }
  (*im).convertTo(*im, CV_32FC3, e);
  for (int h = 0; h < im->rows; h++) {
    for (int w = 0; w < im->cols; w++) {
      im->at<cv::Vec3f>(h, w)[0] =
          (im->at<cv::Vec3f>(h, w)[0] - mean_[0]) / scale_[0];
      im->at<cv::Vec3f>(h, w)[1] =
          (im->at<cv::Vec3f>(h, w)[1] - mean_[1]) / scale_[1];
      im->at<cv::Vec3f>(h, w)[2] =
          (im->at<cv::Vec3f>(h, w)[2] - mean_[2]) / scale_[2];
    }
  }
}

void Permute::Run(cv::Mat* im, ImageBlob* data) {
  (*im).convertTo(*im, CV_32FC3);
  int rh = im->rows;
  int rw = im->cols;
  int rc = im->channels();
  (data->im_data_).resize(rc * rh * rw);
  float* base = (data->im_data_).data();
  for (int i = 0; i < rc; ++i) {
    cv::extractChannel(*im, cv::Mat(rh, rw, CV_32FC1, base + i * rh * rw), i);
  }
}

void Resize::Run(cv::Mat* im, ImageBlob* data) {
  auto resize_scale = GenerateScale(*im);
  data->im_shape_ = {static_cast<float>(im->cols * resize_scale.first),
                     static_cast<float>(im->rows * resize_scale.second)};
  data->in_net_shape_ = {static_cast<float>(im->cols * resize_scale.first),
                         static_cast<float>(im->rows * resize_scale.second)};
  cv::resize(
      *im, *im, cv::Size(), resize_scale.first, resize_scale.second, interp_);
  data->im_shape_ = {
      static_cast<float>(im->rows), static_cast<float>(im->cols),
  };
  data->scale_factor_ = {
      resize_scale.second, resize_scale.first,
  };
}

std::pair<float, float> Resize::GenerateScale(const cv::Mat& im) {
  std::pair<float, float> resize_scale;
  int origin_w = im.cols;
  int origin_h = im.rows;

  if (keep_ratio_) {
    int im_size_max = std::max(origin_w, origin_h);
    int im_size_min = std::min(origin_w, origin_h);
    int target_size_max =
        *std::max_element(target_size_.begin(), target_size_.end());
    int target_size_min =
        *std::min_element(target_size_.begin(), target_size_.end());
    float scale_min =
        static_cast<float>(target_size_min) / static_cast<float>(im_size_min);
    float scale_max =
        static_cast<float>(target_size_max) / static_cast<float>(im_size_max);
    float scale_ratio = std::min(scale_min, scale_max);
    resize_scale = {scale_ratio, scale_ratio};
  } else {
    resize_scale.first =
        static_cast<float>(target_size_[1]) / static_cast<float>(origin_w);
    resize_scale.second =
        static_cast<float>(target_size_[0]) / static_cast<float>(origin_h);
  }
  return resize_scale;
}

void PadStride::Run(cv::Mat* im, ImageBlob* data) {
  if (stride_ <= 0) {
    return;
  }
  int rc = im->channels();
  int rh = im->rows;
  int rw = im->cols;
  int nh = (rh / stride_) * stride_ + (rh % stride_ != 0) * stride_;
  int nw = (rw / stride_) * stride_ + (rw % stride_ != 0) * stride_;
  cv::copyMakeBorder(
      *im, *im, 0, nh - rh, 0, nw - rw, cv::BORDER_CONSTANT, cv::Scalar(0));
  data->in_net_shape_ = {
      static_cast<float>(im->rows), static_cast<float>(im->cols),
  };
}

void TopDownEvalAffine::Run(cv::Mat* im, ImageBlob* data) {
  cv::resize(*im, *im, cv::Size(trainsize_[0], trainsize_[1]), 0, 0, interp_);
  // todo: Simd::ResizeBilinear();
  data->in_net_shape_ = {
      static_cast<float>(trainsize_[1]), static_cast<float>(trainsize_[0]),
  };
}

// Preprocessor op running order
const std::vector<std::string> Preprocessor::RUN_ORDER = {"InitInfo",
                                                          "TopDownEvalAffine",
                                                          "Resize",
                                                          "NormalizeImage",
                                                          "PadStride",
                                                          "Permute"};

void Preprocessor::Run(cv::Mat* im, ImageBlob* data) {
  for (const auto& name : RUN_ORDER) {
    if (ops_.find(name) != ops_.end()) {
      ops_[name]->Run(im, data);
    }
  }
}

void CropImg(cv::Mat& img,
             cv::Mat& crop_img,
             std::vector<int>& area,
             std::vector<float>& center,
             std::vector<float>& scale,
             float expandratio) {
  int crop_x1 = std::max(0, area[0]);
  int crop_y1 = std::max(0, area[1]);
  int crop_x2 = std::min(img.cols - 1, area[2]);
  int crop_y2 = std::min(img.rows - 1, area[3]);
  
  int center_x = (crop_x1 + crop_x2) / 2.;
  int center_y = (crop_y1 + crop_y2) / 2.;
  int half_h = (crop_y2 - crop_y1) / 2.;
  int half_w = (crop_x2 - crop_x1) / 2.;

  if (half_h * 3 > half_w * 4) {
    half_w = static_cast<int>(half_h * 0.75);
  } else {
    half_h = static_cast<int>(half_w * 4 / 3);
  }

  crop_x1 =
      std::max(0, center_x - static_cast<int>(half_w * (1 + expandratio)));
  crop_y1 =
      std::max(0, center_y - static_cast<int>(half_h * (1 + expandratio)));
  crop_x2 = std::min(img.cols - 1,
                     static_cast<int>(center_x + half_w * (1 + expandratio)));
  crop_y2 = std::min(img.rows - 1,
                     static_cast<int>(center_y + half_h * (1 + expandratio)));
  crop_img =
      img(cv::Range(crop_y1, crop_y2 + 1), cv::Range(crop_x1, crop_x2 + 1));

  center.clear();
  center.emplace_back((crop_x1 + crop_x2) / 2);
  center.emplace_back((crop_y1 + crop_y2) / 2);
  scale.clear();
  scale.emplace_back((crop_x2 - crop_x1));
  scale.emplace_back((crop_y2 - crop_y1));
}

}  // namespace PaddleDetection
