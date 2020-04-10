//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <vector>
#include <string>

#include "include/preprocess_op.h"

namespace PaddleDetection {

void Normalize::Run(cv::Mat* im, ImageBlob* data) {
  double e = 1.0;
  if (is_scale_) {
    e /= 255.0;
  }
  (*im).convertTo(*im, CV_32FC3, e);
  for (int h = 0; h < im->rows; h++) {
    for (int w = 0; w < im->cols; w++) {
      im->at<cv::Vec3f>(h, w)[0] =
          (im->at<cv::Vec3f>(h, w)[0] - mean_[0] ) / scale_[0];
      im->at<cv::Vec3f>(h, w)[1] =
          (im->at<cv::Vec3f>(h, w)[1] - mean_[1] ) / scale_[1];
      im->at<cv::Vec3f>(h, w)[2] =
          (im->at<cv::Vec3f>(h, w)[2] - mean_[2] ) / scale_[2];
    }
  }
}

void Permute::Run(cv::Mat* im, ImageBlob* data) {
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
  data->ori_im_size_ = {
      static_cast<int>(im->rows),
      static_cast<int>(im->cols)
  };
  data->ori_im_size_f_ = {
      static_cast<float>(im->rows),
      static_cast<float>(im->cols),
      1.0
  };
  auto resize_scale = GenerateScale(*im);
  cv::resize(
      *im, *im, cv::Size(), resize_scale.first, resize_scale.second, interp_);
  if (max_size_ != 0 && !image_shape_.empty()) {
    // Padding the image with 0 border
    cv::copyMakeBorder(
      *im,
      *im,
      0,
      max_size_ - im->rows,
      0,
      max_size_ - im->cols,
      cv::BORDER_CONSTANT,
      cv::Scalar(0));
  }
  data->eval_im_size_f_ = {
    static_cast<float>(im->rows),
    static_cast<float>(im->cols),
    resize_scale.first
  };
}

std::pair<float, float> Resize::GenerateScale(const cv::Mat& im) {
  std::pair<float, float> resize_scale;
  int origin_w = im.cols;
  int origin_h = im.rows;

  if (max_size_ != 0 && (arch_ == "RCNN" || arch_ == "RetinaNet")) {
    int im_size_max = std::max(origin_w, origin_h);
    int im_size_min = std::min(origin_w, origin_h);
    float scale_ratio =
        static_cast<float>(target_size_) / static_cast<float>(im_size_min);
    if (max_size_ > 0) {
      if (round(scale_ratio * im_size_max) > max_size_) {
        scale_ratio =
            static_cast<float>(max_size_) / static_cast<float>(im_size_max);
      }
    }
    resize_scale = {scale_ratio, scale_ratio};
  } else {
    resize_scale.first =
        static_cast<float>(target_size_) / static_cast<float>(origin_w);
    resize_scale.second =
        static_cast<float>(target_size_) / static_cast<float>(origin_h);
  }
  return resize_scale;
}

void Preprocessor::Init(const YAML::Node& config_node,
                        const std::string& arch) {
  arch_ = arch;
  for (const auto& item : config_node) {
    auto op_type = item["type"].as<std::string>();
    if (op_type == "Resize") {
      int max_size = item["max_size"].as<int>();
      int target_size = item["target_size"].as<int>();
      int interp = item["interp"].as<int>();
      auto img_shape = item["image_shape"].as<std::vector<int>>();
      op_resize_.Init(arch, max_size, target_size, interp, img_shape);
    } else if (op_type == "Normalize") {
      bool is_channel_first = item["is_channel_first"].as<bool>();
      bool is_scale = item["is_scale"].as<bool>();
      auto mean = item["mean"].as<std::vector<float>>();
      auto scale = item["std"].as<std::vector<float>>();
      auto op = std::make_shared<Normalize>();
      op_normalize_.Init(is_channel_first, is_scale, mean, scale);
    } else if (op_type == "Permute") {
      bool to_bgr = item["to_bgr"].as<bool>();
      bool is_channel_first = item["channel_first"].as<bool>();
      auto op = std::make_shared<Permute>();
      op_permute_.Init(to_bgr, is_channel_first);
    }
  }
}

void Preprocessor::Run(cv::Mat* im, ImageBlob* data) {
  op_resize_.Run(im, data);
  op_normalize_.Run(im, data);
  op_permute_.Run(im, data);
}

}  // namespace PaddleDetection
