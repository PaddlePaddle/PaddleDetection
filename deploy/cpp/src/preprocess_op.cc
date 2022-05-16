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
    e /= 255.0;
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
  cv::resize(
      *im, *im, cv::Size(), resize_scale.first, resize_scale.second, interp_);

  data->in_net_shape_ = {static_cast<float>(im->rows),
                         static_cast<float>(im->cols)};
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

void LetterBoxResize::Run(cv::Mat* im, ImageBlob* data) {
  float resize_scale = GenerateScale(*im);
  int new_shape_w = std::round(im->cols * resize_scale);
  int new_shape_h = std::round(im->rows * resize_scale);
  data->im_shape_ = {static_cast<float>(new_shape_h),
                     static_cast<float>(new_shape_w)};
  float padw = (target_size_[1] - new_shape_w) / 2.;
  float padh = (target_size_[0] - new_shape_h) / 2.;

  int top = std::round(padh - 0.1);
  int bottom = std::round(padh + 0.1);
  int left = std::round(padw - 0.1);
  int right = std::round(padw + 0.1);

  cv::resize(
      *im, *im, cv::Size(new_shape_w, new_shape_h), 0, 0, cv::INTER_AREA);

  data->in_net_shape_ = {
      static_cast<float>(im->rows), static_cast<float>(im->cols),
  };
  cv::copyMakeBorder(*im,
                     *im,
                     top,
                     bottom,
                     left,
                     right,
                     cv::BORDER_CONSTANT,
                     cv::Scalar(127.5));

  data->in_net_shape_ = {
      static_cast<float>(im->rows), static_cast<float>(im->cols),
  };

  data->scale_factor_ = {
      resize_scale, resize_scale,
  };
}

float LetterBoxResize::GenerateScale(const cv::Mat& im) {
  int origin_w = im.cols;
  int origin_h = im.rows;

  int target_h = target_size_[0];
  int target_w = target_size_[1];

  float ratio_h = static_cast<float>(target_h) / static_cast<float>(origin_h);
  float ratio_w = static_cast<float>(target_w) / static_cast<float>(origin_w);
  float resize_scale = std::min(ratio_h, ratio_w);
  return resize_scale;
}

void PadStride::Run(cv::Mat* im, ImageBlob* data) {
  if (stride_ <= 0) {
    data->in_net_im_ = im->clone();
    return;
  }
  int rc = im->channels();
  int rh = im->rows;
  int rw = im->cols;
  int nh = (rh / stride_) * stride_ + (rh % stride_ != 0) * stride_;
  int nw = (rw / stride_) * stride_ + (rw % stride_ != 0) * stride_;
  cv::copyMakeBorder(
      *im, *im, 0, nh - rh, 0, nw - rw, cv::BORDER_CONSTANT, cv::Scalar(0));
  data->in_net_im_ = im->clone();
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

void GetAffineTrans(const cv::Point2f center,
                    const cv::Point2f input_size,
                    const cv::Point2f output_size,
                    cv::Mat* trans) {
  cv::Point2f srcTri[3];
  cv::Point2f dstTri[3];
  float src_w = input_size.x;
  float dst_w = output_size.x;
  float dst_h = output_size.y;

  cv::Point2f src_dir(0, -0.5 * src_w);
  cv::Point2f dst_dir(0, -0.5 * dst_w);

  srcTri[0] = center;
  srcTri[1] = center + src_dir;
  cv::Point2f src_d = srcTri[0] - srcTri[1];
  srcTri[2] = srcTri[1] + cv::Point2f(-src_d.y, src_d.x);

  dstTri[0] = cv::Point2f(dst_w * 0.5, dst_h * 0.5);
  dstTri[1] = cv::Point2f(dst_w * 0.5, dst_h * 0.5) + dst_dir;
  cv::Point2f dst_d = dstTri[0] - dstTri[1];
  dstTri[2] = dstTri[1] + cv::Point2f(-dst_d.y, dst_d.x);

  *trans = cv::getAffineTransform(srcTri, dstTri);
}

void WarpAffine::Run(cv::Mat* im, ImageBlob* data) {
  cv::cvtColor(*im, *im, cv::COLOR_RGB2BGR);
  cv::Mat trans(2, 3, CV_32FC1);
  cv::Point2f center;
  cv::Point2f input_size;
  int h = im->rows;
  int w = im->cols;
  if (keep_res_) {
    input_h_ = (h | pad_) + 1;
    input_w_ = (w + pad_) + 1;
    input_size = cv::Point2f(input_w_, input_h_);
    center = cv::Point2f(w / 2, h / 2);
  } else {
    float s = std::max(h, w) * 1.0;
    input_size = cv::Point2f(s, s);
    center = cv::Point2f(w / 2., h / 2.);
  }
  cv::Point2f output_size(input_w_, input_h_);

  GetAffineTrans(center, input_size, output_size, &trans);
  cv::warpAffine(*im, *im, trans, cv::Size(input_w_, input_h_));
  data->in_net_shape_ = {
      static_cast<float>(input_h_), static_cast<float>(input_w_),
  };
}

void Pad::Run(cv::Mat* im, ImageBlob* data) {
  int h = size_[0];
  int w = size_[1];
  int rh = im->rows;
  int rw = im->cols;
  if (h == rh && w == rw){
    data->in_net_im_ = im->clone();
    return;
  }
  cv::copyMakeBorder(
      *im, *im, 0, h - rh, 0, w - rw, cv::BORDER_CONSTANT, cv::Scalar(114));
  data->in_net_im_ = im->clone();
  data->in_net_shape_ = {
      static_cast<float>(im->rows), static_cast<float>(im->cols),
  };
}

// Preprocessor op running order
const std::vector<std::string> Preprocessor::RUN_ORDER = {"InitInfo",
                                                          "TopDownEvalAffine",
                                                          "Resize",
                                                          "LetterBoxResize",
                                                          "WarpAffine",
                                                          "NormalizeImage",
                                                          "PadStride",
                                                          "Pad",
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

  // adjust h or w to keep image ratio, expand the shorter edge
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

bool CheckDynamicInput(const std::vector<cv::Mat>& imgs) {
  if (imgs.size() == 1) return false;

  int h = imgs.at(0).rows;
  int w = imgs.at(0).cols;
  for (int i = 1; i < imgs.size(); ++i) {
    int hi = imgs.at(i).rows;
    int wi = imgs.at(i).cols;
    if (hi != h || wi != w) {
      return true;
    }
  }
  return false;
}

std::vector<cv::Mat> PadBatch(const std::vector<cv::Mat>& imgs) {
  std::vector<cv::Mat> out_imgs;
  int max_h = 0;
  int max_w = 0;
  int rh = 0;
  int rw = 0;
  // find max_h and max_w in batch
  for (int i = 0; i < imgs.size(); ++i) {
    rh = imgs.at(i).rows;
    rw = imgs.at(i).cols;
    if (rh > max_h) max_h = rh;
    if (rw > max_w) max_w = rw;
  }
  for (int i = 0; i < imgs.size(); ++i) {
    cv::Mat im = imgs.at(i);
    cv::copyMakeBorder(im,
                       im,
                       0,
                       max_h - imgs.at(i).rows,
                       0,
                       max_w - imgs.at(i).cols,
                       cv::BORDER_CONSTANT,
                       cv::Scalar(0));
    out_imgs.push_back(im);
  }
  return out_imgs;
}

}  // namespace PaddleDetection
