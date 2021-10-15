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

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "json/json.h"

namespace PaddleDetection {

// Object for storing all preprocessed data
class ImageBlob {
 public:
  // image width and height
  std::vector<float> im_shape_;
  // Buffer for image data after preprocessing
  std::vector<float> im_data_;
  // in net data shape(after pad)
  std::vector<float> in_net_shape_;
  // Evaluation image width and height
  // std::vector<float>  eval_im_size_f_;
  // Scale factor for image size to origin image size
  std::vector<float> scale_factor_;
};

// Abstraction of preprocessing opration class
class PreprocessOp {
 public:
  virtual void Init(const Json::Value& item) = 0;
  virtual void Run(cv::Mat* im, ImageBlob* data) = 0;
};

class InitInfo : public PreprocessOp {
 public:
  virtual void Init(const Json::Value& item) {}
  virtual void Run(cv::Mat* im, ImageBlob* data);
};

class NormalizeImage : public PreprocessOp {
 public:
  virtual void Init(const Json::Value& item) {
    mean_.clear();
    scale_.clear();
    for (auto tmp : item["mean"]) {
      mean_.emplace_back(tmp.as<float>());
    }
    for (auto tmp : item["std"]) {
      scale_.emplace_back(tmp.as<float>());
    }
    is_scale_ = item["is_scale"].as<bool>();
  }

  virtual void Run(cv::Mat* im, ImageBlob* data);

 private:
  // CHW or HWC
  std::vector<float> mean_;
  std::vector<float> scale_;
  bool is_scale_;
};

class Permute : public PreprocessOp {
 public:
  virtual void Init(const Json::Value& item) {}
  virtual void Run(cv::Mat* im, ImageBlob* data);
};

class Resize : public PreprocessOp {
 public:
  virtual void Init(const Json::Value& item) {
    interp_ = item["interp"].as<int>();
    // max_size_ = item["target_size"].as<int>();
    keep_ratio_ = item["keep_ratio"].as<bool>();
    target_size_.clear();
    for (auto tmp : item["target_size"]) {
      target_size_.emplace_back(tmp.as<int>());
    }
  }

  // Compute best resize scale for x-dimension, y-dimension
  std::pair<float, float> GenerateScale(const cv::Mat& im);

  virtual void Run(cv::Mat* im, ImageBlob* data);

 private:
  int interp_;
  bool keep_ratio_;
  std::vector<int> target_size_;
  std::vector<int> in_net_shape_;
};

// Models with FPN need input shape % stride == 0
class PadStride : public PreprocessOp {
 public:
  virtual void Init(const Json::Value& item) {
    stride_ = item["stride"].as<int>();
  }

  virtual void Run(cv::Mat* im, ImageBlob* data);

 private:
  int stride_;
};

class TopDownEvalAffine : public PreprocessOp {
 public:
  virtual void Init(const Json::Value& item) {
    trainsize_.clear();
    for (auto tmp : item["trainsize"]) {
      trainsize_.emplace_back(tmp.as<int>());
    }
  }

  virtual void Run(cv::Mat* im, ImageBlob* data);

 private:
  int interp_ = 1;
  std::vector<int> trainsize_;
};

void CropImg(cv::Mat& img,
             cv::Mat& crop_img,
             std::vector<int>& area,
             std::vector<float>& center,
             std::vector<float>& scale,
             float expandratio = 0.15);

class Preprocessor {
 public:
  void Init(const Json::Value& config_node) {
    // initialize image info at first
    ops_["InitInfo"] = std::make_shared<InitInfo>();
    for (const auto& item : config_node) {
      auto op_name = item["type"].as<std::string>();

      ops_[op_name] = CreateOp(op_name);
      ops_[op_name]->Init(item);
    }
  }

  std::shared_ptr<PreprocessOp> CreateOp(const std::string& name) {
    if (name == "Resize") {
      return std::make_shared<Resize>();
    } else if (name == "Permute") {
      return std::make_shared<Permute>();
    } else if (name == "NormalizeImage") {
      return std::make_shared<NormalizeImage>();
    } else if (name == "PadStride") {
      // use PadStride instead of PadBatch
      return std::make_shared<PadStride>();
    } else if (name == "TopDownEvalAffine") {
      return std::make_shared<TopDownEvalAffine>();
    }
    std::cerr << "can not find function of OP: " << name
              << " and return: nullptr" << std::endl;
    return nullptr;
  }

  void Run(cv::Mat* im, ImageBlob* data);

 public:
  static const std::vector<std::string> RUN_ORDER;

 private:
  std::unordered_map<std::string, std::shared_ptr<PreprocessOp>> ops_;
};

}  // namespace PaddleDetection
