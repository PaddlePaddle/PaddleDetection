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

#pragma once

#include <yaml-cpp/yaml.h>

#include <vector>
#include <string>
#include <utility>
#include <memory>
#include <unordered_map>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace PaddleDetection {

// Object for storing all preprocessed data
class ImageBlob {
 public:
  // Original image width and height
  std::vector<int> ori_im_size_;
  // Buffer for image data after preprocessing
  std::vector<float> im_data_;
  // Original image width, height, shrink in float format
  std::vector<float> ori_im_size_f_;
  // Evaluation image width and height
  std::vector<float>  eval_im_size_f_;
  // Scale factor for image size to origin image size
  std::vector<float> scale_factor_f_;
};

// Abstraction of preprocessing opration class
class PreprocessOp {
 public:
  virtual void Init(const YAML::Node& item, const std::string& arch) = 0;
  virtual void Run(cv::Mat* im, ImageBlob* data) = 0;
};

class InitInfo : public PreprocessOp{
 public:
  virtual void Init(const YAML::Node& item, const std::string& arch) {}
  virtual void Run(cv::Mat* im, ImageBlob* data);
};

class Normalize : public PreprocessOp {
 public:
  virtual void Init(const YAML::Node& item, const std::string& arch) {
    mean_ = item["mean"].as<std::vector<float>>();
    scale_ = item["std"].as<std::vector<float>>();
    is_channel_first_ = item["is_channel_first"].as<bool>();
    is_scale_ = item["is_scale"].as<bool>();
  }

  virtual void Run(cv::Mat* im, ImageBlob* data);

 private:
  // CHW or HWC
  bool is_channel_first_;
  bool is_scale_;
  std::vector<float> mean_;
  std::vector<float> scale_;
};

class Permute : public PreprocessOp {
 public:
  virtual void Init(const YAML::Node& item, const std::string& arch) {
      to_bgr_ = item["to_bgr"].as<bool>();
      is_channel_first_ = item["channel_first"].as<bool>();
  }

  virtual void Run(cv::Mat* im, ImageBlob* data);

 private:
  // RGB to BGR
  bool to_bgr_;
  // CHW or HWC
  bool is_channel_first_;
};

class Resize : public PreprocessOp {
 public:
  virtual void Init(const YAML::Node& item, const std::string& arch) {
    arch_ = arch;
    interp_ = item["interp"].as<int>();
    max_size_ = item["max_size"].as<int>();
  if (item["image_shape"].IsDefined()) {
    image_shape_ = item["image_shape"].as<std::vector<int>>();
    }
    target_size_ = item["target_size"].as<int>();
 }

  // Compute best resize scale for x-dimension, y-dimension
  std::pair<float, float> GenerateScale(const cv::Mat& im);

  virtual void Run(cv::Mat* im, ImageBlob* data);

 private:
  std::string arch_;
  int interp_;
  int max_size_;
  int target_size_;
  std::vector<int> image_shape_;
};

// Models with FPN need input shape % stride == 0
class PadStride : public PreprocessOp {
 public:
  virtual void Init(const YAML::Node& item, const std::string& arch) {
    stride_ = item["stride"].as<int>();
  }

  virtual void Run(cv::Mat* im, ImageBlob* data);

 private:
  int stride_;
};

class Preprocessor {
 public:
  void Init(const YAML::Node& config_node, const std::string& arch) {
    arch_ = arch;
    // initialize image info at first
    ops_["InitInfo"] = std::make_shared<InitInfo>();
    for (const auto& item : config_node) {
      auto op_name = item["type"].as<std::string>();
      ops_[op_name] = CreateOp(op_name);
      ops_[op_name]->Init(item, arch);
    }
  }

  std::shared_ptr<PreprocessOp> CreateOp(const std::string& name) {
    if (name == "Resize") {
      return std::make_shared<Resize>();
    } else if (name == "Permute") {
      return std::make_shared<Permute>();
    } else if (name == "Normalize") {
      return std::make_shared<Normalize>();
    } else if (name == "PadStride") {
      return std::make_shared<PadStride>();
    }
    return nullptr;
  }

  void Run(cv::Mat* im, ImageBlob* data);

 public:
  static const std::vector<std::string> RUN_ORDER;

 private:
  std::string arch_;
  std::unordered_map<std::string, std::shared_ptr<PreprocessOp>> ops_;
};

}  // namespace PaddleDetection

