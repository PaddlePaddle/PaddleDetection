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

#include <glog/logging.h>
#include <yaml-cpp/yaml.h>

#include <vector>
#include <string>
#include <utility>
#include <memory>
#include <unordered_map>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace PaddleDetection {

// Object for storing all preprocessed data
class ImageBlob {
 public:
  // image width and height
  std::vector<float> im_shape_;
  // Buffer for image data after preprocessing
  std::vector<float> im_data_;
  // in net data shape(after pad)
  std::vector<int> in_net_shape_;
  // Evaluation image width and height
  //std::vector<float>  eval_im_size_f_;
  // Scale factor for image size to origin image size
  std::vector<float> scale_factor_;
};

// Abstraction of preprocessing opration class
class PreprocessOp {
 public:
  virtual void Init(const YAML::Node& item, const std::vector<int> image_shape) = 0;
  virtual void Run(cv::Mat* im, ImageBlob* data) = 0;
};

class InitInfo : public PreprocessOp{
 public:
  virtual void Init(const YAML::Node& item, const std::vector<int> image_shape) {}
  virtual void Run(cv::Mat* im, ImageBlob* data);
};

class NormalizeImage : public PreprocessOp {
 public:
  virtual void Init(const YAML::Node& item, const std::vector<int> image_shape) {
    mean_ = item["mean"].as<std::vector<float>>();
    scale_ = item["std"].as<std::vector<float>>();
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
  virtual void Init(const YAML::Node& item, const std::vector<int> image_shape) {}
  virtual void Run(cv::Mat* im, ImageBlob* data);

};

class Resize : public PreprocessOp {
 public:
  virtual void Init(const YAML::Node& item, const std::vector<int> image_shape) {
    interp_ = item["interp"].as<int>();
    //max_size_ = item["target_size"].as<int>();
    keep_ratio_ = item["keep_ratio"].as<bool>();
    target_size_ = item["target_size"].as<std::vector<int>>();
    if (item["keep_ratio"]) {
      in_net_shape_ = image_shape;
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
  virtual void Init(const YAML::Node& item, const std::vector<int> image_shape) {
    stride_ = item["stride"].as<int>();
  }

  virtual void Run(cv::Mat* im, ImageBlob* data);

 private:
  int stride_;
};

class Preprocessor {
 public:
  void Init(const YAML::Node& config_node, const std::vector<int> image_shape) {
    // initialize image info at first
    ops_["InitInfo"] = std::make_shared<InitInfo>();
    for (const auto& item : config_node) {
      auto op_name = item["type"].as<std::string>();

      ops_[op_name] = CreateOp(op_name);
      ops_[op_name]->Init(item, image_shape);
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
    }
    std::cerr << "can not find function of OP: " << name << " and return: nullptr" << std::endl;
    return nullptr;
  }

  void Run(cv::Mat* im, ImageBlob* data);

 public:
  static const std::vector<std::string> RUN_ORDER;

 private:
  std::unordered_map<std::string, std::shared_ptr<PreprocessOp>> ops_;
};

}  // namespace PaddleDetection

