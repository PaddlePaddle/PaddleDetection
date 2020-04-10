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
};

// Abstraction of preprocessing opration class
class PreprocessOp {
 public:
  virtual void Run(cv::Mat* im, ImageBlob* data) = 0;
};

class Normalize : public PreprocessOp {
 public:
  void Init(bool is_channel_first,
            bool is_scale,
            const std::vector<float>& mean,
            const std::vector<float>& scale) {
    mean_ = mean;
    scale_ = scale;
    is_channel_first_ = is_channel_first;
    is_scale_ = is_scale;
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
  void Init(bool to_bgr, bool is_channel_first) {
      to_bgr_ = to_bgr;
      is_channel_first_ = is_channel_first;
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
  void Init(std::string arch,
            int max_size,
            int target_size,
            int interp,
            const std::vector<int>& image_shape) {
    arch_ = arch;
    interp_ = interp;
    max_size_ = max_size;
    target_size_ = target_size;
    image_shape_ = image_shape;
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


class Preprocessor {
 public:
  void Init(const YAML::Node& config_node, const std::string& arch);

  void Run(cv::Mat* im, ImageBlob* data);
 private:
  std::string arch_;
  Resize op_resize_;
  Permute op_permute_;
  Normalize op_normalize_;
};

}  // namespace PaddleDetection
