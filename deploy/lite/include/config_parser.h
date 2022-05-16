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
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "json/json.h"

#ifdef _WIN32
#define OS_PATH_SEP "\\"
#else
#define OS_PATH_SEP "/"
#endif

namespace PaddleDetection {

void load_jsonf(std::string jsonfile, Json::Value& jsondata);

// Inference model configuration parser
class ConfigPaser {
 public:
  ConfigPaser() {}

  ~ConfigPaser() {}

  bool load_config(const std::string& model_dir,
                   const std::string& cfg = "infer_cfg") {
    Json::Value config;
    load_jsonf(model_dir + OS_PATH_SEP + cfg + ".json", config);

    // Get model arch : YOLO, SSD, RetinaNet, RCNN, Face, PicoDet, HRNet
    if (config.isMember("arch")) {
      arch_ = config["arch"].as<std::string>();
    } else {
      std::cerr
          << "Please set model arch,"
          << "support value : YOLO, SSD, RetinaNet, RCNN, Face, PicoDet, HRNet."
          << std::endl;
      return false;
    }

    // Get draw_threshold for visualization
    if (config.isMember("draw_threshold")) {
      draw_threshold_ = config["draw_threshold"].as<float>();
    } else {
      std::cerr << "Please set draw_threshold." << std::endl;
      return false;
    }
    // Get Preprocess for preprocessing
    if (config.isMember("Preprocess")) {
      preprocess_info_ = config["Preprocess"];
    } else {
      std::cerr << "Please set Preprocess." << std::endl;
      return false;
    }
    // Get label_list for visualization
    if (config.isMember("label_list")) {
      label_list_.clear();
      for (auto item : config["label_list"]) {
        label_list_.emplace_back(item.as<std::string>());
      }
    } else {
      std::cerr << "Please set label_list." << std::endl;
      return false;
    }

    // Get NMS for postprocess
    if (config.isMember("NMS")) {
      nms_info_ = config["NMS"];
    }
    // Get fpn_stride in PicoDet
    if (config.isMember("fpn_stride")) {
      fpn_stride_.clear();
      for (auto item : config["fpn_stride"]) {
        fpn_stride_.emplace_back(item.as<int>());
      }
    }

    return true;
  }
  float draw_threshold_;
  std::string arch_;
  Json::Value preprocess_info_;
  Json::Value nms_info_;
  std::vector<std::string> label_list_;
  std::vector<int> fpn_stride_;
};

}  // namespace PaddleDetection
