// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <vector>
#include <string>
#include <map>

namespace PaddleSolution {

class PaddleModelConfigPaser {
    std::map<std::string, int> _scaling_map;

 public:
    PaddleModelConfigPaser()
        :_class_num(0),
        _channels(0),
        _use_gpu(0),
        _batch_size(1),
        _target_short_size(0),
        _model_file_name("__model__"),
        _param_file_name("__params__"),
        _scaling_map{{"UNPADDING", 0},
                     {"RANGE_SCALING", 1}},
        _feeds_size(1),
    _coarsest_stride(1) {}

    ~PaddleModelConfigPaser() {}

    void reset() {
        _crop_size.clear();
        _resize.clear();
        _mean.clear();
        _std.clear();
        _img_type.clear();
        _class_num = 0;
        _channels = 0;
        _use_gpu = 0;
        _target_short_size = 0;
        _batch_size = 1;
        _model_file_name = "__model__";
        _model_path = "./";
        _param_file_name = "__params__";
        _resize_type = 0;
        _resize_max_size = 0;
        _feeds_size = 1;
         _coarsest_stride = 1;
    }

    std::string process_parenthesis(const std::string& str) {
        if (str.size() < 2) {
            return str;
        }
        std::string nstr(str);
        if (str[0] == '(' && str.back() == ')') {
            nstr[0] = '[';
            nstr[str.size() - 1] = ']';
        }
        return nstr;
    }

    template <typename T>
    std::vector<T> parse_str_to_vec(const std::string& str) {
        std::vector<T> data;
        auto node = YAML::Load(str);
        for (const auto& item : node) {
            data.push_back(item.as<T>());
        }
        return data;
    }

    bool load_config(const std::string& conf_file) {
        reset();
        YAML::Node config;
        try {
            config = YAML::LoadFile(conf_file);
        } catch(...) {
            return false;
        }
        // 1. get resize
        if (config["DEPLOY"]["EVAL_CROP_SIZE"].IsDefined()) {
            auto str = config["DEPLOY"]["EVAL_CROP_SIZE"].as<std::string>();
            _resize = parse_str_to_vec<int>(process_parenthesis(str));
        } else {
            std::cerr << "Please set EVAL_CROP_SIZE: (xx, xx)" << std::endl;
            return false;
        }
        // 0. get crop_size
        if (config["DEPLOY"]["CROP_SIZE"].IsDefined()) {
            auto crop_str = config["DEPLOY"]["CROP_SIZE"].as<std::string>();
             _crop_size = parse_str_to_vec<int>(process_parenthesis(crop_str));
        } else {
            _crop_size = _resize;
        }

        // 2. get mean
        if (config["DEPLOY"]["MEAN"].IsDefined()) {
            for (const auto& item : config["DEPLOY"]["MEAN"]) {
                _mean.push_back(item.as<float>());
            }
        } else {
            std::cerr << "Please set MEAN: [xx, xx, xx]" << std::endl;
            return false;
        }
        // 3. get std
        if(config["DEPLOY"]["STD"].IsDefined()) {
            for (const auto& item : config["DEPLOY"]["STD"]) {
                _std.push_back(item.as<float>());
            }
        } else {
            std::cerr << "Please set STD: [xx, xx, xx]" << std::endl;
            return false;
        }
        // 4. get image type
        if (config["DEPLOY"]["IMAGE_TYPE"].IsDefined()) {
            _img_type = config["DEPLOY"]["IMAGE_TYPE"].as<std::string>();
        } else {
            std::cerr << "Please set IMAGE_TYPE: \"rgb\" or \"rgba\"" << std::endl;
            return false;
        }
        // 5. get class number
        if (config["DEPLOY"]["NUM_CLASSES"].IsDefined()) {
            _class_num = config["DEPLOY"]["NUM_CLASSES"].as<int>();
        } else {
            std::cerr << "Please set NUM_CLASSES: x" << std::endl;
            return false;
        }
        // 7. set model path
        if (config["DEPLOY"]["MODEL_PATH"].IsDefined()) {
            _model_path = config["DEPLOY"]["MODEL_PATH"].as<std::string>();
        } else {
            std::cerr << "Please set MODEL_PATH: \"/path/to/model_dir\"" << std::endl;
            return false;
        }
        // 8. get model file_name
        if (config["DEPLOY"]["MODEL_FILENAME"].IsDefined()) {
            _model_file_name = config["DEPLOY"]["MODEL_FILENAME"].as<std::string>();
        } else {
            _model_file_name = "__model__";
        }
        // 9. get model param file name
        if (config["DEPLOY"]["PARAMS_FILENAME"].IsDefined()) {
            _param_file_name
                = config["DEPLOY"]["PARAMS_FILENAME"].as<std::string>();
        } else {
            _param_file_name = "__params__";
        }
        // 10. get pre_processor
        if (config["DEPLOY"]["PRE_PROCESSOR"].IsDefined()) {
            _pre_processor = config["DEPLOY"]["PRE_PROCESSOR"].as<std::string>();
        } else {
            std::cerr << "Please set PRE_PROCESSOR: \"DetectionPreProcessor\"" << std::endl;
            return false;
        }
        // 11. use_gpu
        if (config["DEPLOY"]["USE_GPU"].IsDefined()) { 
            _use_gpu = config["DEPLOY"]["USE_GPU"].as<int>();
        } else {
            _use_gpu = 0;
        }
        // 12. predictor_mode
        if (config["DEPLOY"]["PREDICTOR_MODE"].IsDefined()) {
            _predictor_mode = config["DEPLOY"]["PREDICTOR_MODE"].as<std::string>();
        } else {
            std::cerr << "Please set PREDICTOR_MODE: \"NATIVE\" or \"ANALYSIS\""  << std::endl;
            return false;
        }
        // 13. batch_size
        if (config["DEPLOY"]["BATCH_SIZE"].IsDefined()) {
            _batch_size = config["DEPLOY"]["BATCH_SIZE"].as<int>();
        } else {
            _batch_size = 1;
        }
        // 14. channels
        if (config["DEPLOY"]["CHANNELS"].IsDefined()) {
            _channels = config["DEPLOY"]["CHANNELS"].as<int>();
        } else {
            std::cerr << "Please set CHANNELS: x"  << std::endl;
            return false;
        }
        // 15. target_short_size
        if (config["DEPLOY"]["TARGET_SHORT_SIZE"].IsDefined()) {
           _target_short_size = config["DEPLOY"]["TARGET_SHORT_SIZE"].as<int>();
        }
        // 16.resize_type
        if (config["DEPLOY"]["RESIZE_TYPE"].IsDefined() &&
            _scaling_map.find(config["DEPLOY"]["RESIZE_TYPE"].as<std::string>()) != _scaling_map.end()) {
            _resize_type = _scaling_map[config["DEPLOY"]["RESIZE_TYPE"].as<std::string>()];
        } else {
            _resize_type = 0;
        }
        // 17.resize_max_size
        if (config["DEPLOY"]["RESIZE_MAX_SIZE"].IsDefined()) {
            _resize_max_size = config["DEPLOY"]["RESIZE_MAX_SIZE"].as<int>();
        }
        // 18.feeds_size
        if (config["DEPLOY"]["FEEDS_SIZE"].IsDefined()) {
            _feeds_size = config["DEPLOY"]["FEEDS_SIZE"].as<int>();
        }
        // 19. coarsest_stride
        if (config["DEPLOY"]["COARSEST_STRIDE"].IsDefined()) {
            _coarsest_stride = config["DEPLOY"]["COARSEST_STRIDE"].as<int>();
        }
        return true;
    }

    void debug() const {
        std::cout << "SCALE_RESIZE: (" << _resize[0] << ", "
                  << _resize[1] << ")" << std::endl;

        std::cout << "MEAN: [";
        for (int i = 0; i < _mean.size(); ++i) {
            if (i != _mean.size() - 1) {
                std::cout << _mean[i] << ", ";
            } else {
                std::cout << _mean[i];
            }
        }
        std::cout << "]" << std::endl;

        std::cout << "STD: [";
        for (int i = 0; i < _std.size(); ++i) {
            if (i != _std.size() - 1) {
                std::cout << _std[i] << ", ";
            } else {
                std::cout << _std[i];
            }
        }
        std::cout << "]" << std::endl;
        std::cout << "DEPLOY.TARGET_SHORT_SIZE: " << _target_short_size
                  << std::endl;
        std::cout << "DEPLOY.IMAGE_TYPE: " << _img_type << std::endl;
        std::cout << "DEPLOY.NUM_CLASSES: " << _class_num << std::endl;
        std::cout << "DEPLOY.CHANNELS: " << _channels << std::endl;
        std::cout << "DEPLOY.MODEL_PATH: " << _model_path << std::endl;
        std::cout << "DEPLOY.MODEL_FILENAME: " << _model_file_name
                  << std::endl;
        std::cout << "DEPLOY.PARAMS_FILENAME: " << _param_file_name
                  << std::endl;
        std::cout << "DEPLOY.PRE_PROCESSOR: " << _pre_processor << std::endl;
        std::cout << "DEPLOY.USE_GPU: " << _use_gpu << std::endl;
        std::cout << "DEPLOY.PREDICTOR_MODE: " << _predictor_mode << std::endl;
        std::cout << "DEPLOY.BATCH_SIZE: " << _batch_size << std::endl;
    }
    // DEPLOY.COARSEST_STRIDE
    int _coarsest_stride;
    // DEPLOY.FEEDS_SIZE
    int _feeds_size;
    // DEPLOY.RESIZE_TYPE  0:unpadding 1:rangescaling  Default:0
    int _resize_type;
    // DEPLOY.RESIZE_MAX_SIZE
    int _resize_max_size;
    // DEPLOY.CROP_SIZE
    std::vector<int> _crop_size;
    // DEPLOY.SCALE_RESIZE
    std::vector<int> _resize;
    // DEPLOY.MEAN
    std::vector<float> _mean;
    // DEPLOY.STD
    std::vector<float> _std;
    // DEPLOY.IMAGE_TYPE
    std::string _img_type;
    // DEPLOY.TARGET_SHORT_SIZE
    int _target_short_size;
    // DEPLOY.NUM_CLASSES
    int _class_num;
    // DEPLOY.CHANNELS
    int _channels;
    // DEPLOY.MODEL_PATH
    std::string _model_path;
    // DEPLOY.MODEL_FILENAME
    std::string _model_file_name;
    // DEPLOY.PARAMS_FILENAME
    std::string _param_file_name;
    // DEPLOY.PRE_PROCESSOR
    std::string _pre_processor;
    // DEPLOY.USE_GPU
    int _use_gpu;
    // DEPLOY.PREDICTOR_MODE
    std::string _predictor_mode;
    // DEPLOY.BATCH_SIZE
    int _batch_size;
};
}  // namespace PaddleSolution
