// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <glog/logging.h>

#include "preprocessor.h"
#include "preprocessor_detection.h"
#include <iostream>

namespace PaddleSolution {

    std::shared_ptr<ImagePreProcessor> create_processor(const std::string& conf_file) {
        auto config = std::make_shared<PaddleSolution::PaddleModelConfigPaser>();
        if (!config->load_config(conf_file)) {
        #ifdef _WIN32
            std::cerr << "fail to load conf file [" << conf_file << "]" << std::endl;
        #else
            LOG(FATAL) << "fail to load conf file [" << conf_file << "]";
        #endif
            return nullptr;
        }

        if (config->_pre_processor == "DetectionPreProcessor") {
            auto p = std::make_shared<DetectionPreProcessor>();
            if (!p->init(config)) {
                return nullptr;
            }
            return p;
        }
        #ifdef _WIN32
        std::cerr << "unknown processor_name [" << config->_pre_processor << "],"
                  << "please check whether PRE_PROCESSOR is set correctly" << std::endl;
        #else
        LOG(FATAL) << "unknown processor_name [" << config->_pre_processor << "],"
                  << "please check whether PRE_PROCESSOR is set correctly";
        #endif
        return nullptr;
    }
}  // namespace PaddleSolution
