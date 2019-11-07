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
#include <utils/utils.h>
#include <predictor/detection_predictor.h>

DEFINE_string(conf, "", "Configuration File Path");
DEFINE_string(input_dir, "", "Directory of Input Images");

int main(int argc, char** argv) {
    // 0. parse args
    google::ParseCommandLineFlags(&argc, &argv, true);
    if (FLAGS_conf.empty() || FLAGS_input_dir.empty()) {
        std::cout << "Usage: ./predictor --conf=/config/path/to/your/model "
                  << "--input_dir=/directory/of/your/input/images" << std::endl;
        return -1;
    }
    // 1. create a predictor and init it with conf
    PaddleSolution::DetectionPredictor predictor;
    if (predictor.init(FLAGS_conf) != 0) {
     #ifdef _WIN32
        std::cerr << "Fail to init predictor" << std::endl;
     #else
        LOG(FATAL) << "Fail to init predictor";
     #endif
        return -1;
    }

    // 2. get all the images with extension '.jpeg' at input_dir
    auto imgs = PaddleSolution::utils::get_directory_images(FLAGS_input_dir,
                                ".jpeg|.jpg|.JPEG|.JPG|.bmp|.BMP|.png|.PNG");

    // 3. predict
    predictor.predict(imgs);
    return 0;
}
