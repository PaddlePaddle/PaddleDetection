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

#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstring>
#ifdef _WIN32
#include <filesystem>
#else
#include <dirent.h>
#include <sys/types.h>
#endif

namespace PaddleSolution {
namespace utils {

enum SCALE_TYPE{
    UNPADDING,
    RANGE_SCALING
};

inline std::string path_join(const std::string& dir, const std::string& path) {
    std::string seperator = "/";
    #ifdef _WIN32
    seperator = "\\";
    #endif
    return dir + seperator + path;
}
#ifndef _WIN32
// scan a directory and get all files with input extensions
inline std::vector<std::string> get_directory_images(const std::string& path,
                                                     const std::string& exts) {
    std::vector<std::string> imgs;
    struct dirent *entry;
    DIR *dir = opendir(path.c_str());
    if (dir == NULL) {
        closedir(dir);
        return imgs;
    }
    while ((entry = readdir(dir)) != NULL) {
        std::string item = entry->d_name;
        auto ext = strrchr(entry->d_name, '.');
        if (!ext || std::string(ext) == "." || std::string(ext) == "..") {
            continue;
        }
        if (exts.find(ext) != std::string::npos) {
            imgs.push_back(path_join(path, entry->d_name));
        }
    }
    sort(imgs.begin(), imgs.end());
    return imgs;
}
#else
// scan a directory and get all files with input extensions
inline std::vector<std::string> get_directory_images(const std::string& path,
                                                     const std::string& exts) {
    std::vector<std::string> imgs;
    for (const auto& item :
                std::experimental::filesystem::directory_iterator(path)) {
        auto suffix = item.path().extension().string();
        if (exts.find(suffix) != std::string::npos && suffix.size() > 0) {
            auto fullname = path_join(path, item.path().filename().string());
            imgs.push_back(item.path().string());
        }
    }
    sort(imgs.begin(), imgs.end());
    return imgs;
}
#endif

inline int scaling(int resize_type, int &w, int &h, int new_w, int new_h,
                   int target_size, int max_size, float &im_scale_ratio) {
    if (w <= 0 || h <= 0 || new_w <= 0 || new_h <= 0) {
        return -1;
    }
    switch (resize_type) {
        case SCALE_TYPE::UNPADDING:
        {
            w = new_w;
            h = new_h;
            im_scale_ratio = 0;
        }
            break;
        case SCALE_TYPE::RANGE_SCALING:
        {
            int im_max_size = std::max(w, h);
            int im_min_size = std::min(w, h);
            float scale_ratio = static_cast<float>(target_size)
                             / static_cast<float>(im_min_size);
            if (max_size > 0) {
                if (round(scale_ratio * im_max_size) > max_size) {
                    scale_ratio = static_cast<float>(max_size)
                                / static_cast<float>(im_max_size);
                }
            }
            w = round(scale_ratio * static_cast<float>(w));
            h = round(scale_ratio * static_cast<float>(h));
            im_scale_ratio = scale_ratio;
        }
       break;
       default :
       {
            std::cout << "Can't support this type of scaling strategy."
                      << std::endl;
            std::cout << "Throw exception at file " << __FILE__
                      << " on line " << __LINE__ << std::endl;
            throw 0;
       }
       break;
    }
    return 0;
}
}  // namespace utils
}  // namespace PaddleSolution
