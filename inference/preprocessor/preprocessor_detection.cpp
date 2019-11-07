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

#include <glog/logging.h>
#include <thread>
#include <mutex>

#include "preprocessor_detection.h"
#include "utils/utils.h"

namespace PaddleSolution {
bool DetectionPreProcessor::single_process(const std::string& fname,
                                           std::vector<float> &vec_data,
                                           int* ori_w, int* ori_h,
                                           int* resize_w, int* resize_h,
                                           float* scale_ratio) {
    cv::Mat im1 = cv::imread(fname, -1);
    cv::Mat im;
    if (_config->_feeds_size == 3) {  // faster rcnn
        im1.convertTo(im, CV_32FC3, 1/255.0);
    } else if (_config->_feeds_size == 2) {  // yolo v3
        im = im1;
    }
    if (im.data == nullptr || im.empty()) {
    #ifdef _WIN32
        std::cerr << "Failed to open image: " << fname << std::endl;
    #else
        LOG(ERROR) << "Failed to open image: " << fname;
    #endif
        return false;
    }
    int channels = im.channels();
    if (channels == 1) {
        cv::cvtColor(im, im, cv::COLOR_GRAY2BGR);
    }
    channels = im.channels();
    if (channels != 3 && channels != 4) {
    #ifdef _WIN32
        std::cerr << "Only support rgb(gray) and rgba image." << std::endl;
    #else
        LOG(ERROR) << "Only support rgb(gray) and rgba image.";
    #endif 
        return false;
    }
    *ori_w = im.cols;
    *ori_h = im.rows;
    cv::cvtColor(im, im, cv::COLOR_BGR2RGB);
    // channels = im.channels();

    // resize
    int rw = im.cols;
    int rh = im.rows;
    float im_scale_ratio;
    utils::scaling(_config->_resize_type, rw, rh, _config->_resize[0],
                   _config->_resize[1], _config->_target_short_size,
                   _config->_resize_max_size, im_scale_ratio);
    cv::Size resize_size(rw, rh);
    *resize_w = rw;
    *resize_h = rh;
    *scale_ratio = im_scale_ratio;
    if (*ori_h != rh || *ori_w != rw) {
        cv::Mat im_temp;
        if (_config->_resize_type == utils::SCALE_TYPE::UNPADDING) {
            cv::resize(im, im_temp, resize_size, 0, 0, cv::INTER_LINEAR);
        } else if (_config->_resize_type == utils::SCALE_TYPE::RANGE_SCALING) {
                cv::resize(im, im_temp, cv::Size(), im_scale_ratio,
                           im_scale_ratio, cv::INTER_LINEAR);
        }
        im = im_temp;
    }

    vec_data.resize(channels * rw * rh);
    float *data = vec_data.data();

    float* pmean = _config->_mean.data();
    float* pscale = _config->_std.data();
    for (int h = 0; h < rh; ++h) {
        const uchar* uptr = im.ptr<uchar>(h);
        const float* fptr = im.ptr<float>(h);
        int im_index = 0;
        for (int w = 0; w < rw; ++w) {
            for (int c = 0; c < channels; ++c) {
                int top_index = (c * rh + h) * rw + w;
                float pixel;
                if (_config->_feeds_size == 2) {  // yolo v3
                    pixel = static_cast<float>(uptr[im_index++]) / 255.0;
                } else if (_config->_feeds_size == 3) {
                    pixel = fptr[im_index++];
                }
                pixel = (pixel - pmean[c]) / pscale[c];
                data[top_index] = pixel;
            }
        }
    }
    return true;
}

bool DetectionPreProcessor::batch_process(const std::vector<std::string>& imgs,
                                          std::vector<std::vector<float>> &data,
                                          int* ori_w, int* ori_h, int* resize_w,
                                          int* resize_h, float* scale_ratio) {
    auto ic = _config->_channels;
    auto iw = _config->_resize[0];
    auto ih = _config->_resize[1];
    std::vector<std::thread> threads;
    for (int i = 0; i < imgs.size(); ++i) {
        std::string path = imgs[i];
        int* width = &ori_w[i];
        int* height = &ori_h[i];
        int* resize_width = &resize_w[i];
        int* resize_height = &resize_h[i];
        float* sr = &scale_ratio[i];
        threads.emplace_back([this, &data, i, path, width, height,
                              resize_width, resize_height, sr] {
            std::vector<float> buffer;
            single_process(path, buffer, width, height, resize_width,
                           resize_height, sr);
            data[i] = buffer;
        });
    }
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    return true;
}

bool DetectionPreProcessor::init(std::shared_ptr<PaddleSolution::PaddleModelConfigPaser> config) {
    _config = config;
    return true;
}
}  //  namespace PaddleSolution
