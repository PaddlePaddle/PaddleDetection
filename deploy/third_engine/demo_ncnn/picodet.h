// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
// reference from https://github.com/RangiLyu/nanodet/tree/main/demo_ncnn

#ifndef PICODET_H
#define PICODET_H

#include <opencv2/core/core.hpp>
#include <net.h>

typedef struct HeadInfo
{
    std::string cls_layer;
    std::string dis_layer;
    int stride;
};

typedef struct BoxInfo
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

class PicoDet
{
public:
    PicoDet(const char* param, const char* bin, bool useGPU);

    ~PicoDet();

    static PicoDet* detector;
    ncnn::Net* Net;
    static bool hasGPU;

    std::vector<HeadInfo> heads_info{
        // cls_pred|dis_pred|stride
        {"save_infer_model/scale_0.tmp_1", "save_infer_model/scale_4.tmp_1", 8},
        {"save_infer_model/scale_1.tmp_1", "save_infer_model/scale_5.tmp_1", 16},
        {"save_infer_model/scale_2.tmp_1", "save_infer_model/scale_6.tmp_1", 32},
        {"save_infer_model/scale_3.tmp_1", "save_infer_model/scale_7.tmp_1", 64},
    };

    std::vector<BoxInfo> detect(cv::Mat image, float score_threshold, float nms_threshold);

    std::vector<std::string> labels{ "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                                    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                                    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                                    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                                    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                                    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                                    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                                    "hair drier", "toothbrush" };
private:
    void preprocess(cv::Mat& image, ncnn::Mat& in);
    void decode_infer(ncnn::Mat& cls_pred, ncnn::Mat& dis_pred, int stride, float threshold, std::vector<std::vector<BoxInfo>>& results);
    BoxInfo disPred2Bbox(const float*& dfl_det, int label, float score, int x, int y, int stride);
    static void nms(std::vector<BoxInfo>& result, float nms_threshold);
    int input_size[2] = {320, 320};
    int num_class = 80;
    int reg_max = 7;

};


#endif
