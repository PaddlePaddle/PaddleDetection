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

#include "picodet.h"
#include <benchmark.h>
#include <iostream>
#include <net.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define __SAVE_RESULT__ // if defined save drawed results to ../results, else
                        // show it in windows
struct object_rect {
  int x;
  int y;
  int width;
  int height;
};

std::vector<int> GenerateColorMap(int num_class) {
  auto colormap = std::vector<int>(3 * num_class, 0);
  for (int i = 0; i < num_class; ++i) {
    int j = 0;
    int lab = i;
    while (lab) {
      colormap[i * 3] |= (((lab >> 0) & 1) << (7 - j));
      colormap[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j));
      colormap[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j));
      ++j;
      lab >>= 3;
    }
  }
  return colormap;
}

void draw_bboxes(const cv::Mat &im, const std::vector<BoxInfo> &bboxes,
                 std::string save_path = "None") {
  static const char *class_names[] = {
      "person",        "bicycle",      "car",
      "motorcycle",    "airplane",     "bus",
      "train",         "truck",        "boat",
      "traffic light", "fire hydrant", "stop sign",
      "parking meter", "bench",        "bird",
      "cat",           "dog",          "horse",
      "sheep",         "cow",          "elephant",
      "bear",          "zebra",        "giraffe",
      "backpack",      "umbrella",     "handbag",
      "tie",           "suitcase",     "frisbee",
      "skis",          "snowboard",    "sports ball",
      "kite",          "baseball bat", "baseball glove",
      "skateboard",    "surfboard",    "tennis racket",
      "bottle",        "wine glass",   "cup",
      "fork",          "knife",        "spoon",
      "bowl",          "banana",       "apple",
      "sandwich",      "orange",       "broccoli",
      "carrot",        "hot dog",      "pizza",
      "donut",         "cake",         "chair",
      "couch",         "potted plant", "bed",
      "dining table",  "toilet",       "tv",
      "laptop",        "mouse",        "remote",
      "keyboard",      "cell phone",   "microwave",
      "oven",          "toaster",      "sink",
      "refrigerator",  "book",         "clock",
      "vase",          "scissors",     "teddy bear",
      "hair drier",    "toothbrush"};

  cv::Mat image = im.clone();
  int src_w = image.cols;
  int src_h = image.rows;
  int thickness = 2;
  auto colormap = GenerateColorMap(sizeof(class_names));

  for (size_t i = 0; i < bboxes.size(); i++) {
    const BoxInfo &bbox = bboxes[i];
    std::cout << bbox.x1 << ". " << bbox.y1 << ". " << bbox.x2 << ". "
              << bbox.y2 << ". " << std::endl;
    int c1 = colormap[3 * bbox.label + 0];
    int c2 = colormap[3 * bbox.label + 1];
    int c3 = colormap[3 * bbox.label + 2];
    cv::Scalar color = cv::Scalar(c1, c2, c3);
    // cv::Scalar color = cv::Scalar(0, 0, 255);
    cv::rectangle(image, cv::Rect(cv::Point(bbox.x1, bbox.y1),
                                  cv::Point(bbox.x2, bbox.y2)),
                  color, 1);

    char text[256];
    sprintf(text, "%s %.1f%%", class_names[bbox.label], bbox.score * 100);

    int baseLine = 0;
    cv::Size label_size =
        cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

    int x = bbox.x1;
    int y = bbox.y1 - label_size.height - baseLine;
    if (y < 0)
      y = 0;
    if (x + label_size.width > image.cols)
      x = image.cols - label_size.width;

    cv::rectangle(image, cv::Rect(cv::Point(x, y),
                                  cv::Size(label_size.width,
                                           label_size.height + baseLine)),
                  color, -1);

    cv::putText(image, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
  }

  if (save_path == "None") {
    cv::imshow("image", image);
  } else {
    cv::imwrite(save_path, image);
    std::cout << "Result save in: " << save_path << std::endl;
  }
}

int image_demo(PicoDet &detector, const char *imagepath,
               int has_postprocess = 0) {
  std::vector<cv::String> filenames;
  cv::glob(imagepath, filenames, false);
  bool is_postprocess = has_postprocess > 0 ? true : false;
  for (auto img_name : filenames) {
    cv::Mat image = cv::imread(img_name, cv::IMREAD_COLOR);
    if (image.empty()) {
      fprintf(stderr, "cv::imread %s failed\n", img_name.c_str());
      return -1;
    }
    std::vector<BoxInfo> results;
    detector.detect(image, results, is_postprocess);
    std::cout << "detect done." << std::endl;

#ifdef __SAVE_RESULT__
    std::string save_path = img_name;
    draw_bboxes(image, results, save_path.replace(3, 4, "results"));
#else
    draw_bboxes(image, results);
    cv::waitKey(0);
#endif
  }
  return 0;
}

int benchmark(PicoDet &detector, int width, int height,
              int has_postprocess = 0) {
  int loop_num = 100;
  int warm_up = 8;

  double time_min = DBL_MAX;
  double time_max = -DBL_MAX;
  double time_avg = 0;
  cv::Mat image(width, height, CV_8UC3, cv::Scalar(1, 1, 1));
  bool is_postprocess = has_postprocess > 0 ? true : false;
  for (int i = 0; i < warm_up + loop_num; i++) {
    double start = ncnn::get_current_time();
    std::vector<BoxInfo> results;
    detector.detect(image, results, is_postprocess);
    double end = ncnn::get_current_time();

    double time = end - start;
    if (i >= warm_up) {
      time_min = (std::min)(time_min, time);
      time_max = (std::max)(time_max, time);
      time_avg += time;
    }
  }
  time_avg /= loop_num;
  fprintf(stderr, "%20s  min = %7.2f  max = %7.2f  avg = %7.2f\n", "picodet",
          time_min, time_max, time_avg);
  return 0;
}

int main(int argc, char **argv) {
  int mode = atoi(argv[1]);
  char *bin_model_path = argv[2];
  char *param_model_path = argv[3];
  int height = 320;
  int width = 320;
  if (argc == 5) {
    height = atoi(argv[4]);
    width = atoi(argv[5]);
  }
  PicoDet detector =
      PicoDet(param_model_path, bin_model_path, width, height, true, 0.45, 0.3);
  if (mode == 1) {

    benchmark(detector, width, height, atoi(argv[6]));
  } else {
    if (argc != 6) {
      std::cout << "Must set image file, such as ./picodet_demo 0 "
                   "../picodet_s_320_lcnet.bin ../picodet_s_320_lcnet.param "
                   "320 320 img.jpg"
                << std::endl;
    }
    const char *images = argv[6];
    image_demo(detector, images, atoi(argv[7]));
  }
}
