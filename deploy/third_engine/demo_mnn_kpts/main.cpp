// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
// reference from https://github.com/RangiLyu/nanodet/tree/main/demo_mnn

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "keypoint_detector.h"
#include "picodet_mnn.h"

#define __SAVE_RESULT__  // if defined save drawed results to ../results, else
                         // show it in windows

using namespace PaddleDetection;

struct object_rect {
  int x;
  int y;
  int width;
  int height;
};

int resize_uniform(cv::Mat& src,
                   cv::Mat& dst,
                   cv::Size dst_size,
                   object_rect& effect_area) {
  int w = src.cols;
  int h = src.rows;
  int dst_w = dst_size.width;
  int dst_h = dst_size.height;
  dst = cv::Mat(cv::Size(dst_w, dst_h), CV_8UC3, cv::Scalar(0));

  float ratio_src = w * 1.0 / h;
  float ratio_dst = dst_w * 1.0 / dst_h;

  int tmp_w = 0;
  int tmp_h = 0;
  if (ratio_src > ratio_dst) {
    tmp_w = dst_w;
    tmp_h = floor((dst_w * 1.0 / w) * h);
  } else if (ratio_src < ratio_dst) {
    tmp_h = dst_h;
    tmp_w = floor((dst_h * 1.0 / h) * w);
  } else {
    cv::resize(src, dst, dst_size);
    effect_area.x = 0;
    effect_area.y = 0;
    effect_area.width = dst_w;
    effect_area.height = dst_h;
    return 0;
  }
  cv::Mat tmp;
  cv::resize(src, tmp, cv::Size(tmp_w, tmp_h));

  if (tmp_w != dst_w) {
    int index_w = floor((dst_w - tmp_w) / 2.0);
    for (int i = 0; i < dst_h; i++) {
      memcpy(dst.data + i * dst_w * 3 + index_w * 3,
             tmp.data + i * tmp_w * 3,
             tmp_w * 3);
    }
    effect_area.x = index_w;
    effect_area.y = 0;
    effect_area.width = tmp_w;
    effect_area.height = tmp_h;
  } else if (tmp_h != dst_h) {
    int index_h = floor((dst_h - tmp_h) / 2.0);
    memcpy(dst.data + index_h * dst_w * 3, tmp.data, tmp_w * tmp_h * 3);
    effect_area.x = 0;
    effect_area.y = index_h;
    effect_area.width = tmp_w;
    effect_area.height = tmp_h;
  } else {
    printf("error\n");
  }
  return 0;
}

const int color_list[80][3] = {
    {216, 82, 24},   {236, 176, 31},  {125, 46, 141},  {118, 171, 47},
    {76, 189, 237},  {238, 19, 46},   {76, 76, 76},    {153, 153, 153},
    {255, 0, 0},     {255, 127, 0},   {190, 190, 0},   {0, 255, 0},
    {0, 0, 255},     {170, 0, 255},   {84, 84, 0},     {84, 170, 0},
    {84, 255, 0},    {170, 84, 0},    {170, 170, 0},   {170, 255, 0},
    {255, 84, 0},    {255, 170, 0},   {255, 255, 0},   {0, 84, 127},
    {0, 170, 127},   {0, 255, 127},   {84, 0, 127},    {84, 84, 127},
    {84, 170, 127},  {84, 255, 127},  {170, 0, 127},   {170, 84, 127},
    {170, 170, 127}, {170, 255, 127}, {255, 0, 127},   {255, 84, 127},
    {255, 170, 127}, {255, 255, 127}, {0, 84, 255},    {0, 170, 255},
    {0, 255, 255},   {84, 0, 255},    {84, 84, 255},   {84, 170, 255},
    {84, 255, 255},  {170, 0, 255},   {170, 84, 255},  {170, 170, 255},
    {170, 255, 255}, {255, 0, 255},   {255, 84, 255},  {255, 170, 255},
    {42, 0, 0},      {84, 0, 0},      {127, 0, 0},     {170, 0, 0},
    {212, 0, 0},     {255, 0, 0},     {0, 42, 0},      {0, 84, 0},
    {0, 127, 0},     {0, 170, 0},     {0, 212, 0},     {0, 255, 0},
    {0, 0, 42},      {0, 0, 84},      {0, 0, 127},     {0, 0, 170},
    {0, 0, 212},     {0, 0, 255},     {0, 0, 0},       {36, 36, 36},
    {72, 72, 72},    {109, 109, 109}, {145, 145, 145}, {182, 182, 182},
    {218, 218, 218}, {0, 113, 188},   {80, 182, 188},  {127, 127, 0},
};

void draw_bboxes(const cv::Mat& bgr,
                 const std::vector<BoxInfo>& bboxes,
                 object_rect effect_roi,
                 std::string save_path = "None") {
  static const char* class_names[] = {
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

  cv::Mat image = bgr.clone();
  int src_w = image.cols;
  int src_h = image.rows;
  int dst_w = effect_roi.width;
  int dst_h = effect_roi.height;
  float width_ratio = (float)src_w / (float)dst_w;
  float height_ratio = (float)src_h / (float)dst_h;

  for (size_t i = 0; i < bboxes.size(); i++) {
    const BoxInfo& bbox = bboxes[i];
    cv::Scalar color = cv::Scalar(color_list[bbox.label][0],
                                  color_list[bbox.label][1],
                                  color_list[bbox.label][2]);
    cv::rectangle(image,
                  cv::Rect(cv::Point((bbox.x1 - effect_roi.x) * width_ratio,
                                     (bbox.y1 - effect_roi.y) * height_ratio),
                           cv::Point((bbox.x2 - effect_roi.x) * width_ratio,
                                     (bbox.y2 - effect_roi.y) * height_ratio)),
                  color);

    char text[256];
    sprintf(text, "%s %.1f%%", class_names[bbox.label], bbox.score * 100);

    int baseLine = 0;
    cv::Size label_size =
        cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

    int x = (bbox.x1 - effect_roi.x) * width_ratio;
    int y =
        (bbox.y1 - effect_roi.y) * height_ratio - label_size.height - baseLine;
    if (y < 0) y = 0;
    if (x + label_size.width > image.cols) x = image.cols - label_size.width;

    cv::rectangle(
        image,
        cv::Rect(cv::Point(x, y),
                 cv::Size(label_size.width, label_size.height + baseLine)),
        color,
        -1);

    cv::putText(image,
                text,
                cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX,
                0.4,
                cv::Scalar(255, 255, 255));
  }

  if (save_path == "None") {
    cv::imshow("image", image);
  } else {
    cv::imwrite(save_path, image);
    std::cout << save_path << std::endl;
  }
}

std::vector<BoxInfo> coordsback(const cv::Mat image,
                                const object_rect effect_roi,
                                const std::vector<BoxInfo>& bboxes) {
  int src_w = image.cols;
  int src_h = image.rows;
  int dst_w = effect_roi.width;
  int dst_h = effect_roi.height;
  float width_ratio = (float)src_w / (float)dst_w;
  float height_ratio = (float)src_h / (float)dst_h;

  std::vector<BoxInfo> bboxes_oimg;

  for (int i = 0; i < bboxes.size(); i++) {
    auto bbox = bboxes[i];
    bbox.x1 = (bbox.x1 - effect_roi.x) * width_ratio;
    bbox.y1 = (bbox.y1 - effect_roi.y) * height_ratio;
    bbox.x2 = (bbox.x2 - effect_roi.x) * width_ratio;
    bbox.y2 = (bbox.y2 - effect_roi.y) * height_ratio;
    bboxes_oimg.emplace_back(bbox);
  }
  return bboxes_oimg;
}

void image_infer_kpts(KeyPointDetector* kpts_detector,
                      cv::Mat image,
                      const object_rect effect_roi,
                      const std::vector<BoxInfo>& results,
                      std::string img_name = "kpts_vis",
                      bool save_img = true) {
  std::vector<cv::Mat> cropimgs;
  std::vector<std::vector<float>> center_bs;
  std::vector<std::vector<float>> scale_bs;
  std::vector<KeyPointResult> kpts_results;
  auto results_oimg = coordsback(image, effect_roi, results);

  for (int i = 0; i < results_oimg.size(); i++) {
    auto rect = results_oimg[i];
    if (rect.label == 0) {
      cv::Mat cropimg;
      std::vector<float> center, scale;
      std::vector<int> area = {static_cast<int>(rect.x1),
                               static_cast<int>(rect.y1),
                               static_cast<int>(rect.x2),
                               static_cast<int>(rect.y2)};
      CropImg(image, cropimg, area, center, scale);
      //   cv::imwrite("./test_crop_"+std::to_string(i)+".jpg", cropimg);
      cropimgs.emplace_back(cropimg);
      center_bs.emplace_back(center);
      scale_bs.emplace_back(scale);
    }
    if (cropimgs.size() == 1 ||
        (cropimgs.size() > 0 && i == results_oimg.size() - 1)) {
      kpts_detector->Predict(cropimgs, center_bs, scale_bs, &kpts_results);
      cropimgs.clear();
      center_bs.clear();
      scale_bs.clear();
    }
  }
  std::vector<int> compression_params;
  compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
  compression_params.push_back(95);
  std::string kpts_savepath =
      "keypoint_" + img_name.substr(img_name.find_last_of('/') + 1);
  cv::Mat kpts_vis_img =
      VisualizeKptsResult(image, kpts_results, {0, 255, 0}, 0.3);
  if (save_img) {
    cv::imwrite(kpts_savepath, kpts_vis_img, compression_params);
    printf("Visualized output saved as %s\n", kpts_savepath.c_str());
  } else {
    cv::imshow("image", kpts_vis_img);
  }
}

int image_demo(PicoDet& detector,
               KeyPointDetector* kpts_detector,
               const char* imagepath) {
  std::vector<cv::String> filenames;
  cv::glob(imagepath, filenames, false);

  for (auto img_name : filenames) {
    cv::Mat image = cv::imread(img_name);
    if (image.empty()) {
      fprintf(stderr, "cv::imread %s failed\n", img_name.c_str());
      return -1;
    }
    object_rect effect_roi;
    cv::Mat resized_img;
    resize_uniform(image, resized_img, cv::Size(320, 320), effect_roi);
    std::vector<BoxInfo> results;
    detector.detect(resized_img, results);
    if (kpts_detector) {
      image_infer_kpts(kpts_detector, image, effect_roi, results, img_name);
    }
  }
  return 0;
}

int webcam_demo(PicoDet& detector,
                KeyPointDetector* kpts_detector,
                int cam_id) {
  cv::Mat image;
  cv::VideoCapture cap(cam_id);

  while (true) {
    cap >> image;
    object_rect effect_roi;
    cv::Mat resized_img;
    resize_uniform(image, resized_img, cv::Size(320, 320), effect_roi);
    std::vector<BoxInfo> results;
    detector.detect(resized_img, results);
    if (kpts_detector) {
      image_infer_kpts(kpts_detector, image, effect_roi, results, "", false);
    }
  }
  return 0;
}

int video_demo(PicoDet& detector,
               KeyPointDetector* kpts_detector,
               const char* path) {
  cv::Mat image;
  cv::VideoCapture cap(path);

  while (true) {
    cap >> image;
    object_rect effect_roi;
    cv::Mat resized_img;
    resize_uniform(image, resized_img, cv::Size(320, 320), effect_roi);
    std::vector<BoxInfo> results;
    detector.detect(resized_img, results);
    if (kpts_detector) {
      image_infer_kpts(kpts_detector, image, effect_roi, results, "", false);
    }
  }
  return 0;
}

int benchmark(KeyPointDetector* kpts_detector) {
  int loop_num = 100;
  int warm_up = 8;

  double time_min = DBL_MAX;
  double time_max = -DBL_MAX;
  double time_avg = 0;
  cv::Mat image(256, 192, CV_8UC3, cv::Scalar(1, 1, 1));
  std::vector<float> center = {128, 96};
  std::vector<float> scale = {256, 192};
  std::vector<cv::Mat> cropimgs = {image};
  std::vector<std::vector<float>> center_bs = {center};
  std::vector<std::vector<float>> scale_bs = {scale};
  std::vector<KeyPointResult> kpts_results;

  for (int i = 0; i < warm_up + loop_num; i++) {
    auto start = std::chrono::steady_clock::now();
    std::vector<BoxInfo> results;
    kpts_detector->Predict(cropimgs, center_bs, scale_bs, &kpts_results);
    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    double time = elapsed.count();
    if (i >= warm_up) {
      time_min = (std::min)(time_min, time);
      time_max = (std::max)(time_max, time);
      time_avg += time;
    }
  }
  time_avg /= loop_num;
  fprintf(stderr,
          "%20s  min = %7.2f  max = %7.2f  avg = %7.2f\n",
          "tinypose",
          time_min,
          time_max,
          time_avg);
  return 0;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr,
            "usage: %s [mode] [path]. \n For webcam mode=0, path is cam id; \n "
            "For image demo, mode=1, path=xxx/xxx/*.jpg; \n For video, mode=2; "
            "\n For benchmark, mode=3 path=0.\n",
            argv[0]);
    return -1;
  }
  PicoDet detector =
      PicoDet("../weight/picodet_m_416.mnn", 416, 416, 4, 0.45, 0.3);
  KeyPointDetector* kpts_detector =
      new KeyPointDetector("../weight/tinypose256.mnn", 4, 256, 192);
  int mode = atoi(argv[1]);
  switch (mode) {
    case 0: {
      int cam_id = atoi(argv[2]);
      webcam_demo(detector, kpts_detector, cam_id);
      break;
    }
    case 1: {
      const char* images = argv[2];
      image_demo(detector, kpts_detector, images);
      break;
    }
    case 2: {
      const char* path = argv[2];
      video_demo(detector, kpts_detector, path);
      break;
    }
    case 3: {
      benchmark(kpts_detector);
      break;
    }
    default: {
      fprintf(stderr,
              "usage: %s [mode] [path]. \n For webcam mode=0, path is cam id; "
              "\n For image demo, mode=1, path=xxx/xxx/*.jpg; \n For video, "
              "mode=2; \n For benchmark, mode=3 path=0.\n",
              argv[0]);
      break;
    }
  }
  delete kpts_detector;
  kpts_detector = nullptr;
}
