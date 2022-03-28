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

#include <math.h>
#include <stdarg.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "include/config_parser.h"
#include "include/keypoint_detector.h"
#include "include/object_detector.h"
#include "include/preprocess_op.h"
#include "json/json.h"

Json::Value RT_Config;

void PrintBenchmarkLog(std::vector<double> det_time, int img_num) {
  std::cout << "----------------------- Config info -----------------------"
            << std::endl;
  std::cout << "num_threads: " << RT_Config["cpu_threads"].as<int>()
            << std::endl;
  std::cout << "----------------------- Data info -----------------------"
            << std::endl;
  std::cout << "batch_size_det: " << RT_Config["batch_size_det"].as<int>()
            << std::endl;
  std::cout << "----------------------- Model info -----------------------"
            << std::endl;
  RT_Config["model_dir_det"].as<std::string>().erase(
      RT_Config["model_dir_det"].as<std::string>().find_last_not_of("/") + 1);
  std::cout << "detection model_name: "
            << RT_Config["model_dir_det"].as<std::string>() << std::endl;
  std::cout << "----------------------- Perf info ------------------------"
            << std::endl;
  std::cout << "Total number of predicted data: " << img_num
            << " and total time spent(ms): "
            << std::accumulate(det_time.begin(), det_time.end(), 0.)
            << std::endl;
  img_num = std::max(1, img_num);
  std::cout << "preproce_time(ms): " << det_time[0] / img_num
            << ", inference_time(ms): " << det_time[1] / img_num
            << ", postprocess_time(ms): " << det_time[2] / img_num << std::endl;
}

void PrintKptsBenchmarkLog(std::vector<double> det_time, int img_num) {
  std::cout << "----------------------- Data info -----------------------"
            << std::endl;
  std::cout << "batch_size_keypoint: "
            << RT_Config["batch_size_keypoint"].as<int>() << std::endl;
  std::cout << "----------------------- Model info -----------------------"
            << std::endl;
  RT_Config["model_dir_keypoint"].as<std::string>().erase(
      RT_Config["model_dir_keypoint"].as<std::string>().find_last_not_of("/") +
      1);
  std::cout << "keypoint model_name: "
            << RT_Config["model_dir_keypoint"].as<std::string>() << std::endl;
  std::cout << "----------------------- Perf info ------------------------"
            << std::endl;
  std::cout << "Total number of predicted data: " << img_num
            << " and total time spent(ms): "
            << std::accumulate(det_time.begin(), det_time.end(), 0.)
            << std::endl;
  img_num = std::max(1, img_num);
  std::cout << "Average time cost per person:" << std::endl
            << "preproce_time(ms): " << det_time[0] / img_num
            << ", inference_time(ms): " << det_time[1] / img_num
            << ", postprocess_time(ms): " << det_time[2] / img_num << std::endl;
}

void PrintTotalIimeLog(double det_time,
                       double keypoint_time,
                       double crop_time) {
  std::cout << "----------------------- Time info ------------------------"
            << std::endl;
  std::cout << "Total Pipeline time(ms) per image: "
            << det_time + keypoint_time + crop_time << std::endl;
  std::cout << "Average det time(ms) per image: " << det_time
            << ", average keypoint time(ms) per image: " << keypoint_time
            << ", average crop time(ms) per image: " << crop_time << std::endl;
}

static std::string DirName(const std::string& filepath) {
  auto pos = filepath.rfind(OS_PATH_SEP);
  if (pos == std::string::npos) {
    return "";
  }
  return filepath.substr(0, pos);
}

static bool PathExists(const std::string& path) {
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
}

static void MkDir(const std::string& path) {
  if (PathExists(path)) return;
  int ret = 0;
  ret = mkdir(path.c_str(), 0755);
  if (ret != 0) {
    std::string path_error(path);
    path_error += " mkdir failed!";
    throw std::runtime_error(path_error);
  }
}

static void MkDirs(const std::string& path) {
  if (path.empty()) return;
  if (PathExists(path)) return;

  MkDirs(DirName(path));
  MkDir(path);
}

void PredictImage(const std::vector<std::string> all_img_paths,
                  const int batch_size_det,
                  const double threshold_det,
                  const bool run_benchmark,
                  PaddleDetection::ObjectDetector* det,
                  PaddleDetection::KeyPointDetector* keypoint,
                  const std::string& output_dir = "output") {
  std::vector<double> det_t = {0, 0, 0};
  int steps = ceil(static_cast<float>(all_img_paths.size()) / batch_size_det);
  int kpts_imgs = 0;
  std::vector<double> keypoint_t = {0, 0, 0};
  double midtimecost = 0;
  for (int idx = 0; idx < steps; idx++) {
    std::vector<cv::Mat> batch_imgs;
    int left_image_cnt = all_img_paths.size() - idx * batch_size_det;
    if (left_image_cnt > batch_size_det) {
      left_image_cnt = batch_size_det;
    }
    for (int bs = 0; bs < left_image_cnt; bs++) {
      std::string image_file_path = all_img_paths.at(idx * batch_size_det + bs);
      cv::Mat im = cv::imread(image_file_path, 1);
      batch_imgs.insert(batch_imgs.end(), im);
    }
    // Store all detected result
    std::vector<PaddleDetection::ObjectResult> result;
    std::vector<int> bbox_num;
    std::vector<double> det_times;

    // Store keypoint results
    std::vector<PaddleDetection::KeyPointResult> result_kpts;
    std::vector<cv::Mat> imgs_kpts;
    std::vector<std::vector<float>> center_bs;
    std::vector<std::vector<float>> scale_bs;
    std::vector<int> colormap_kpts = PaddleDetection::GenerateColorMap(20);
    bool is_rbox = false;
    if (run_benchmark) {
      det->Predict(
          batch_imgs, threshold_det, 50, 50, &result, &bbox_num, &det_times);
    } else {
      det->Predict(
          batch_imgs, threshold_det, 0, 1, &result, &bbox_num, &det_times);
    }

    // get labels and colormap
    auto labels = det->GetLabelList();
    auto colormap = PaddleDetection::GenerateColorMap(labels.size());
    int item_start_idx = 0;
    for (int i = 0; i < left_image_cnt; i++) {
      cv::Mat im = batch_imgs[i];
      std::vector<PaddleDetection::ObjectResult> im_result;
      int detect_num = 0;
      for (int j = 0; j < bbox_num[i]; j++) {
        PaddleDetection::ObjectResult item = result[item_start_idx + j];
        if (item.confidence < threshold_det || item.class_id == -1) {
          continue;
        }
        detect_num += 1;
        im_result.push_back(item);
        if (item.rect.size() > 6) {
          is_rbox = true;
          printf("class=%d confidence=%.4f rect=[%d %d %d %d %d %d %d %d]\n",
                 item.class_id,
                 item.confidence,
                 item.rect[0],
                 item.rect[1],
                 item.rect[2],
                 item.rect[3],
                 item.rect[4],
                 item.rect[5],
                 item.rect[6],
                 item.rect[7]);
        } else {
          printf("class=%d confidence=%.4f rect=[%d %d %d %d]\n",
                 item.class_id,
                 item.confidence,
                 item.rect[0],
                 item.rect[1],
                 item.rect[2],
                 item.rect[3]);
        }
      }
      std::cout << all_img_paths.at(idx * batch_size_det + i)
                << " The number of detected box: " << detect_num << std::endl;
      item_start_idx = item_start_idx + bbox_num[i];

      std::vector<int> compression_params;
      compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
      compression_params.push_back(95);
      std::string output_path(output_dir);
      if (output_dir.rfind(OS_PATH_SEP) != output_dir.size() - 1) {
        output_path += OS_PATH_SEP;
      }
      std::string image_file_path = all_img_paths.at(idx * batch_size_det + i);
      if (keypoint) {
        int imsize = im_result.size();
        for (int i = 0; i < imsize; i++) {
          auto keypoint_start_time = std::chrono::steady_clock::now();
          auto item = im_result[i];
          cv::Mat crop_img;
          std::vector<double> keypoint_times;
          std::vector<int> rect = {
              item.rect[0], item.rect[1], item.rect[2], item.rect[3]};
          std::vector<float> center;
          std::vector<float> scale;
          if (item.class_id == 0) {
            PaddleDetection::CropImg(im, crop_img, rect, center, scale);
            center_bs.emplace_back(center);
            scale_bs.emplace_back(scale);
            imgs_kpts.emplace_back(crop_img);
            kpts_imgs += 1;
          }
          auto keypoint_crop_time = std::chrono::steady_clock::now();

          std::chrono::duration<float> midtimediff =
              keypoint_crop_time - keypoint_start_time;
          midtimecost += static_cast<double>(midtimediff.count() * 1000);

          if (imgs_kpts.size() == RT_Config["batch_size_keypoint"].as<int>() ||
              ((i == imsize - 1) && !imgs_kpts.empty())) {
            if (run_benchmark) {
              keypoint->Predict(imgs_kpts,
                                center_bs,
                                scale_bs,
                                10,
                                10,
                                &result_kpts,
                                &keypoint_times);
            } else {
              keypoint->Predict(imgs_kpts,
                                center_bs,
                                scale_bs,
                                0,
                                1,
                                &result_kpts,
                                &keypoint_times);
            }
            imgs_kpts.clear();
            center_bs.clear();
            scale_bs.clear();
            keypoint_t[0] += keypoint_times[0];
            keypoint_t[1] += keypoint_times[1];
            keypoint_t[2] += keypoint_times[2];
          }
        }
        std::string kpts_savepath =
            output_path + "keypoint_" +
            image_file_path.substr(image_file_path.find_last_of('/') + 1);
        cv::Mat kpts_vis_img = VisualizeKptsResult(
            im, result_kpts, colormap_kpts, keypoint->get_threshold());
        cv::imwrite(kpts_savepath, kpts_vis_img, compression_params);
        printf("Visualized output saved as %s\n", kpts_savepath.c_str());
      } else {
        // Visualization result
        cv::Mat vis_img = PaddleDetection::VisualizeResult(
            im, im_result, labels, colormap, is_rbox);
        std::string det_savepath =
            output_path + "result_" +
            image_file_path.substr(image_file_path.find_last_of('/') + 1);
        cv::imwrite(det_savepath, vis_img, compression_params);
        printf("Visualized output saved as %s\n", det_savepath.c_str());
      }
    }

    det_t[0] += det_times[0];
    det_t[1] += det_times[1];
    det_t[2] += det_times[2];
  }
  PrintBenchmarkLog(det_t, all_img_paths.size());
  if (keypoint) {
    PrintKptsBenchmarkLog(keypoint_t, kpts_imgs);
    PrintTotalIimeLog(
        (det_t[0] + det_t[1] + det_t[2]) / all_img_paths.size(),
        (keypoint_t[0] + keypoint_t[1] + keypoint_t[2]) / all_img_paths.size(),
        midtimecost / all_img_paths.size());
  }
}

int main(int argc, char** argv) {
  std::cout << "Usage: " << argv[0] << " [config_path] [image_dir](option)\n";
  if (argc < 2) {
    std::cout << "Usage: ./main det_runtime_config.json" << std::endl;
    return -1;
  }
  std::string config_path = argv[1];
  std::string img_path = "";

  if (argc >= 3) {
    img_path = argv[2];
  }
  // Parsing command-line
  PaddleDetection::load_jsonf(config_path, RT_Config);
  if (RT_Config["model_dir_det"].as<std::string>().empty()) {
    std::cout << "Please set [model_det_dir] in " << config_path << std::endl;
    return -1;
  }
  if (RT_Config["image_file"].as<std::string>().empty() &&
      RT_Config["image_dir"].as<std::string>().empty() && img_path.empty()) {
    std::cout << "Please set [image_file] or [image_dir] in " << config_path
              << " Or use command: <" << argv[0] << " [image_dir]>"
              << std::endl;
    return -1;
  }
  if (!img_path.empty()) {
    std::cout << "Use image_dir in command line overide the path in config file"
              << std::endl;
    RT_Config["image_dir"] = img_path;
    RT_Config["image_file"] = "";
  }
  // Load model and create a object detector
  PaddleDetection::ObjectDetector det(
      RT_Config["model_dir_det"].as<std::string>(),
      RT_Config["cpu_threads"].as<int>(),
      RT_Config["batch_size_det"].as<int>());

  PaddleDetection::KeyPointDetector* keypoint = nullptr;
  if (!RT_Config["model_dir_keypoint"].as<std::string>().empty()) {
    keypoint = new PaddleDetection::KeyPointDetector(
        RT_Config["model_dir_keypoint"].as<std::string>(),
        RT_Config["cpu_threads"].as<int>(),
        RT_Config["batch_size_keypoint"].as<int>(),
        RT_Config["use_dark_decode"].as<bool>());
    RT_Config["batch_size_det"] = 1;
    printf(
        "batchsize of detection forced to be 1 while keypoint model is not "
        "empty()");
  }
  // Do inference on input image

  if (!RT_Config["image_file"].as<std::string>().empty() ||
      !RT_Config["image_dir"].as<std::string>().empty()) {
    if (!PathExists(RT_Config["output_dir"].as<std::string>())) {
      MkDirs(RT_Config["output_dir"].as<std::string>());
    }
    std::vector<std::string> all_img_paths;
    std::vector<cv::String> cv_all_img_paths;
    if (!RT_Config["image_file"].as<std::string>().empty()) {
      all_img_paths.push_back(RT_Config["image_file"].as<std::string>());
      if (RT_Config["batch_size_det"].as<int>() > 1) {
        std::cout << "batch_size_det should be 1, when set `image_file`."
                  << std::endl;
        return -1;
      }
    } else {
      cv::glob(RT_Config["image_dir"].as<std::string>(), cv_all_img_paths);
      for (const auto& img_path : cv_all_img_paths) {
        all_img_paths.push_back(img_path);
      }
    }
    PredictImage(all_img_paths,
                 RT_Config["batch_size_det"].as<int>(),
                 RT_Config["threshold_det"].as<float>(),
                 RT_Config["run_benchmark"].as<bool>(),
                 &det,
                 keypoint,
                 RT_Config["output_dir"].as<std::string>());
  }
  delete keypoint;
  keypoint = nullptr;
  return 0;
}
