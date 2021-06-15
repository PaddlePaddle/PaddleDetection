//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <dirent.h>
#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <sys/types.h>
#include <sys/stat.h>
#include <math.h>

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#elif LINUX
#include <stdarg.h>
#include <sys/stat.h>
#endif

#include "include/object_detector.h"
#include <gflags/gflags.h>


DEFINE_string(model_dir, "", "Path of inference model");
DEFINE_string(image_file, "", "Path of input image");
DEFINE_string(image_dir, "", "Dir of input image, `image_file` has a higher priority.");
DEFINE_int32(batch_size, 1, "batch_size");
DEFINE_string(video_file, "", "Path of input video, `video_file` or `camera_id` has a highest priority.");
DEFINE_int32(camera_id, -1, "Device id of camera to predict");
DEFINE_bool(use_gpu, false, "Infering with GPU or CPU");
DEFINE_double(threshold, 0.5, "Threshold of score.");
DEFINE_string(output_dir, "output", "Directory of output visualization files.");
DEFINE_string(run_mode, "fluid", "Mode of running(fluid/trt_fp32/trt_fp16/trt_int8)");
DEFINE_int32(gpu_id, 0, "Device id of GPU to execute");
DEFINE_bool(run_benchmark, false, "Whether to predict a image_file repeatedly for benchmark");
DEFINE_bool(use_mkldnn, false, "Whether use mkldnn with CPU");
DEFINE_int32(cpu_threads, 1, "Num of threads with CPU");
DEFINE_bool(use_dynamic_shape, false, "Trt use dynamic shape or not");
DEFINE_int32(trt_min_shape, 1, "Min shape of TRT DynamicShapeI");
DEFINE_int32(trt_max_shape, 1280, "Max shape of TRT DynamicShapeI");
DEFINE_int32(trt_opt_shape, 640, "Opt shape of TRT DynamicShapeI");
DEFINE_bool(trt_calib_mode, false, "If the model is produced by TRT offline quantitative calibration, trt_calib_mode need to set True");

void PrintBenchmarkLog(std::vector<double> det_time, int img_num){
  LOG(INFO) << "----------------------- Config info -----------------------";
  LOG(INFO) << "runtime_device: " << (FLAGS_use_gpu ? "gpu" : "cpu");
  LOG(INFO) << "ir_optim: " << "True";
  LOG(INFO) << "enable_memory_optim: " << "True";
  int has_trt = FLAGS_run_mode.find("trt");
  if (has_trt >= 0) {
    LOG(INFO) << "enable_tensorrt: " << "True";
    std::string precision = FLAGS_run_mode.substr(4, 8);
    LOG(INFO) << "precision: " << precision;
  } else {
    LOG(INFO) << "enable_tensorrt: " << "False";
    LOG(INFO) << "precision: " << "fp32";
  }
  LOG(INFO) << "enable_mkldnn: " << (FLAGS_use_mkldnn ? "True" : "False");
  LOG(INFO) << "cpu_math_library_num_threads: " << FLAGS_cpu_threads;
  LOG(INFO) << "----------------------- Data info -----------------------";
  LOG(INFO) << "batch_size: " << FLAGS_batch_size;
  LOG(INFO) << "input_shape: " << "dynamic shape";
  LOG(INFO) << "----------------------- Model info -----------------------";
  FLAGS_model_dir.erase(FLAGS_model_dir.find_last_not_of("/") + 1);
  LOG(INFO) << "model_name: " << FLAGS_model_dir.substr(FLAGS_model_dir.find_last_of('/') + 1);
  LOG(INFO) << "----------------------- Perf info ------------------------";
  LOG(INFO) << "Total number of predicted data: " << img_num
            << " and total time spent(s): "
            << std::accumulate(det_time.begin(), det_time.end(), 0);
  LOG(INFO) << "preproce_time(ms): " << det_time[0] / img_num
            << ", inference_time(ms): " << det_time[1] / img_num
            << ", postprocess_time(ms): " << det_time[2];
}

static std::string DirName(const std::string &filepath) {
  auto pos = filepath.rfind(OS_PATH_SEP);
  if (pos == std::string::npos) {
    return "";
  }
  return filepath.substr(0, pos);
}

static bool PathExists(const std::string& path){
#ifdef _WIN32
  struct _stat buffer;
  return (_stat(path.c_str(), &buffer) == 0);
#else
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
#endif  // !_WIN32
}

static void MkDir(const std::string& path) {
  if (PathExists(path)) return;
  int ret = 0;
#ifdef _WIN32
  ret = _mkdir(path.c_str());
#else
  ret = mkdir(path.c_str(), 0755);
#endif  // !_WIN32
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

void GetAllFiles(const char *dir_name,
                          std::vector<std::string> &all_inputs) {
  if (NULL == dir_name) {
    std::cout << " dir_name is null ! " << std::endl;
    return;
  }
  struct stat s;
  lstat(dir_name, &s);
  if (!S_ISDIR(s.st_mode)) {
    std::cout << "dir_name is not a valid directory !" << std::endl;
    all_inputs.push_back(dir_name);
    return;
  } else {
    struct dirent *filename; // return value for readdir()
    DIR *dir;                // return value for opendir()
    dir = opendir(dir_name);
    if (NULL == dir) {
      std::cout << "Can not open dir " << dir_name << std::endl;
      return;
    }
    std::cout << "Successfully opened the dir !" << std::endl;
    while ((filename = readdir(dir)) != NULL) {
      if (strcmp(filename->d_name, ".") == 0 ||
          strcmp(filename->d_name, "..") == 0)
        continue;
      all_inputs.push_back(dir_name + std::string("/") +
                           std::string(filename->d_name));
    }
  }
}

void PredictVideo(const std::string& video_path,
                  PaddleDetection::ObjectDetector* det) {
  // Open video
  cv::VideoCapture capture;
  if (FLAGS_camera_id != -1){
    capture.open(FLAGS_camera_id);
  }else{
    capture.open(video_path.c_str());
  }
  if (!capture.isOpened()) {
    printf("can not open video : %s\n", video_path.c_str());
    return;
  }

  // Get Video info : resolution, fps
  int video_width = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_WIDTH));
  int video_height = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_HEIGHT));
  int video_fps = static_cast<int>(capture.get(CV_CAP_PROP_FPS));

  // Create VideoWriter for output
  cv::VideoWriter video_out;
  std::string video_out_path = "output.mp4";
  video_out.open(video_out_path.c_str(),
                 0x00000021,
                 video_fps,
                 cv::Size(video_width, video_height),
                 true);
  if (!video_out.isOpened()) {
    printf("create video writer failed!\n");
    return;
  }

  std::vector<PaddleDetection::ObjectResult> result;
  std::vector<int> bbox_num;
  std::vector<double> det_times;
  auto labels = det->GetLabelList();
  auto colormap = PaddleDetection::GenerateColorMap(labels.size());
  // Capture all frames and do inference
  cv::Mat frame;
  int frame_id = 0;
  bool is_rbox = false;
  while (capture.read(frame)) {
    if (frame.empty()) {
      break;
    }
    std::vector<cv::Mat> imgs;
    imgs.push_back(frame);
    det->Predict(imgs, 0.5, 0, 1, &result, &bbox_num, &det_times);
    for (const auto& item : result) {
      if (item.rect.size() > 6){
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
      }
      else{
        printf("class=%d confidence=%.4f rect=[%d %d %d %d]\n",
          item.class_id,
          item.confidence,
          item.rect[0],
          item.rect[1],
          item.rect[2],
          item.rect[3]);
      }
   }

   cv::Mat out_im = PaddleDetection::VisualizeResult(
        frame, result, labels, colormap, is_rbox);

    video_out.write(out_im);
    frame_id += 1;
  }
  capture.release();
  video_out.release();
}

void PredictImage(const std::vector<std::string> all_img_paths,
                  const int batch_size,
                  const double threshold,
                  const bool run_benchmark,
                  PaddleDetection::ObjectDetector* det,
                  const std::string& output_dir = "output") {
  std::vector<double> det_t = {0, 0, 0};
  int steps = ceil(float(all_img_paths.size()) / batch_size);
  printf("total images = %d, batch_size = %d, total steps = %d\n",
                all_img_paths.size(), batch_size, steps);
  for (int idx = 0; idx < steps; idx++) {
    std::vector<cv::Mat> batch_imgs;
    int left_image_cnt = all_img_paths.size() - idx * batch_size;
    if (left_image_cnt > batch_size) {
      left_image_cnt = batch_size;
    }
    for (int bs = 0; bs < left_image_cnt; bs++) {
      std::string image_file_path = all_img_paths.at(idx * batch_size+bs);
      cv::Mat im = cv::imread(image_file_path, 1);
      batch_imgs.insert(batch_imgs.end(), im);
    }
    
    // Store all detected result
    std::vector<PaddleDetection::ObjectResult> result;
    std::vector<int> bbox_num;
    std::vector<double> det_times;
    bool is_rbox = false;
    if (run_benchmark) {
      det->Predict(batch_imgs, threshold, 10, 10, &result, &bbox_num,  &det_times);
    } else {
      det->Predict(batch_imgs, 0.5, 0, 1, &result, &bbox_num, &det_times);
      // get labels and colormap
      auto labels = det->GetLabelList();
      auto colormap = PaddleDetection::GenerateColorMap(labels.size());

      int item_start_idx = 0;
      for (int i = 0; i < left_image_cnt; i++) {
        std::cout << all_img_paths.at(idx * batch_size + i) << " bbox_num " << bbox_num[i] << std::endl;
        if (bbox_num[i] <= 1) {
            continue;
        }
        for (int j = 0; j < bbox_num[i]; j++) {
          PaddleDetection::ObjectResult item = result[item_start_idx + j];
          if (item.confidence < threshold) {
            continue;
          }
          if (item.rect.size() > 6){
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
          }
          else{
            printf("class=%d confidence=%.4f rect=[%d %d %d %d]\n",
              item.class_id,
              item.confidence,
              item.rect[0],
              item.rect[1],
              item.rect[2],
              item.rect[3]);
          }
        }
        item_start_idx = item_start_idx + bbox_num[i];
      }
      // Visualization result
      int bbox_idx = 0;
      for (int bs = 0; bs < batch_imgs.size(); bs++) {
        if (bbox_num[bs] <= 1) {
            continue;
        }
        cv::Mat im = batch_imgs[bs];
        std::vector<PaddleDetection::ObjectResult> im_result;
        for (int k = 0; k < bbox_num[bs]; k++) {
          im_result.push_back(result[bbox_idx+k]);
        }
        bbox_idx += bbox_num[bs];
        cv::Mat vis_img = PaddleDetection::VisualizeResult(
            im, im_result, labels, colormap, is_rbox);
        std::vector<int> compression_params;
        compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
        compression_params.push_back(95);
        std::string output_path(output_dir);
        if (output_dir.rfind(OS_PATH_SEP) != output_dir.size() - 1) {
          output_path += OS_PATH_SEP;
        }
        std::string image_file_path = all_img_paths.at(idx * batch_size + bs);
        output_path += image_file_path.substr(image_file_path.find_last_of('/') + 1);
        cv::imwrite(output_path, vis_img, compression_params);
        printf("Visualized output saved as %s\n", output_path.c_str());
      }
    }
    det_t[0] += det_times[0];
    det_t[1] += det_times[1];
    det_t[2] += det_times[2];
  }
  PrintBenchmarkLog(det_t, all_img_paths.size());
}

int main(int argc, char** argv) {
  // Parsing command-line
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_model_dir.empty()
      || (FLAGS_image_file.empty() && FLAGS_image_dir.empty() && FLAGS_video_file.empty())) {
    std::cout << "Usage: ./main --model_dir=/PATH/TO/INFERENCE_MODEL/ "
                << "--image_file=/PATH/TO/INPUT/IMAGE/" << std::endl;
    return -1;
  }
  if (!(FLAGS_run_mode == "fluid" || FLAGS_run_mode == "trt_fp32"
      || FLAGS_run_mode == "trt_fp16" || FLAGS_run_mode == "trt_int8")) {
    std::cout << "run_mode should be 'fluid', 'trt_fp32', 'trt_fp16' or 'trt_int8'.";
    return -1;
  }
  // Load model and create a object detector
  PaddleDetection::ObjectDetector det(FLAGS_model_dir, FLAGS_use_gpu, FLAGS_use_mkldnn,
                        FLAGS_cpu_threads, FLAGS_run_mode, FLAGS_batch_size,FLAGS_gpu_id, FLAGS_use_dynamic_shape,
                        FLAGS_trt_min_shape, FLAGS_trt_max_shape, FLAGS_trt_opt_shape, FLAGS_trt_calib_mode);
  // Do inference on input video or image
  if (!FLAGS_video_file.empty() || FLAGS_camera_id != -1) {
    PredictVideo(FLAGS_video_file, &det);
  } else if (!FLAGS_image_file.empty() || !FLAGS_image_dir.empty()) {
    if (!PathExists(FLAGS_output_dir)) {
      MkDirs(FLAGS_output_dir);
    }
    std::vector<std::string> all_imgs;
    if (!FLAGS_image_file.empty()) {
      all_imgs.push_back(FLAGS_image_file);
      if (FLAGS_batch_size > 1) {
        std::cout << "batch_size should be 1, when image_file is not None" << std::endl;
        FLAGS_batch_size = 1;
      }
    } else {
      GetAllFiles((char *)FLAGS_image_dir.c_str(), all_imgs);
    }
    PredictImage(all_imgs, FLAGS_batch_size, FLAGS_threshold, FLAGS_run_benchmark, &det, FLAGS_output_dir);
  }
  return 0;
}
