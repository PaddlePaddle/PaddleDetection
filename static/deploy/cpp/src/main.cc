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

#include <iostream>
#include <string>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <algorithm>

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
DEFINE_string(video_path, "", "Path of input video");
DEFINE_bool(use_gpu, false, "Deprecated, please use `--device` to set the device you want to run.");
DEFINE_string(device, "CPU", "Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU.");
DEFINE_bool(use_camera, false, "Use camera or not");
DEFINE_string(run_mode, "fluid", "Mode of running(fluid/trt_fp32/trt_fp16)");
DEFINE_int32(gpu_id, 0, "Device id of GPU to execute");
DEFINE_int32(camera_id, -1, "Device id of camera to predict");
DEFINE_bool(run_benchmark, false, "Whether to predict a image_file repeatedly for benchmark");
DEFINE_double(threshold, 0.5, "Threshold of score.");
DEFINE_string(output_dir, "output", "Directory of output visualization files.");
DEFINE_bool(trt_calib_mode, false, "If the model is produced by TRT offline quantitative calibration, trt_calib_mode need to set True");

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
  auto labels = det->GetLabelList();
  auto colormap = PaddleDetection::GenerateColorMap(labels.size());
  // Capture all frames and do inference
  cv::Mat frame;
  int frame_id = 0;
  while (capture.read(frame)) {
    if (frame.empty()) {
      break;
    }
    det->Predict(frame, 0.5, 0, 1, false, &result);
    cv::Mat out_im = PaddleDetection::VisualizeResult(
        frame, result, labels, colormap);
    for (const auto& item : result) {
      printf("In frame id %d, we detect: class=%d confidence=%.2f rect=[%d %d %d %d]\n",
        frame_id,
        item.class_id,
        item.confidence,
        item.rect[0],
        item.rect[1],
        item.rect[2],
        item.rect[3]);
   }   
    video_out.write(out_im);
    frame_id += 1;
  }
  capture.release();
  video_out.release();
}

void PredictImage(const std::string& image_path,
                  const double threshold,
                  const bool run_benchmark,
                  PaddleDetection::ObjectDetector* det,
                  const std::string& output_dir = "output") {
  // Open input image as an opencv cv::Mat object
  cv::Mat im = cv::imread(image_path, 1);
  // Store all detected result
  std::vector<PaddleDetection::ObjectResult> result;
  if (run_benchmark)
  {
    det->Predict(im, threshold, 100, 100, run_benchmark, &result);
  }else
  {
    det->Predict(im, 0.5, 0, 1, run_benchmark, &result);
    for (const auto& item : result) {
      printf("class=%d confidence=%.4f rect=[%d %d %d %d]\n",
          item.class_id,
          item.confidence,
          item.rect[0],
          item.rect[1],
          item.rect[2],
          item.rect[3]);
    }
    // Visualization result
    auto labels = det->GetLabelList();
    auto colormap = PaddleDetection::GenerateColorMap(labels.size());
    cv::Mat vis_img = PaddleDetection::VisualizeResult(
        im, result, labels, colormap);
    std::vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
    compression_params.push_back(95);
    std::string output_path(output_dir);
    if (output_dir.rfind(OS_PATH_SEP) != output_dir.size() - 1) {
      output_path += OS_PATH_SEP;
    }
    output_path += "output.jpg";
    cv::imwrite(output_path, vis_img, compression_params);
    printf("Visualized output saved as %s\n", output_path.c_str());
  }
}

int main(int argc, char** argv) {
  // Parsing command-line
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_model_dir.empty()
      || (FLAGS_image_file.empty() && FLAGS_video_path.empty())) {
    std::cout << "Usage: ./main --model_dir=/PATH/TO/INFERENCE_MODEL/ "
                << "--image_file=/PATH/TO/INPUT/IMAGE/" << std::endl;
    return -1;
  }
  if (!(FLAGS_run_mode == "fluid" || FLAGS_run_mode == "trt_fp32"
      || FLAGS_run_mode == "trt_fp16" || FLAGS_run_mode == "trt_int8")) {
    std::cout << "run_mode should be 'fluid', 'trt_fp32', 'trt_fp16' or 'trt_int8'.";
    return -1;
  }
  transform(FLAGS_device.begin(),FLAGS_device.end(),FLAGS_device.begin(),::toupper);
  if (!(FLAGS_device == "CPU" || FLAGS_device == "GPU" || FLAGS_device == "XPU")) {
    std::cout << "device should be 'CPU', 'GPU' or 'XPU'.";
    return -1;
  }
  if (FLAGS_use_gpu) {
    std::cout << "Deprecated, please use `--device` to set the device you want to run.";
    return -1;
  }

  // Load model and create a object detector
  PaddleDetection::ObjectDetector det(FLAGS_model_dir, FLAGS_device,
    FLAGS_run_mode, FLAGS_gpu_id, FLAGS_trt_calib_mode);
  // Do inference on input video or image
  if (!FLAGS_video_path.empty() || FLAGS_use_camera) {
    PredictVideo(FLAGS_video_path, &det);
  } else if (!FLAGS_image_file.empty()) {
    if (!PathExists(FLAGS_output_dir)) {
      MkDirs(FLAGS_output_dir);
    }
    PredictImage(FLAGS_image_file, FLAGS_threshold, FLAGS_run_benchmark, &det, FLAGS_output_dir);
  }
  return 0;
}
