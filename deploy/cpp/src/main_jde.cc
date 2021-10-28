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

#include <glog/logging.h>

#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <sys/types.h>
#include <sys/stat.h>
#include <math.h>
#include <algorithm>

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#elif LINUX
#include <stdarg.h>
#include <sys/stat.h>
#endif

#include "include/object_detector.h"
#include "include/jde_detector.h"
#include <gflags/gflags.h>
#include <opencv2/opencv.hpp>


DEFINE_string(model_dir, "", "Path of inference model");
DEFINE_int32(batch_size, 1, "batch_size");
DEFINE_string(video_file, "", "Path of input video, `video_file` or `camera_id` has a highest priority.");
DEFINE_int32(camera_id, -1, "Device id of camera to predict");
DEFINE_bool(use_gpu, false, "Deprecated, please use `--device` to set the device you want to run.");
DEFINE_string(device, "CPU", "Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU.");
DEFINE_double(threshold, 0.5, "Threshold of score.");
DEFINE_string(output_dir, "output", "Directory of output visualization files.");
DEFINE_string(run_mode, "fluid", "Mode of running(fluid/trt_fp32/trt_fp16/trt_int8)");
DEFINE_int32(gpu_id, 0, "Device id of GPU to execute");
DEFINE_bool(run_benchmark, false, "Whether to predict a image_file repeatedly for benchmark");
DEFINE_bool(use_mkldnn, false, "Whether use mkldnn with CPU");
DEFINE_int32(cpu_threads, 1, "Num of threads with CPU");
DEFINE_int32(trt_min_shape, 1, "Min shape of TRT DynamicShapeI");
DEFINE_int32(trt_max_shape, 1280, "Max shape of TRT DynamicShapeI");
DEFINE_int32(trt_opt_shape, 640, "Opt shape of TRT DynamicShapeI");
DEFINE_bool(trt_calib_mode, false, "If the model is produced by TRT offline quantitative calibration, trt_calib_mode need to set True");

void PrintBenchmarkLog(std::vector<double> det_time, int img_num){
  LOG(INFO) << "----------------------- Config info -----------------------";
  LOG(INFO) << "runtime_device: " << FLAGS_device;
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
            << " and total time spent(ms): "
            << std::accumulate(det_time.begin(), det_time.end(), 0);
  LOG(INFO) << "preproce_time(ms): " << det_time[0] / img_num
            << ", inference_time(ms): " << det_time[1] / img_num
            << ", postprocess_time(ms): " << det_time[2] / img_num;
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

void PredictVideo(const std::string& video_path,
                  PaddleDetection::JDEDetector* mot,
                  const std::string& output_dir = "output") {
  // Open video
  cv::VideoCapture capture;
  std::string video_out_name = "output.mp4";
  if (FLAGS_camera_id != -1){
    capture.open(FLAGS_camera_id);
  }else{
    capture.open(video_path.c_str());
    video_out_name = video_path.substr(video_path.find_last_of(OS_PATH_SEP) + 1);
  }
  if (!capture.isOpened()) {
    printf("can not open video : %s\n", video_path.c_str());
    return;
  }

  // Get Video info : resolution, fps, frame count
  int video_width = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_WIDTH));
  int video_height = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_HEIGHT));
  int video_fps = static_cast<int>(capture.get(CV_CAP_PROP_FPS));
  int video_frame_count = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_COUNT));
  printf("fps: %d, frame_count: %d\n", video_fps, video_frame_count);

  // Create VideoWriter for output
  cv::VideoWriter video_out;
  std::string video_out_path(output_dir);
  if (output_dir.rfind(OS_PATH_SEP) != output_dir.size() - 1) {
    video_out_path += OS_PATH_SEP;
  }
  video_out_path += video_out_name;
  video_out.open(video_out_path.c_str(),
                 0x00000021,
                 video_fps,
                 cv::Size(video_width, video_height),
                 true);
  if (!video_out.isOpened()) {
    printf("create video writer failed!\n");
    return;
  }

  PaddleDetection::MOT_Result result;
  std::vector<double> det_times(3);
  double times;
  // Capture all frames and do inference
  cv::Mat frame;
  int frame_id = 1;
  while (capture.read(frame)) {
    if (frame.empty()) {
      break;
    }
    std::vector<cv::Mat> imgs;
    imgs.push_back(frame);
    printf("detect frame: %d\n", frame_id);
    mot->Predict(imgs, FLAGS_threshold, 0, 1, &result, &det_times);
    frame_id += 1;
    times = std::accumulate(det_times.begin(), det_times.end(), 0) / frame_id;

    cv::Mat out_im = PaddleDetection::VisualizeTrackResult(
        frame, result, 1000./times, frame_id);
    
    video_out.write(out_im);
  }
  capture.release();
  video_out.release();
  PrintBenchmarkLog(det_times, frame_id);
  printf("Visualized output saved as %s\n", video_out_path.c_str());      
}

int main(int argc, char** argv) {
  // Parsing command-line
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_model_dir.empty()
      || FLAGS_video_file.empty()) {
    std::cout << "Usage: ./main --model_dir=/PATH/TO/INFERENCE_MODEL/ "
                << "--video_file=/PATH/TO/INPUT/VIDEO/" << std::endl;
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

  // Do inference on input video or image
  PaddleDetection::JDEDetector mot(FLAGS_model_dir, FLAGS_device, FLAGS_use_mkldnn,
                        FLAGS_cpu_threads, FLAGS_run_mode, FLAGS_batch_size,FLAGS_gpu_id,
                        FLAGS_trt_min_shape, FLAGS_trt_max_shape, FLAGS_trt_opt_shape,
			FLAGS_trt_calib_mode);
  if (!PathExists(FLAGS_output_dir)) {
      MkDirs(FLAGS_output_dir);
  }
  PredictVideo(FLAGS_video_file, &mot, FLAGS_output_dir);
  return 0;
}
