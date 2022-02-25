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

#include <math.h>
#include <sys/types.h>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#else
#include <stdarg.h>
#include <sys/stat.h>
#endif

#include <gflags/gflags.h>
#include "include/pipeline.h"

DEFINE_string(video_file, "", "Path of input video.");
DEFINE_string(video_other_file,
              "",
              "Path of other input video used for MTMCT.");
DEFINE_string(device,
              "CPU",
              "Choose the device you want to run, it can be: CPU/GPU/XPU, "
              "default is CPU.");
DEFINE_double(threshold, 0.5, "Threshold of score.");
DEFINE_string(output_dir, "output", "Directory of output visualization files.");
DEFINE_string(run_mode,
              "paddle",
              "Mode of running(paddle/trt_fp32/trt_fp16/trt_int8)");
DEFINE_int32(gpu_id, 0, "Device id of GPU to execute");
DEFINE_bool(use_mkldnn, false, "Whether use mkldnn with CPU");
DEFINE_int32(cpu_threads, 1, "Num of threads with CPU");
DEFINE_bool(trt_calib_mode,
            false,
            "If the model is produced by TRT offline quantitative calibration, "
            "trt_calib_mode need to set True");
DEFINE_bool(tiny_obj, false, "Whether tracking tiny object");
DEFINE_bool(do_entrance_counting,
            false,
            "Whether counting the numbers of identifiers entering "
            "or getting out from the entrance.");
DEFINE_int32(secs_interval, 10, "The seconds interval to count after tracking");
DEFINE_bool(save_result, false, "Whether saving result after tracking");
DEFINE_string(
    scene,
    "",
    "scene of tracking system, it can be : pedestrian/vehicle/multiclass");
DEFINE_bool(is_mtmct, false, "Whether use multi-target multi-camera tracking");
DEFINE_string(track_model_dir, "", "Path of tracking model");
DEFINE_string(det_model_dir, "", "Path of detection model");
DEFINE_string(reid_model_dir, "", "Path of reid model");

static std::string DirName(const std::string& filepath) {
  auto pos = filepath.rfind(OS_PATH_SEP);
  if (pos == std::string::npos) {
    return "";
  }
  return filepath.substr(0, pos);
}

static bool PathExists(const std::string& path) {
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

int main(int argc, char** argv) {
  // Parsing command-line
  google::ParseCommandLineFlags(&argc, &argv, true);
  bool has_model_dir =
      !(FLAGS_track_model_dir.empty() && FLAGS_det_model_dir.empty() &&
        FLAGS_reid_model_dir.empty());
  if (FLAGS_video_file.empty() || (FLAGS_scene.empty() && !has_model_dir)) {
    LOG(ERROR) << "Usage: \n"
               << "1. ./main -video_file=/PATH/TO/INPUT/IMAGE/ "
               << "-scene=pedestrian/vehicle/multiclass\n"
               << "2. ./main -video_file=/PATH/TO/INPUT/IMAGE/ "
               << "-track_model_dir=/PATH/TO/MODEL_DIR" << std::endl;

    return -1;
  }
  if (!(FLAGS_run_mode == "paddle" || FLAGS_run_mode == "trt_fp32" ||
        FLAGS_run_mode == "trt_fp16" || FLAGS_run_mode == "trt_int8")) {
    LOG(ERROR)
        << "run_mode should be 'paddle', 'trt_fp32', 'trt_fp16' or 'trt_int8'.";
    return -1;
  }
  transform(FLAGS_device.begin(),
            FLAGS_device.end(),
            FLAGS_device.begin(),
            ::toupper);
  if (!(FLAGS_device == "CPU" || FLAGS_device == "GPU" ||
        FLAGS_device == "XPU")) {
    LOG(ERROR) << "device should be 'CPU', 'GPU' or 'XPU'.";
    return -1;
  }

  if (!PathExists(FLAGS_output_dir)) {
    MkDirs(FLAGS_output_dir);
  }

  PaddleDetection::Pipeline pipeline(FLAGS_device,
                                     FLAGS_threshold,
                                     FLAGS_output_dir,
                                     FLAGS_run_mode,
                                     FLAGS_gpu_id,
                                     FLAGS_use_mkldnn,
                                     FLAGS_cpu_threads,
                                     FLAGS_trt_calib_mode,
                                     FLAGS_do_entrance_counting,
                                     FLAGS_save_result,
                                     FLAGS_scene,
                                     FLAGS_tiny_obj,
                                     FLAGS_is_mtmct,
                                     FLAGS_secs_interval,
                                     FLAGS_track_model_dir,
                                     FLAGS_det_model_dir,
                                     FLAGS_reid_model_dir);

  pipeline.SetInput(FLAGS_video_file);
  if (!FLAGS_video_other_file.empty()) {
    pipeline.SetInput(FLAGS_video_other_file);
  }
  pipeline.Run();
  return 0;
}
