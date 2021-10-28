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

#include "include/predictor.h"

namespace PaddleDetection {

class Pipeline {
 public:
  explicit Pipeline(const std::string& device,
                  const double threshold,
                  const std::string& output_dir,
                  const std::string& run_mode="fluid",
                  const int gpu_id=0,
                  const bool use_mkldnn=false,
                  const int cpu_threads=1,
                  const bool trt_calib_mode=false,
                  const bool count=false,
                  const bool save_result=false) {
    std::vector<std::string> input;
    this->input_ = input;
    this->device_ = device;
    this->threshold_ = threshold;
    this->output_dir_ = output_dir;
    this->run_mode_ = run_mode;
    this->gpu_id_ = gpu_id;
    this->use_mkldnn_ = use_mkldnn;
    this->cpu_threads_ = cpu_threads;
    this->trt_calib_mode_ = trt_calib_mode;
    this->count_ = count;
    this->save_result_ = save_result;
  }


  // Select model according to scenes, it must execute before Run()
  void SelectModel(const std::string& scene="pedestrian",
                   const bool tiny_obj=false,
                   const bool is_mct=false);

  // Set input, it must execute before Run()
  void SetInput(std::string& input_video);

  // Run pipeline
  void Run();

  void PredictSCT(const std::string& video_path);
  void PredictMCT(const std::vector<std::string> video_inputs);

  void PrintBenchmarkLog(std::vector<double> det_time, int img_num);

 private:
  std::vector<std::string> input_;
  std::string device_;
  double threshold_;
  std::string output_dir_;
  std::string track_model_dir_;
  std::string det_model_dir_;
  std::string reid_model_dir_;
  std::string mct_model_dir_;
  std::string run_mode_ = "fluid";
  int gpu_id_ = 0;
  bool use_mkldnn_ = false;
  int cpu_threads_ = 1;
  bool trt_calib_mode_ = false;
  bool count_ = false;
  bool save_result_ = false;
};

} // namespace PaddleDetection
