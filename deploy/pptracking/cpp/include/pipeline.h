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

#ifndef DEPLOY_PPTRACKING_CPP_INCLUDE_PIPELINE_H_
#define DEPLOY_PPTRACKING_CPP_INCLUDE_PIPELINE_H_

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
#elif LINUX
#include <stdarg.h>
#include <sys/stat.h>
#endif

#include "include/jde_predictor.h"
#include "include/sde_predictor.h"

namespace PaddleDetection {

class Pipeline {
 public:
  explicit Pipeline(const std::string& device,
                    const double threshold,
                    const std::string& output_dir,
                    const std::string& run_mode = "paddle",
                    const int gpu_id = 0,
                    const bool use_mkldnn = false,
                    const int cpu_threads = 1,
                    const bool trt_calib_mode = false,
                    const bool do_entrance_counting = false,
                    const bool save_result = false,
                    const std::string& scene = "pedestrian",
                    const bool tiny_obj = false,
                    const bool is_mtmct = false,
                    const int secs_interval = 10,
                    const std::string track_model_dir = "",
                    const std::string det_model_dir = "",
                    const std::string reid_model_dir = "") {
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
    this->do_entrance_counting_ = do_entrance_counting;
    this->secs_interval_ = secs_interval_;
    this->save_result_ = save_result;
    SelectModel(scene,
                tiny_obj,
                is_mtmct,
                track_model_dir,
                det_model_dir,
                reid_model_dir);
    InitPredictor();
  }

  // Set input, it must execute before Run()
  void SetInput(const std::string& input_video);
  void ClearInput();

  // Run pipeline in video
  void Run();
  void PredictMOT(const std::string& video_path);
  void PredictMTMCT(const std::vector<std::string> video_inputs);

  // Run pipeline in stream
  void RunMOTStream(const cv::Mat img,
                    const int frame_id,
                    const int video_fps,
                    const Rect entrance,
                    cv::Mat out_img,
                    std::vector<std::string>* records,
                    std::set<int>* count_set,
                    std::set<int>* interval_count_set,
                    std::vector<int>* in_count_list,
                    std::vector<int>* out_count_list,
                    std::map<int, std::vector<float>>* prev_center,
                    std::vector<std::string>* flow_records);
  void RunMTMCTStream(const std::vector<cv::Mat> imgs,
                      std::vector<std::string>* records);

  void PrintBenchmarkLog(const std::vector<double> det_time, const int img_num);

 private:
  // Select model according to scenes, it must execute before Run()
  void SelectModel(const std::string& scene = "pedestrian",
                   const bool tiny_obj = false,
                   const bool is_mtmct = false,
                   const std::string track_model_dir = "",
                   const std::string det_model_dir = "",
                   const std::string reid_model_dir = "");
  void InitPredictor();

  std::shared_ptr<PaddleDetection::JDEPredictor> jde_sct_;
  std::shared_ptr<PaddleDetection::SDEPredictor> sde_sct_;

  std::vector<std::string> input_;
  std::vector<cv::Mat> stream_;
  std::string device_;
  double threshold_;
  std::string output_dir_;
  std::string track_model_dir_;
  std::string det_model_dir_;
  std::string reid_model_dir_;
  std::string run_mode_ = "paddle";
  int gpu_id_ = 0;
  bool use_mkldnn_ = false;
  int cpu_threads_ = 1;
  bool trt_calib_mode_ = false;
  bool do_entrance_counting_ = false;
  bool save_result_ = false;
  int secs_interval_ = 10;
};

}  // namespace PaddleDetection

#endif  // DEPLOY_PPTRACKING_CPP_INCLUDE_PIPELINE_H_
