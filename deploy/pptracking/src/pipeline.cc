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

#include <sstream>
// for setprecision
#include <iostream>
#include <string>
#include <iomanip>
#include <chrono>
#include "include/pipeline.h"
#include "include/postprocess.h"
#include "include/predictor.h"

namespace PaddleDetection {


void Pipeline::SetInput(std::string& input_video) {
  input_.push_back(input_video);
}

void Pipeline::SelectModel(const std::string& scene,
                           const bool tiny_obj,
                           const bool is_mct) {
  // Single camera model
  // use deepsort for multiclass tracking
  // use fairmot for single class tracking
  if (scene == "pedestrian") {
      track_model_dir_ = "../pedestrian_track";
  } else if (scene != "vehicle") {
      track_model_dir_ = "../vehicle_track";
  } else if (scene == "multiclass") {
      det_model_dir_ = "../multiclass_det";
      reid_model_dir_ = "../multiclass_reid";
  }

  // Multi-camera model
  if (is_mct && scene == "pedestrian") {
      mct_model_dir_ = "../pedestrian_mct";
  } else if (is_mct && scene == "vehicle") {
      mct_model_dir_ = "../vehicle_mct";
  } else if (is_mct && scene == "multiclass") {
      throw "Multi-camera tracking is not supported in multiclass scene now.";
  } 
}

void Pipeline::Run() {

  if (track_model_dir_.empty()) {
    std::cout << "Pipeline must use SelectModel before Run";
    return;
  }
  if (input_.size() == 0) {
    std::cout << "Pipeline must use SetInput before Run";
    return;
  }

  if (mct_model_dir_.empty()) {
    // single camera
    if (input_.size() > 1) {
      throw "Single camera tracking except single video, but received %d", input_.size();
    }
    PredictSCT(input_[0]);
  } else {
    // multi cameras
    if (input_.size() != 2) {
      throw "Multi camera tracking except two videos, but received %d", input_.size();
    }
    PredictMCT(input_);
  }
}

void Pipeline::PredictSCT(const std::string& video_path) {

  PaddleDetection::Predictor sct(device_, track_model_dir_, det_model_dir_, reid_model_dir_, threshold_, run_mode_, gpu_id_, use_mkldnn_, cpu_threads_, trt_calib_mode_);
  // Open video
  cv::VideoCapture capture;
  capture.open(video_path.c_str());
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
  std::string video_out_path = "mot_output.mp4";
  int fcc = cv::VideoWriter::fourcc('m','p','4','v');
  video_out.open(video_out_path.c_str(),
                 fcc, //0x00000021,
                 video_fps,
                 cv::Size(video_width, video_height),
                 true);
  if (!video_out.isOpened()) {
    printf("create video writer failed!\n");
    return;
  }

  PaddleDetection::MOTResult result;
  std::vector<double> det_times(3);
  std::vector<int> count_list;
  std::vector<int> in_count_list;
  std::vector<int> out_count_list;
  double times;
  // Capture all frames and do inference
  cv::Mat frame;
  int frame_id = 0;
  while (capture.read(frame)) {
    if (frame.empty()) {
      break;
    }
    std::vector<cv::Mat> imgs;
    imgs.push_back(frame);
    printf("frame_id: %d\n", frame_id);
    sct.Predict(imgs, threshold_, &result, &det_times);
    frame_id += 1;
    times = std::accumulate(det_times.begin(), det_times.end(), 0) / frame_id;

    cv::Mat out_im = PaddleDetection::VisualizeTrackResult(
        frame, result, 1000./times, frame_id);
    
    if (count_) {
      // Count total number 
      // Count in & out number
      PaddleDetection::FlowStatistic(result, frame_id, &count_list, &in_count_list, &out_count_list);
    }
    if (save_result_) {
      PaddleDetection::SaveResult(result, output_dir_);
    }
    video_out.write(out_im);
  }
  capture.release();
  video_out.release();
  PrintBenchmarkLog(det_times, frame_id);
  printf("Visualized output saved as %s\n", video_out_path.c_str());
}

void Pipeline::PredictMCT(const std::vector<std::string> video_path) {
  throw "Not Implement!";
}

void Pipeline::PrintBenchmarkLog(std::vector<double> det_time, int img_num){
  LOG(INFO) << "----------------------- Config info -----------------------";
  LOG(INFO) << "runtime_device: " << device_;
  LOG(INFO) << "ir_optim: " << "True";
  LOG(INFO) << "enable_memory_optim: " << "True";
  int has_trt = run_mode_.find("trt");
  if (has_trt >= 0) {
    LOG(INFO) << "enable_tensorrt: " << "True";
    std::string precision = run_mode_.substr(4, 8);
    LOG(INFO) << "precision: " << precision;
  } else {
    LOG(INFO) << "enable_tensorrt: " << "False";
    LOG(INFO) << "precision: " << "fp32";
  }
  LOG(INFO) << "enable_mkldnn: " << (use_mkldnn_ ? "True" : "False");
  LOG(INFO) << "cpu_math_library_num_threads: " << cpu_threads_;
  LOG(INFO) << "----------------------- Perf info ------------------------";
  LOG(INFO) << "Total number of predicted data: " << img_num
            << " and total time spent(ms): "
            << std::accumulate(det_time.begin(), det_time.end(), 0.);
  img_num = std::max(1, img_num);
  LOG(INFO) << "preproce_time(ms): " << det_time[0] / img_num
            << ", inference_time(ms): " << det_time[1] / img_num
            << ", postprocess_time(ms): " << det_time[2] / img_num;
}


} // namespace PaddleDetection

