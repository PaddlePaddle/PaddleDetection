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
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include "include/pipeline.h"
#include "include/postprocess.h"
#include "include/predictor.h"

namespace PaddleDetection {

void Pipeline::SetInput(const std::string& input_video) {
  input_.push_back(input_video);
}

void Pipeline::ClearInput() {
  input_.clear();
  stream_.clear();
}

void Pipeline::SelectModel(const std::string& scene,
                           const bool tiny_obj,
                           const bool is_mtmct,
                           const std::string track_model_dir,
                           const std::string det_model_dir,
                           const std::string reid_model_dir) {
  // model_dir has higher priority
  if (!track_model_dir.empty()) {
    track_model_dir_ = track_model_dir;
    return;
  }
  if (!det_model_dir.empty() && !reid_model_dir.empty()) {
    det_model_dir_ = det_model_dir;
    reid_model_dir_ = reid_model_dir;
    return;
  }

  // Single camera model, based on FairMot
  if (scene == "pedestrian") {
    if (tiny_obj) {
      track_model_dir_ = "../pedestrian_track_tiny";
    } else {
      track_model_dir_ = "../pedestrian_track";
    }
  } else if (scene != "vehicle") {
    if (tiny_obj) {
      track_model_dir_ = "../vehicle_track_tiny";
    } else {
      track_model_dir_ = "../vehicle_track";
    }
  } else if (scene == "multiclass") {
    if (tiny_obj) {
      track_model_dir_ = "../multiclass_track_tiny";
    } else {
      track_model_dir_ = "../multiclass_track";
    }
  }

  // Multi-camera model, based on PicoDet & LCNet
  if (is_mtmct && scene == "pedestrian") {
    det_model_dir_ = "../pedestrian_det";
    reid_model_dir_ = "../pedestrian_reid";
  } else if (is_mtmct && scene == "vehicle") {
    det_model_dir_ = "../vehicle_det";
    reid_model_dir_ = "../vehicle_reid";
  } else if (is_mtmct && scene == "multiclass") {
    throw "Multi-camera tracking is not supported in multiclass scene now.";
  }
}

void Pipeline::InitPredictor() {
  if (track_model_dir_.empty() && det_model_dir_.empty()) {
    throw "Predictor must receive track_model or det_model!";
  }

  if (!track_model_dir_.empty()) {
    jde_sct_ = std::make_shared<PaddleDetection::JDEPredictor>(device_,
                                                               track_model_dir_,
                                                               threshold_,
                                                               run_mode_,
                                                               gpu_id_,
                                                               use_mkldnn_,
                                                               cpu_threads_,
                                                               trt_calib_mode_);
  }
  if (!det_model_dir_.empty()) {
    sde_sct_ = std::make_shared<PaddleDetection::SDEPredictor>(device_,
                                                               det_model_dir_,
                                                               reid_model_dir_,
                                                               threshold_,
                                                               run_mode_,
                                                               gpu_id_,
                                                               use_mkldnn_,
                                                               cpu_threads_,
                                                               trt_calib_mode_);
  }
}

void Pipeline::Run() {
  if (track_model_dir_.empty() && det_model_dir_.empty()) {
    LOG(ERROR) << "Pipeline must use SelectModel before Run";
    return;
  }
  if (input_.size() == 0) {
    LOG(ERROR) << "Pipeline must use SetInput before Run";
    return;
  }

  if (!track_model_dir_.empty()) {
    // single camera
    if (input_.size() > 1) {
      throw "Single camera tracking except single video, but received %d",
          input_.size();
    }
    PredictMOT(input_[0]);
  } else {
    // multi cameras
    if (input_.size() != 2) {
      throw "Multi camera tracking except two videos, but received %d",
          input_.size();
    }
    PredictMTMCT(input_);
  }
}

void Pipeline::PredictMOT(const std::string& video_path) {
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

  LOG(INFO) << "----------------------- Input info -----------------------";
  LOG(INFO) << "video_width: " << video_width;
  LOG(INFO) << "video_height: " << video_height;
  LOG(INFO) << "input fps: " << video_fps;

  // Create VideoWriter for output
  cv::VideoWriter video_out;
  std::string video_out_path = output_dir_ + OS_PATH_SEP + "mot_output.mp4";
  int fcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
  video_out.open(video_out_path.c_str(),
                 fcc,  // 0x00000021,
                 video_fps,
                 cv::Size(video_width, video_height),
                 true);
  if (!video_out.isOpened()) {
    printf("create video writer failed!\n");
    return;
  }

  PaddleDetection::MOTResult result;
  std::vector<double> det_times(3);
  std::set<int> id_set;
  std::set<int> interval_id_set;
  std::vector<int> in_id_list;
  std::vector<int> out_id_list;
  std::map<int, std::vector<float>> prev_center;
  Rect entrance = {0,
                   static_cast<float>(video_height) / 2,
                   static_cast<float>(video_width),
                   static_cast<float>(video_height) / 2};
  double times;
  double total_time;
  // Capture all frames and do inference
  cv::Mat frame;
  int frame_id = 0;

  std::vector<std::string> records;
  std::vector<std::string> flow_records;
  records.push_back("result format: frame_id, track_id, x1, y1, w, h\n");

  LOG(INFO) << "------------------- Predict info ------------------------";
  while (capture.read(frame)) {
    if (frame.empty()) {
      break;
    }
    std::vector<cv::Mat> imgs;
    imgs.push_back(frame);
    jde_sct_->Predict(imgs, threshold_, &result, &det_times);
    frame_id += 1;
    total_time = std::accumulate(det_times.begin(), det_times.end(), 0.);
    times = total_time / frame_id;

    LOG(INFO) << "frame_id: " << frame_id
              << " predict time(s): " << total_time / 1000;

    cv::Mat out_img = PaddleDetection::VisualizeTrackResult(
        frame, result, 1000. / times, frame_id);

    // TODO(qianhui): the entrance line can be set by users
    PaddleDetection::FlowStatistic(result,
                                   frame_id,
                                   secs_interval_,
                                   do_entrance_counting_,
                                   video_fps,
                                   entrance,
                                   &id_set,
                                   &interval_id_set,
                                   &in_id_list,
                                   &out_id_list,
                                   &prev_center,
                                   &flow_records);

    if (save_result_) {
      PaddleDetection::SaveMOTResult(result, frame_id, &records);
    }

    // Draw the entrance line
    if (do_entrance_counting_) {
      float line_thickness = std::max(1, static_cast<int>(video_width / 500.));
      cv::Point pt1 = cv::Point(entrance.left, entrance.top);
      cv::Point pt2 = cv::Point(entrance.right, entrance.bottom);
      cv::line(out_img, pt1, pt2, cv::Scalar(0, 255, 255), line_thickness);
    }
    video_out.write(out_img);
  }
  capture.release();
  video_out.release();
  PrintBenchmarkLog(det_times, frame_id);
  LOG(INFO) << "-------------------- Final Output info -------------------";
  LOG(INFO) << "Total frame: " << frame_id;
  LOG(INFO) << "Visualized output saved as " << video_out_path.c_str();
  if (save_result_) {
    FILE* fp;

    std::string result_output_path =
        output_dir_ + OS_PATH_SEP + "mot_output.txt";
    if ((fp = fopen(result_output_path.c_str(), "w+")) == NULL) {
      printf("Open %s error.\n", result_output_path.c_str());
      return;
    }
    for (int l; l < records.size(); ++l) {
      fprintf(fp, records[l].c_str());
    }

    fclose(fp);
    LOG(INFO) << "txt result output saved as " << result_output_path.c_str();

    result_output_path = output_dir_ + OS_PATH_SEP + "flow_statistic.txt";
    if ((fp = fopen(result_output_path.c_str(), "w+")) == NULL) {
      printf("Open %s error.\n", result_output_path);
      return;
    }
    for (int l; l < flow_records.size(); ++l) {
      fprintf(fp, flow_records[l].c_str());
    }
    fclose(fp);
    LOG(INFO) << "txt flow statistic saved as " << result_output_path.c_str();
  }
}

void Pipeline::PredictMTMCT(const std::vector<std::string> video_path) {
  throw "Not Implement!";
}

void Pipeline::RunMOTStream(const cv::Mat img,
                            const int frame_id,
                            const int video_fps,
                            const Rect entrance,
                            cv::Mat out_img,
                            std::vector<std::string>* records,
                            std::set<int>* id_set,
                            std::set<int>* interval_id_set,
                            std::vector<int>* in_id_list,
                            std::vector<int>* out_id_list,
                            std::map<int, std::vector<float>>* prev_center,
                            std::vector<std::string>* flow_records) {
  PaddleDetection::MOTResult result;
  std::vector<double> det_times(3);
  double times;
  double total_time;

  LOG(INFO) << "------------------- Predict info ------------------------";
  std::vector<cv::Mat> imgs;
  imgs.push_back(img);
  jde_sct_->Predict(imgs, threshold_, &result, &det_times);
  total_time = std::accumulate(det_times.begin(), det_times.end(), 0.);
  times = total_time / frame_id;

  LOG(INFO) << "frame_id: " << frame_id
            << " predict time(s): " << total_time / 1000;

  out_img = PaddleDetection::VisualizeTrackResult(
      img, result, 1000. / times, frame_id);

  // Count total number
  // Count in & out number
  PaddleDetection::FlowStatistic(result,
                                 frame_id,
                                 secs_interval_,
                                 do_entrance_counting_,
                                 video_fps,
                                 entrance,
                                 id_set,
                                 interval_id_set,
                                 in_id_list,
                                 out_id_list,
                                 prev_center,
                                 flow_records);

  PrintBenchmarkLog(det_times, frame_id);
  if (save_result_) {
    PaddleDetection::SaveMOTResult(result, frame_id, records);
  }
}

void Pipeline::RunMTMCTStream(const std::vector<cv::Mat> imgs,
                              std::vector<std::string>* records) {
  throw "Not Implement!";
}

void Pipeline::PrintBenchmarkLog(const std::vector<double> det_time,
                                 const int img_num) {
  LOG(INFO) << "----------------------- Config info -----------------------";
  LOG(INFO) << "runtime_device: " << device_;
  LOG(INFO) << "ir_optim: "
            << "True";
  LOG(INFO) << "enable_memory_optim: "
            << "True";
  int has_trt = run_mode_.find("trt");
  if (has_trt >= 0) {
    LOG(INFO) << "enable_tensorrt: "
              << "True";
    std::string precision = run_mode_.substr(4, 8);
    LOG(INFO) << "precision: " << precision;
  } else {
    LOG(INFO) << "enable_tensorrt: "
              << "False";
    LOG(INFO) << "precision: "
              << "fp32";
  }
  LOG(INFO) << "enable_mkldnn: " << (use_mkldnn_ ? "True" : "False");
  LOG(INFO) << "cpu_math_library_num_threads: " << cpu_threads_;
  LOG(INFO) << "----------------------- Perf info ------------------------";
  LOG(INFO) << "Total number of predicted data: " << img_num
            << " and total time spent(s): "
            << std::accumulate(det_time.begin(), det_time.end(), 0.) / 1000;
  int num = std::max(1, img_num);
  LOG(INFO) << "preproce_time(ms): " << det_time[0] / num
            << ", inference_time(ms): " << det_time[1] / num
            << ", postprocess_time(ms): " << det_time[2] / num;
}

}  // namespace PaddleDetection
