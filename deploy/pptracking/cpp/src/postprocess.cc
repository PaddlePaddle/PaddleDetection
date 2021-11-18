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
#include "include/postprocess.h"

namespace PaddleDetection {

cv::Scalar GetColor(int idx) {
  idx = idx * 3;
  cv::Scalar color =
      cv::Scalar((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255);
  return color;
}

cv::Mat VisualizeTrackResult(const cv::Mat& img,
                             const MOTResult& results,
                             const float fps,
                             const int frame_id) {
  cv::Mat vis_img = img.clone();
  int im_h = img.rows;
  int im_w = img.cols;
  float text_scale = std::max(1, static_cast<int>(im_w / 1600.));
  float text_thickness = 2.;
  float line_thickness = std::max(1, static_cast<int>(im_w / 500.));

  std::ostringstream oss;
  oss << std::setiosflags(std::ios::fixed) << std::setprecision(4);
  oss << "frame: " << frame_id << " ";
  oss << "fps: " << fps << " ";
  oss << "num: " << results.size();
  std::string text = oss.str();

  cv::Point origin;
  origin.x = 0;
  origin.y = static_cast<int>(15 * text_scale);
  cv::putText(vis_img,
              text,
              origin,
              cv::FONT_HERSHEY_PLAIN,
              text_scale,
              (0, 0, 255),
              2);

  for (int i = 0; i < results.size(); ++i) {
    const int obj_id = results[i].ids;
    const float score = results[i].score;

    cv::Scalar color = GetColor(obj_id);

    cv::Point pt1 = cv::Point(results[i].rects.left, results[i].rects.top);
    cv::Point pt2 = cv::Point(results[i].rects.right, results[i].rects.bottom);
    cv::Point id_pt =
        cv::Point(results[i].rects.left, results[i].rects.top + 10);
    cv::Point score_pt =
        cv::Point(results[i].rects.left, results[i].rects.top - 10);
    cv::rectangle(vis_img, pt1, pt2, color, line_thickness);

    std::ostringstream idoss;
    idoss << std::setiosflags(std::ios::fixed) << std::setprecision(4);
    idoss << obj_id;
    std::string id_text = idoss.str();

    cv::putText(vis_img,
                id_text,
                id_pt,
                cv::FONT_HERSHEY_PLAIN,
                text_scale,
                cv::Scalar(0, 255, 255),
                text_thickness);

    std::ostringstream soss;
    soss << std::setiosflags(std::ios::fixed) << std::setprecision(2);
    soss << score;
    std::string score_text = soss.str();

    cv::putText(vis_img,
                score_text,
                score_pt,
                cv::FONT_HERSHEY_PLAIN,
                text_scale,
                cv::Scalar(0, 255, 255),
                text_thickness);
  }
  return vis_img;
}

void FlowStatistic(const MOTResult& results,
                   const int frame_id,
                   const int secs_interval,
                   const bool do_entrance_counting,
                   const int video_fps,
                   const Rect entrance,
                   std::set<int>* id_set,
                   std::set<int>* interval_id_set,
                   std::vector<int>* in_id_list,
                   std::vector<int>* out_id_list,
                   std::map<int, std::vector<float>>* prev_center,
                   std::vector<std::string>* records) {
  if (frame_id == 0) interval_id_set->clear();

  if (do_entrance_counting) {
    // Count in and out number:
    // Use horizontal center line as the entrance just for simplification.
    // If a person located in the above the horizontal center line
    // at the previous frame and is in the below the line at the current frame,
    // the in number is increased by one.
    // If a person was in the below the horizontal center line
    // at the previous frame and locates in the below the line at the current
    // frame,
    // the out number is increased by one.
    // TODO(qianhui): if the entrance is not the horizontal center line,
    // the counting method should be optimized.

    float entrance_y = entrance.top;
    for (const auto& result : results) {
      float center_x = (result.rects.left + result.rects.right) / 2;
      float center_y = (result.rects.top + result.rects.bottom) / 2;
      int ids = result.ids;
      std::map<int, std::vector<float>>::iterator iter;
      iter = prev_center->find(ids);
      if (iter != prev_center->end()) {
        if (iter->second[1] <= entrance_y && center_y > entrance_y) {
          in_id_list->push_back(ids);
        }
        if (iter->second[1] >= entrance_y && center_y < entrance_y) {
          out_id_list->push_back(ids);
        }
        (*prev_center)[ids][0] = center_x;
        (*prev_center)[ids][1] = center_y;
      } else {
        prev_center->insert(
            std::pair<int, std::vector<float>>(ids, {center_x, center_y}));
      }
    }
  }

  // Count totol number, number at a manual-setting interval
  for (const auto& result : results) {
    id_set->insert(result.ids);
    interval_id_set->insert(result.ids);
  }

  std::ostringstream os;
  os << "Frame id: " << frame_id << ", Total count: " << id_set->size();
  if (do_entrance_counting) {
    os << ", In count: " << in_id_list->size()
       << ", Out count: " << out_id_list->size();
  }

  // Reset counting at the interval beginning
  int curr_interval_count = -1;
  if (frame_id % video_fps == 0 && frame_id / video_fps % secs_interval == 0) {
    curr_interval_count = interval_id_set->size();
    os << ", Count during " << secs_interval
       << " secs: " << curr_interval_count;
    interval_id_set->clear();
  }
  os << "\n";
  std::string record = os.str();
  records->push_back(record);
  LOG(INFO) << record;
}

void SaveMOTResult(const MOTResult& results,
                   const int frame_id,
                   std::vector<std::string>* records) {
  // result format: frame_id, track_id, x1, y1, w, h
  std::string record;
  for (int i = 0; i < results.size(); ++i) {
    MOTTrack mot_track = results[i];
    int ids = mot_track.ids;
    float score = mot_track.score;
    Rect rects = mot_track.rects;
    float x1 = rects.left;
    float y1 = rects.top;
    float x2 = rects.right;
    float y2 = rects.bottom;
    float w = x2 - x1;
    float h = y2 - y1;
    if (w == 0 || h == 0) {
      continue;
    }
    std::ostringstream os;
    os << frame_id << " " << ids << "" << x1 << " " << y1 << " " << w << " "
       << h << "\n";
    record = os.str();
    records->push_back(record);
  }
}

}  // namespace PaddleDetection
