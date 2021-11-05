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
#include <iomanip>
#include <chrono>
#include "include/postprocess.h"

namespace PaddleDetection {

cv::Scalar GetColor(int idx) {
  idx = idx * 3;
  cv::Scalar color = cv::Scalar((37 * idx) % 255, 
                                (17 * idx) % 255, 
                                (29 * idx) % 255);
  return color;
}

cv::Mat VisualizeTrackResult(const cv::Mat& img,
                        const MOTResult& results,
                        const float fps, const int frame_id) {
  cv::Mat vis_img = img.clone();
  int im_h = img.rows;
  int im_w = img.cols;
  float text_scale = std::max(1, int(im_w / 1600.));
  float text_thickness = 2.;
  float line_thickness = std::max(1, int(im_w / 500.));

  std::ostringstream oss;
  oss << std::setiosflags(std::ios::fixed) << std::setprecision(4);
  oss << "frame: " << frame_id<<" ";
  oss << "fps: " << fps<<" ";
  oss << "num: " << results.size();
  std::string text = oss.str();

  cv::Point origin;
  origin.x = 0;
  origin.y = int(15 * text_scale);
  cv::putText(
        vis_img,
        text,
        origin,
        cv::FONT_HERSHEY_PLAIN,
        text_scale, (0, 0, 255), 2);

  for (int i = 0; i < results.size(); ++i) {
    const int obj_id = results[i].ids;
    const float score = results[i].score;
    
    cv::Scalar color = GetColor(obj_id);

    cv::Point pt1 = cv::Point(results[i].rects.left, results[i].rects.top);
    cv::Point pt2 = cv::Point(results[i].rects.right, results[i].rects.bottom);
    cv::Point id_pt = cv::Point(results[i].rects.left, results[i].rects.top + 10);
    cv::Point score_pt = cv::Point(results[i].rects.left, results[i].rects.top - 10);
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

void FlowStatistic(const MOTResult& results, const int frame_id,
                   std::vector<int>* count_list, 
                   std::vector<int>* in_count_list, 
                   std::vector<int>* out_count_list) {
  throw "Not Implement";
}

void SaveMOTResult(const MOTResult& results, const int frame_id, std::vector<std::string>& records) {
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
    os << frame_id << " " << ids << ""
       << x1 << " " << y1 << " "
       << w << " " << h <<"\n";
    record = os.str();
    records.push_back(record);
  }
}

} // namespace PaddleDetection
