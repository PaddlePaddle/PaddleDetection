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

// The code is based on:
// https://github.com/CnybTseng/JDE/blob/master/platforms/common/jdetracker.h
// Ths copyright of CnybTseng/JDE is as follows:
// MIT License

#pragma once

#include <map>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "include/trajectory.h"

namespace PaddleDetection {

typedef std::map<int, int> Match;
typedef std::map<int, int>::iterator MatchIterator;

struct Track {
  int id;
  float score;
  cv::Vec4f ltrb;
};

class JDETracker {
 public:
  static JDETracker *instance(void);
  virtual bool update(const cv::Mat &dets,
                      const cv::Mat &emb,
                      std::vector<Track> *tracks);

 private:
  JDETracker(void);
  virtual ~JDETracker(void) {}
  cv::Mat motion_distance(const TrajectoryPtrPool &a, const TrajectoryPool &b);
  void linear_assignment(const cv::Mat &cost,
                         float cost_limit,
                         Match *matches,
                         std::vector<int> *mismatch_row,
                         std::vector<int> *mismatch_col);
  void remove_duplicate_trajectory(TrajectoryPool *a,
                                   TrajectoryPool *b,
                                   float iou_thresh = 0.15f);

 private:
  static JDETracker *me;
  int timestamp;
  TrajectoryPool tracked_trajectories;
  TrajectoryPool lost_trajectories;
  TrajectoryPool removed_trajectories;
  int max_lost_time;
  float lambda;
  float det_thresh;
};

}  // namespace PaddleDetection
