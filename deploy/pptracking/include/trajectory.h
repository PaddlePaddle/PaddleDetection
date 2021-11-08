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
// https://github.com/CnybTseng/JDE/blob/master/platforms/common/trajectory.h
// Ths copyright of CnybTseng/JDE is as follows:
// MIT License

#pragma once

#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/video/tracking.hpp"

namespace PaddleDetection {

typedef enum { New = 0, Tracked = 1, Lost = 2, Removed = 3 } TrajectoryState;

class Trajectory;
typedef std::vector<Trajectory> TrajectoryPool;
typedef std::vector<Trajectory>::iterator TrajectoryPoolIterator;
typedef std::vector<Trajectory *> TrajectoryPtrPool;
typedef std::vector<Trajectory *>::iterator TrajectoryPtrPoolIterator;

class TKalmanFilter : public cv::KalmanFilter {
 public:
  TKalmanFilter(void);
  virtual ~TKalmanFilter(void) {}
  virtual void init(const cv::Mat &measurement);
  virtual const cv::Mat &predict();
  virtual const cv::Mat &correct(const cv::Mat &measurement);
  virtual void project(cv::Mat *mean, cv::Mat *covariance) const;

 private:
  float std_weight_position;
  float std_weight_velocity;
};

inline TKalmanFilter::TKalmanFilter(void) : cv::KalmanFilter(8, 4) {
  cv::KalmanFilter::transitionMatrix = cv::Mat::eye(8, 8, CV_32F);
  for (int i = 0; i < 4; ++i)
    cv::KalmanFilter::transitionMatrix.at<float>(i, i + 4) = 1;
  cv::KalmanFilter::measurementMatrix = cv::Mat::eye(4, 8, CV_32F);
  std_weight_position = 1 / 20.f;
  std_weight_velocity = 1 / 160.f;
}

class Trajectory : public TKalmanFilter {
 public:
  Trajectory();
  Trajectory(const cv::Vec4f &ltrb, float score, const cv::Mat &embedding);
  Trajectory(const Trajectory &other);
  Trajectory &operator=(const Trajectory &rhs);
  virtual ~Trajectory(void) {}

  static int next_id();
  virtual const cv::Mat &predict(void);
  virtual void update(Trajectory *traj,
                      int timestamp,
                      bool update_embedding = true);
  virtual void activate(int timestamp);
  virtual void reactivate(Trajectory *traj, int timestamp, bool newid = false);
  virtual void mark_lost(void);
  virtual void mark_removed(void);

  friend TrajectoryPool operator+(const TrajectoryPool &a,
                                  const TrajectoryPool &b);
  friend TrajectoryPool operator+(const TrajectoryPool &a,
                                  const TrajectoryPtrPool &b);
  friend TrajectoryPool &operator+=(TrajectoryPool &a,  // NOLINT
                                    const TrajectoryPtrPool &b);
  friend TrajectoryPool operator-(const TrajectoryPool &a,
                                  const TrajectoryPool &b);
  friend TrajectoryPool &operator-=(TrajectoryPool &a,  // NOLINT
                                    const TrajectoryPool &b);
  friend TrajectoryPtrPool operator+(const TrajectoryPtrPool &a,
                                     const TrajectoryPtrPool &b);
  friend TrajectoryPtrPool operator+(const TrajectoryPtrPool &a,
                                     TrajectoryPool *b);
  friend TrajectoryPtrPool operator-(const TrajectoryPtrPool &a,
                                     const TrajectoryPtrPool &b);

  friend cv::Mat embedding_distance(const TrajectoryPool &a,
                                    const TrajectoryPool &b);
  friend cv::Mat embedding_distance(const TrajectoryPtrPool &a,
                                    const TrajectoryPtrPool &b);
  friend cv::Mat embedding_distance(const TrajectoryPtrPool &a,
                                    const TrajectoryPool &b);

  friend cv::Mat mahalanobis_distance(const TrajectoryPool &a,
                                      const TrajectoryPool &b);
  friend cv::Mat mahalanobis_distance(const TrajectoryPtrPool &a,
                                      const TrajectoryPtrPool &b);
  friend cv::Mat mahalanobis_distance(const TrajectoryPtrPool &a,
                                      const TrajectoryPool &b);

  friend cv::Mat iou_distance(const TrajectoryPool &a, const TrajectoryPool &b);
  friend cv::Mat iou_distance(const TrajectoryPtrPool &a,
                              const TrajectoryPtrPool &b);
  friend cv::Mat iou_distance(const TrajectoryPtrPool &a,
                              const TrajectoryPool &b);

 private:
  void update_embedding(const cv::Mat &embedding);

 public:
  TrajectoryState state;
  cv::Vec4f ltrb;
  cv::Mat smooth_embedding;
  int id;
  bool is_activated;
  int timestamp;
  int starttime;
  float score;

 private:
  static int count;
  cv::Vec4f xyah;
  cv::Mat current_embedding;
  float eta;
  int length;
};

inline cv::Vec4f ltrb2xyah(const cv::Vec4f &ltrb) {
  cv::Vec4f xyah;
  xyah[0] = (ltrb[0] + ltrb[2]) * 0.5f;
  xyah[1] = (ltrb[1] + ltrb[3]) * 0.5f;
  xyah[3] = ltrb[3] - ltrb[1];
  xyah[2] = (ltrb[2] - ltrb[0]) / xyah[3];
  return xyah;
}

inline Trajectory::Trajectory()
    : state(New),
      ltrb(cv::Vec4f()),
      smooth_embedding(cv::Mat()),
      id(0),
      is_activated(false),
      timestamp(0),
      starttime(0),
      score(0),
      eta(0.9),
      length(0) {}

inline Trajectory::Trajectory(const cv::Vec4f &ltrb_,
                              float score_,
                              const cv::Mat &embedding)
    : state(New),
      ltrb(ltrb_),
      smooth_embedding(cv::Mat()),
      id(0),
      is_activated(false),
      timestamp(0),
      starttime(0),
      score(score_),
      eta(0.9),
      length(0) {
  xyah = ltrb2xyah(ltrb);
  update_embedding(embedding);
}

inline Trajectory::Trajectory(const Trajectory &other)
    : state(other.state),
      ltrb(other.ltrb),
      id(other.id),
      is_activated(other.is_activated),
      timestamp(other.timestamp),
      starttime(other.starttime),
      xyah(other.xyah),
      score(other.score),
      eta(other.eta),
      length(other.length) {
  other.smooth_embedding.copyTo(smooth_embedding);
  other.current_embedding.copyTo(current_embedding);
  // copy state in KalmanFilter

  other.statePre.copyTo(cv::KalmanFilter::statePre);
  other.statePost.copyTo(cv::KalmanFilter::statePost);
  other.errorCovPre.copyTo(cv::KalmanFilter::errorCovPre);
  other.errorCovPost.copyTo(cv::KalmanFilter::errorCovPost);
}

inline Trajectory &Trajectory::operator=(const Trajectory &rhs) {
  this->state = rhs.state;
  this->ltrb = rhs.ltrb;
  rhs.smooth_embedding.copyTo(this->smooth_embedding);
  this->id = rhs.id;
  this->is_activated = rhs.is_activated;
  this->timestamp = rhs.timestamp;
  this->starttime = rhs.starttime;
  this->xyah = rhs.xyah;
  this->score = rhs.score;
  rhs.current_embedding.copyTo(this->current_embedding);
  this->eta = rhs.eta;
  this->length = rhs.length;

  // copy state in KalmanFilter

  rhs.statePre.copyTo(cv::KalmanFilter::statePre);
  rhs.statePost.copyTo(cv::KalmanFilter::statePost);
  rhs.errorCovPre.copyTo(cv::KalmanFilter::errorCovPre);
  rhs.errorCovPost.copyTo(cv::KalmanFilter::errorCovPost);

  return *this;
}

inline int Trajectory::next_id() {
  ++count;
  return count;
}

inline void Trajectory::mark_lost(void) { state = Lost; }

inline void Trajectory::mark_removed(void) { state = Removed; }

}  // namespace PaddleDetection
