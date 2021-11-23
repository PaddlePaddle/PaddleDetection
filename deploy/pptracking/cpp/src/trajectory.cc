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
// https://github.com/CnybTseng/JDE/blob/master/platforms/common/trajectory.cpp
// Ths copyright of CnybTseng/JDE is as follows:
// MIT License

#include "include/trajectory.h"
#include <algorithm>

namespace PaddleDetection {

void TKalmanFilter::init(const cv::Mat &measurement) {
  measurement.copyTo(statePost(cv::Rect(0, 0, 1, 4)));
  statePost(cv::Rect(0, 4, 1, 4)).setTo(0);
  statePost.copyTo(statePre);

  float varpos = 2 * std_weight_position * (*measurement.ptr<float>(3));
  varpos *= varpos;
  float varvel = 10 * std_weight_velocity * (*measurement.ptr<float>(3));
  varvel *= varvel;

  errorCovPost.setTo(0);
  *errorCovPost.ptr<float>(0, 0) = varpos;
  *errorCovPost.ptr<float>(1, 1) = varpos;
  *errorCovPost.ptr<float>(2, 2) = 1e-4f;
  *errorCovPost.ptr<float>(3, 3) = varpos;
  *errorCovPost.ptr<float>(4, 4) = varvel;
  *errorCovPost.ptr<float>(5, 5) = varvel;
  *errorCovPost.ptr<float>(6, 6) = 1e-10f;
  *errorCovPost.ptr<float>(7, 7) = varvel;
  errorCovPost.copyTo(errorCovPre);
}

const cv::Mat &TKalmanFilter::predict() {
  float varpos = std_weight_position * (*statePre.ptr<float>(3));
  varpos *= varpos;
  float varvel = std_weight_velocity * (*statePre.ptr<float>(3));
  varvel *= varvel;

  processNoiseCov.setTo(0);
  *processNoiseCov.ptr<float>(0, 0) = varpos;
  *processNoiseCov.ptr<float>(1, 1) = varpos;
  *processNoiseCov.ptr<float>(2, 2) = 1e-4f;
  *processNoiseCov.ptr<float>(3, 3) = varpos;
  *processNoiseCov.ptr<float>(4, 4) = varvel;
  *processNoiseCov.ptr<float>(5, 5) = varvel;
  *processNoiseCov.ptr<float>(6, 6) = 1e-10f;
  *processNoiseCov.ptr<float>(7, 7) = varvel;

  return cv::KalmanFilter::predict();
}

const cv::Mat &TKalmanFilter::correct(const cv::Mat &measurement) {
  float varpos = std_weight_position * (*measurement.ptr<float>(3));
  varpos *= varpos;

  measurementNoiseCov.setTo(0);
  *measurementNoiseCov.ptr<float>(0, 0) = varpos;
  *measurementNoiseCov.ptr<float>(1, 1) = varpos;
  *measurementNoiseCov.ptr<float>(2, 2) = 1e-2f;
  *measurementNoiseCov.ptr<float>(3, 3) = varpos;

  return cv::KalmanFilter::correct(measurement);
}

void TKalmanFilter::project(cv::Mat *mean, cv::Mat *covariance) const {
  float varpos = std_weight_position * (*statePost.ptr<float>(3));
  varpos *= varpos;

  cv::Mat measurementNoiseCov_ = cv::Mat::eye(4, 4, CV_32F);
  *measurementNoiseCov_.ptr<float>(0, 0) = varpos;
  *measurementNoiseCov_.ptr<float>(1, 1) = varpos;
  *measurementNoiseCov_.ptr<float>(2, 2) = 1e-2f;
  *measurementNoiseCov_.ptr<float>(3, 3) = varpos;

  *mean = measurementMatrix * statePost;
  cv::Mat temp = measurementMatrix * errorCovPost;
  gemm(temp,
       measurementMatrix,
       1,
       measurementNoiseCov_,
       1,
       *covariance,
       cv::GEMM_2_T);
}

int Trajectory::count = 0;

const cv::Mat &Trajectory::predict(void) {
  if (state != Tracked) *cv::KalmanFilter::statePost.ptr<float>(7) = 0;
  return TKalmanFilter::predict();
}

void Trajectory::update(Trajectory *traj,
                        int timestamp_,
                        bool update_embedding_) {
  timestamp = timestamp_;
  ++length;
  ltrb = traj->ltrb;
  xyah = traj->xyah;
  TKalmanFilter::correct(cv::Mat(traj->xyah));
  state = Tracked;
  is_activated = true;
  score = traj->score;
  if (update_embedding_) update_embedding(traj->current_embedding);
}

void Trajectory::activate(int timestamp_) {
  id = next_id();
  TKalmanFilter::init(cv::Mat(xyah));
  length = 0;
  state = Tracked;
  if (timestamp_ == 1) {
    is_activated = true;
  }
  timestamp = timestamp_;
  starttime = timestamp_;
}

void Trajectory::reactivate(Trajectory *traj, int timestamp_, bool newid) {
  TKalmanFilter::correct(cv::Mat(traj->xyah));
  update_embedding(traj->current_embedding);
  length = 0;
  state = Tracked;
  is_activated = true;
  timestamp = timestamp_;
  if (newid) id = next_id();
}

void Trajectory::update_embedding(const cv::Mat &embedding) {
  current_embedding = embedding / cv::norm(embedding);
  if (smooth_embedding.empty()) {
    smooth_embedding = current_embedding;
  } else {
    smooth_embedding = eta * smooth_embedding + (1 - eta) * current_embedding;
  }
  smooth_embedding = smooth_embedding / cv::norm(smooth_embedding);
}

TrajectoryPool operator+(const TrajectoryPool &a, const TrajectoryPool &b) {
  TrajectoryPool sum;
  sum.insert(sum.end(), a.begin(), a.end());

  std::vector<int> ids(a.size());
  for (size_t i = 0; i < a.size(); ++i) ids[i] = a[i].id;

  for (size_t i = 0; i < b.size(); ++i) {
    std::vector<int>::iterator iter = find(ids.begin(), ids.end(), b[i].id);
    if (iter == ids.end()) {
      sum.push_back(b[i]);
      ids.push_back(b[i].id);
    }
  }

  return sum;
}

TrajectoryPool operator+(const TrajectoryPool &a, const TrajectoryPtrPool &b) {
  TrajectoryPool sum;
  sum.insert(sum.end(), a.begin(), a.end());

  std::vector<int> ids(a.size());
  for (size_t i = 0; i < a.size(); ++i) ids[i] = a[i].id;

  for (size_t i = 0; i < b.size(); ++i) {
    std::vector<int>::iterator iter = find(ids.begin(), ids.end(), b[i]->id);
    if (iter == ids.end()) {
      sum.push_back(*b[i]);
      ids.push_back(b[i]->id);
    }
  }

  return sum;
}

TrajectoryPool &operator+=(TrajectoryPool &a,  // NOLINT
                           const TrajectoryPtrPool &b) {
  std::vector<int> ids(a.size());
  for (size_t i = 0; i < a.size(); ++i) ids[i] = a[i].id;

  for (size_t i = 0; i < b.size(); ++i) {
    if (b[i]->smooth_embedding.empty()) continue;
    std::vector<int>::iterator iter = find(ids.begin(), ids.end(), b[i]->id);
    if (iter == ids.end()) {
      a.push_back(*b[i]);
      ids.push_back(b[i]->id);
    }
  }

  return a;
}

TrajectoryPool operator-(const TrajectoryPool &a, const TrajectoryPool &b) {
  TrajectoryPool dif;
  std::vector<int> ids(b.size());
  for (size_t i = 0; i < b.size(); ++i) ids[i] = b[i].id;

  for (size_t i = 0; i < a.size(); ++i) {
    std::vector<int>::iterator iter = find(ids.begin(), ids.end(), a[i].id);
    if (iter == ids.end()) dif.push_back(a[i]);
  }

  return dif;
}

TrajectoryPool &operator-=(TrajectoryPool &a,  // NOLINT
                           const TrajectoryPool &b) {
  std::vector<int> ids(b.size());
  for (size_t i = 0; i < b.size(); ++i) ids[i] = b[i].id;

  TrajectoryPoolIterator piter;
  for (piter = a.begin(); piter != a.end();) {
    std::vector<int>::iterator iter = find(ids.begin(), ids.end(), piter->id);
    if (iter == ids.end())
      ++piter;
    else
      piter = a.erase(piter);
  }

  return a;
}

TrajectoryPtrPool operator+(const TrajectoryPtrPool &a,
                            const TrajectoryPtrPool &b) {
  TrajectoryPtrPool sum;
  sum.insert(sum.end(), a.begin(), a.end());

  std::vector<int> ids(a.size());
  for (size_t i = 0; i < a.size(); ++i) ids[i] = a[i]->id;

  for (size_t i = 0; i < b.size(); ++i) {
    std::vector<int>::iterator iter = find(ids.begin(), ids.end(), b[i]->id);
    if (iter == ids.end()) {
      sum.push_back(b[i]);
      ids.push_back(b[i]->id);
    }
  }

  return sum;
}

TrajectoryPtrPool operator+(const TrajectoryPtrPool &a, TrajectoryPool *b) {
  TrajectoryPtrPool sum;
  sum.insert(sum.end(), a.begin(), a.end());

  std::vector<int> ids(a.size());
  for (size_t i = 0; i < a.size(); ++i) ids[i] = a[i]->id;

  for (size_t i = 0; i < b->size(); ++i) {
    std::vector<int>::iterator iter = find(ids.begin(), ids.end(), (*b)[i].id);
    if (iter == ids.end()) {
      sum.push_back(&(*b)[i]);
      ids.push_back((*b)[i].id);
    }
  }

  return sum;
}

TrajectoryPtrPool operator-(const TrajectoryPtrPool &a,
                            const TrajectoryPtrPool &b) {
  TrajectoryPtrPool dif;
  std::vector<int> ids(b.size());
  for (size_t i = 0; i < b.size(); ++i) ids[i] = b[i]->id;

  for (size_t i = 0; i < a.size(); ++i) {
    std::vector<int>::iterator iter = find(ids.begin(), ids.end(), a[i]->id);
    if (iter == ids.end()) dif.push_back(a[i]);
  }

  return dif;
}

cv::Mat embedding_distance(const TrajectoryPool &a, const TrajectoryPool &b) {
  cv::Mat dists(a.size(), b.size(), CV_32F);
  for (size_t i = 0; i < a.size(); ++i) {
    float *distsi = dists.ptr<float>(i);
    for (size_t j = 0; j < b.size(); ++j) {
      cv::Mat u = a[i].smooth_embedding;
      cv::Mat v = b[j].smooth_embedding;
      double uv = u.dot(v);
      double uu = u.dot(u);
      double vv = v.dot(v);
      double dist = std::abs(1. - uv / std::sqrt(uu * vv));
      // double dist = cv::norm(a[i].smooth_embedding, b[j].smooth_embedding,
      // cv::NORM_L2);
      distsi[j] = static_cast<float>(std::max(std::min(dist, 2.), 0.));
    }
  }
  return dists;
}

cv::Mat embedding_distance(const TrajectoryPtrPool &a,
                           const TrajectoryPtrPool &b) {
  cv::Mat dists(a.size(), b.size(), CV_32F);
  for (size_t i = 0; i < a.size(); ++i) {
    float *distsi = dists.ptr<float>(i);
    for (size_t j = 0; j < b.size(); ++j) {
      // double dist = cv::norm(a[i]->smooth_embedding, b[j]->smooth_embedding,
      // cv::NORM_L2);
      // distsi[j] = static_cast<float>(dist);
      cv::Mat u = a[i]->smooth_embedding;
      cv::Mat v = b[j]->smooth_embedding;
      double uv = u.dot(v);
      double uu = u.dot(u);
      double vv = v.dot(v);
      double dist = std::abs(1. - uv / std::sqrt(uu * vv));
      distsi[j] = static_cast<float>(std::max(std::min(dist, 2.), 0.));
    }
  }

  return dists;
}

cv::Mat embedding_distance(const TrajectoryPtrPool &a,
                           const TrajectoryPool &b) {
  cv::Mat dists(a.size(), b.size(), CV_32F);
  for (size_t i = 0; i < a.size(); ++i) {
    float *distsi = dists.ptr<float>(i);
    for (size_t j = 0; j < b.size(); ++j) {
      // double dist = cv::norm(a[i]->smooth_embedding, b[j].smooth_embedding,
      // cv::NORM_L2);
      // distsi[j] = static_cast<float>(dist);
      cv::Mat u = a[i]->smooth_embedding;
      cv::Mat v = b[j].smooth_embedding;
      double uv = u.dot(v);
      double uu = u.dot(u);
      double vv = v.dot(v);
      double dist = std::abs(1. - uv / std::sqrt(uu * vv));
      distsi[j] = static_cast<float>(std::max(std::min(dist, 2.), 0.));
    }
  }

  return dists;
}

cv::Mat mahalanobis_distance(const TrajectoryPool &a, const TrajectoryPool &b) {
  std::vector<cv::Mat> means(a.size());
  std::vector<cv::Mat> icovariances(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    cv::Mat covariance;
    a[i].project(&means[i], &covariance);
    cv::invert(covariance, icovariances[i]);
  }

  cv::Mat dists(a.size(), b.size(), CV_32F);
  for (size_t i = 0; i < a.size(); ++i) {
    float *distsi = dists.ptr<float>(i);
    for (size_t j = 0; j < b.size(); ++j) {
      const cv::Mat x(b[j].xyah);
      float dist =
          static_cast<float>(cv::Mahalanobis(x, means[i], icovariances[i]));
      distsi[j] = dist * dist;
    }
  }

  return dists;
}

cv::Mat mahalanobis_distance(const TrajectoryPtrPool &a,
                             const TrajectoryPtrPool &b) {
  std::vector<cv::Mat> means(a.size());
  std::vector<cv::Mat> icovariances(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    cv::Mat covariance;
    a[i]->project(&means[i], &covariance);
    cv::invert(covariance, icovariances[i]);
  }

  cv::Mat dists(a.size(), b.size(), CV_32F);
  for (size_t i = 0; i < a.size(); ++i) {
    float *distsi = dists.ptr<float>(i);
    for (size_t j = 0; j < b.size(); ++j) {
      const cv::Mat x(b[j]->xyah);
      float dist =
          static_cast<float>(cv::Mahalanobis(x, means[i], icovariances[i]));
      distsi[j] = dist * dist;
    }
  }

  return dists;
}

cv::Mat mahalanobis_distance(const TrajectoryPtrPool &a,
                             const TrajectoryPool &b) {
  std::vector<cv::Mat> means(a.size());
  std::vector<cv::Mat> icovariances(a.size());

  for (size_t i = 0; i < a.size(); ++i) {
    cv::Mat covariance;
    a[i]->project(&means[i], &covariance);
    cv::invert(covariance, icovariances[i]);
  }

  cv::Mat dists(a.size(), b.size(), CV_32F);
  for (size_t i = 0; i < a.size(); ++i) {
    float *distsi = dists.ptr<float>(i);
    for (size_t j = 0; j < b.size(); ++j) {
      const cv::Mat x(b[j].xyah);
      float dist =
          static_cast<float>(cv::Mahalanobis(x, means[i], icovariances[i]));
      distsi[j] = dist * dist;
    }
  }

  return dists;
}

static inline float calc_inter_area(const cv::Vec4f &a, const cv::Vec4f &b) {
  if (a[2] < b[0] || a[0] > b[2] || a[3] < b[1] || a[1] > b[3]) return 0.f;

  float w = std::min(a[2], b[2]) - std::max(a[0], b[0]);
  float h = std::min(a[3], b[3]) - std::max(a[1], b[1]);
  return w * h;
}

cv::Mat iou_distance(const TrajectoryPool &a, const TrajectoryPool &b) {
  std::vector<float> areaa(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    float w = a[i].ltrb[2] - a[i].ltrb[0];
    float h = a[i].ltrb[3] - a[i].ltrb[1];
    areaa[i] = w * h;
  }

  std::vector<float> areab(b.size());
  for (size_t j = 0; j < b.size(); ++j) {
    float w = b[j].ltrb[2] - b[j].ltrb[0];
    float h = b[j].ltrb[3] - b[j].ltrb[1];
    areab[j] = w * h;
  }

  cv::Mat dists(a.size(), b.size(), CV_32F);
  for (size_t i = 0; i < a.size(); ++i) {
    const cv::Vec4f &boxa = a[i].ltrb;
    float *distsi = dists.ptr<float>(i);
    for (size_t j = 0; j < b.size(); ++j) {
      const cv::Vec4f &boxb = b[j].ltrb;
      float inters = calc_inter_area(boxa, boxb);
      distsi[j] = 1.f - inters / (areaa[i] + areab[j] - inters);
    }
  }

  return dists;
}

cv::Mat iou_distance(const TrajectoryPtrPool &a, const TrajectoryPtrPool &b) {
  std::vector<float> areaa(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    float w = a[i]->ltrb[2] - a[i]->ltrb[0];
    float h = a[i]->ltrb[3] - a[i]->ltrb[1];
    areaa[i] = w * h;
  }

  std::vector<float> areab(b.size());
  for (size_t j = 0; j < b.size(); ++j) {
    float w = b[j]->ltrb[2] - b[j]->ltrb[0];
    float h = b[j]->ltrb[3] - b[j]->ltrb[1];
    areab[j] = w * h;
  }

  cv::Mat dists(a.size(), b.size(), CV_32F);
  for (size_t i = 0; i < a.size(); ++i) {
    const cv::Vec4f &boxa = a[i]->ltrb;
    float *distsi = dists.ptr<float>(i);
    for (size_t j = 0; j < b.size(); ++j) {
      const cv::Vec4f &boxb = b[j]->ltrb;
      float inters = calc_inter_area(boxa, boxb);
      distsi[j] = 1.f - inters / (areaa[i] + areab[j] - inters);
    }
  }

  return dists;
}

cv::Mat iou_distance(const TrajectoryPtrPool &a, const TrajectoryPool &b) {
  std::vector<float> areaa(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    float w = a[i]->ltrb[2] - a[i]->ltrb[0];
    float h = a[i]->ltrb[3] - a[i]->ltrb[1];
    areaa[i] = w * h;
  }

  std::vector<float> areab(b.size());
  for (size_t j = 0; j < b.size(); ++j) {
    float w = b[j].ltrb[2] - b[j].ltrb[0];
    float h = b[j].ltrb[3] - b[j].ltrb[1];
    areab[j] = w * h;
  }

  cv::Mat dists(a.size(), b.size(), CV_32F);
  for (size_t i = 0; i < a.size(); ++i) {
    const cv::Vec4f &boxa = a[i]->ltrb;
    float *distsi = dists.ptr<float>(i);
    for (size_t j = 0; j < b.size(); ++j) {
      const cv::Vec4f &boxb = b[j].ltrb;
      float inters = calc_inter_area(boxa, boxb);
      distsi[j] = 1.f - inters / (areaa[i] + areab[j] - inters);
    }
  }

  return dists;
}

}  // namespace PaddleDetection
