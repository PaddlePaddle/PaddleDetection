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
// https://github.com/CnybTseng/JDE/blob/master/platforms/common/jdetracker.cpp
// Ths copyright of CnybTseng/JDE is as follows:
// MIT License

#include <limits.h>
#include <stdio.h>
#include <algorithm>
#include <map>

#include "include/lapjv.h"
#include "include/tracker.h"

#define mat2vec4f(m)             \
  cv::Vec4f(*m.ptr<float>(0, 0), \
            *m.ptr<float>(0, 1), \
            *m.ptr<float>(0, 2), \
            *m.ptr<float>(0, 3))

namespace PaddleDetection {

static std::map<int, float> chi2inv95 = {{1, 3.841459f},
                                         {2, 5.991465f},
                                         {3, 7.814728f},
                                         {4, 9.487729f},
                                         {5, 11.070498f},
                                         {6, 12.591587f},
                                         {7, 14.067140f},
                                         {8, 15.507313f},
                                         {9, 16.918978f}};

JDETracker *JDETracker::me = new JDETracker;

JDETracker *JDETracker::instance(void) { return me; }

JDETracker::JDETracker(void)
    : timestamp(0), max_lost_time(30), lambda(0.98f), det_thresh(0.3f) {}

bool JDETracker::update(const cv::Mat &dets,
                        const cv::Mat &emb,
                        std::vector<Track> *tracks) {
  ++timestamp;
  TrajectoryPool candidates(dets.rows);
  for (int i = 0; i < dets.rows; ++i) {
    float score = *dets.ptr<float>(i, 4);
    const cv::Mat &ltrb_ = dets(cv::Rect(0, i, 4, 1));
    cv::Vec4f ltrb = mat2vec4f(ltrb_);
    const cv::Mat &embedding = emb(cv::Rect(0, i, emb.cols, 1));
    candidates[i] = Trajectory(ltrb, score, embedding);
  }

  TrajectoryPtrPool tracked_trajectories;
  TrajectoryPtrPool unconfirmed_trajectories;
  for (size_t i = 0; i < this->tracked_trajectories.size(); ++i) {
    if (this->tracked_trajectories[i].is_activated)
      tracked_trajectories.push_back(&this->tracked_trajectories[i]);
    else
      unconfirmed_trajectories.push_back(&this->tracked_trajectories[i]);
  }

  TrajectoryPtrPool trajectory_pool =
      tracked_trajectories + &(this->lost_trajectories);

  for (size_t i = 0; i < trajectory_pool.size(); ++i)
    trajectory_pool[i]->predict();

  Match matches;
  std::vector<int> mismatch_row;
  std::vector<int> mismatch_col;

  cv::Mat cost = motion_distance(trajectory_pool, candidates);
  linear_assignment(cost, 0.7f, &matches, &mismatch_row, &mismatch_col);

  MatchIterator miter;
  TrajectoryPtrPool activated_trajectories;
  TrajectoryPtrPool retrieved_trajectories;

  for (miter = matches.begin(); miter != matches.end(); miter++) {
    Trajectory *pt = trajectory_pool[miter->first];
    Trajectory &ct = candidates[miter->second];
    if (pt->state == Tracked) {
      pt->update(&ct, timestamp);
      activated_trajectories.push_back(pt);
    } else {
      pt->reactivate(&ct, timestamp);
      retrieved_trajectories.push_back(pt);
    }
  }

  TrajectoryPtrPool next_candidates(mismatch_col.size());
  for (size_t i = 0; i < mismatch_col.size(); ++i)
    next_candidates[i] = &candidates[mismatch_col[i]];

  TrajectoryPtrPool next_trajectory_pool;
  for (size_t i = 0; i < mismatch_row.size(); ++i) {
    int j = mismatch_row[i];
    if (trajectory_pool[j]->state == Tracked)
      next_trajectory_pool.push_back(trajectory_pool[j]);
  }

  cost = iou_distance(next_trajectory_pool, next_candidates);
  linear_assignment(cost, 0.5f, &matches, &mismatch_row, &mismatch_col);

  for (miter = matches.begin(); miter != matches.end(); miter++) {
    Trajectory *pt = next_trajectory_pool[miter->first];
    Trajectory *ct = next_candidates[miter->second];
    if (pt->state == Tracked) {
      pt->update(ct, timestamp);
      activated_trajectories.push_back(pt);
    } else {
      pt->reactivate(ct, timestamp);
      retrieved_trajectories.push_back(pt);
    }
  }

  TrajectoryPtrPool lost_trajectories;
  for (size_t i = 0; i < mismatch_row.size(); ++i) {
    Trajectory *pt = next_trajectory_pool[mismatch_row[i]];
    if (pt->state != Lost) {
      pt->mark_lost();
      lost_trajectories.push_back(pt);
    }
  }

  TrajectoryPtrPool nnext_candidates(mismatch_col.size());
  for (size_t i = 0; i < mismatch_col.size(); ++i)
    nnext_candidates[i] = next_candidates[mismatch_col[i]];
  cost = iou_distance(unconfirmed_trajectories, nnext_candidates);
  linear_assignment(cost, 0.7f, &matches, &mismatch_row, &mismatch_col);

  for (miter = matches.begin(); miter != matches.end(); miter++) {
    unconfirmed_trajectories[miter->first]->update(
        nnext_candidates[miter->second], timestamp);
    activated_trajectories.push_back(unconfirmed_trajectories[miter->first]);
  }

  TrajectoryPtrPool removed_trajectories;

  for (size_t i = 0; i < mismatch_row.size(); ++i) {
    unconfirmed_trajectories[mismatch_row[i]]->mark_removed();
    removed_trajectories.push_back(unconfirmed_trajectories[mismatch_row[i]]);
  }

  for (size_t i = 0; i < mismatch_col.size(); ++i) {
    if (nnext_candidates[mismatch_col[i]]->score < det_thresh) continue;
    nnext_candidates[mismatch_col[i]]->activate(timestamp);
    activated_trajectories.push_back(nnext_candidates[mismatch_col[i]]);
  }

  for (size_t i = 0; i < this->lost_trajectories.size(); ++i) {
    Trajectory &lt = this->lost_trajectories[i];
    if (timestamp - lt.timestamp > max_lost_time) {
      lt.mark_removed();
      removed_trajectories.push_back(&lt);
    }
  }

  TrajectoryPoolIterator piter;
  for (piter = this->tracked_trajectories.begin();
       piter != this->tracked_trajectories.end();) {
    if (piter->state != Tracked)
      piter = this->tracked_trajectories.erase(piter);
    else
      ++piter;
  }

  this->tracked_trajectories += activated_trajectories;
  this->tracked_trajectories += retrieved_trajectories;

  this->lost_trajectories -= this->tracked_trajectories;
  this->lost_trajectories += lost_trajectories;
  this->lost_trajectories -= this->removed_trajectories;
  this->removed_trajectories += removed_trajectories;
  remove_duplicate_trajectory(&this->tracked_trajectories,
                              &this->lost_trajectories);

  tracks->clear();
  for (size_t i = 0; i < this->tracked_trajectories.size(); ++i) {
    if (this->tracked_trajectories[i].is_activated) {
      Track track = {this->tracked_trajectories[i].id,
                     this->tracked_trajectories[i].score,
                     this->tracked_trajectories[i].ltrb};
      tracks->push_back(track);
    }
  }
  return 0;
}

cv::Mat JDETracker::motion_distance(const TrajectoryPtrPool &a,
                                    const TrajectoryPool &b) {
  if (0 == a.size() || 0 == b.size())
    return cv::Mat(a.size(), b.size(), CV_32F);

  cv::Mat edists = embedding_distance(a, b);
  cv::Mat mdists = mahalanobis_distance(a, b);
  cv::Mat fdists = lambda * edists + (1 - lambda) * mdists;

  const float gate_thresh = chi2inv95[4];
  for (int i = 0; i < fdists.rows; ++i) {
    for (int j = 0; j < fdists.cols; ++j) {
      if (*mdists.ptr<float>(i, j) > gate_thresh)
        *fdists.ptr<float>(i, j) = FLT_MAX;
    }
  }

  return fdists;
}

void JDETracker::linear_assignment(const cv::Mat &cost,
                                   float cost_limit,
                                   Match *matches,
                                   std::vector<int> *mismatch_row,
                                   std::vector<int> *mismatch_col) {
  matches->clear();
  mismatch_row->clear();
  mismatch_col->clear();
  if (cost.empty()) {
    for (int i = 0; i < cost.rows; ++i) mismatch_row->push_back(i);
    for (int i = 0; i < cost.cols; ++i) mismatch_col->push_back(i);
    return;
  }

  float opt = 0;
  cv::Mat x(cost.rows, 1, CV_32S);
  cv::Mat y(cost.cols, 1, CV_32S);

  lapjv_internal(cost,
                 true,
                 cost_limit,
                 reinterpret_cast<int *>(x.data),
                 reinterpret_cast<int *>(y.data));

  for (int i = 0; i < x.rows; ++i) {
    int j = *x.ptr<int>(i);
    if (j >= 0)
      matches->insert({i, j});
    else
      mismatch_row->push_back(i);
  }

  for (int i = 0; i < y.rows; ++i) {
    int j = *y.ptr<int>(i);
    if (j < 0) mismatch_col->push_back(i);
  }

  return;
}

void JDETracker::remove_duplicate_trajectory(TrajectoryPool *a,
                                             TrajectoryPool *b,
                                             float iou_thresh) {
  if (a->size() == 0 || b->size() == 0) return;

  cv::Mat dist = iou_distance(*a, *b);
  cv::Mat mask = dist < iou_thresh;
  std::vector<cv::Point> idx;
  cv::findNonZero(mask, idx);

  std::vector<int> da;
  std::vector<int> db;
  for (size_t i = 0; i < idx.size(); ++i) {
    int ta = (*a)[idx[i].y].timestamp - (*a)[idx[i].y].starttime;
    int tb = (*b)[idx[i].x].timestamp - (*b)[idx[i].x].starttime;
    if (ta > tb)
      db.push_back(idx[i].x);
    else
      da.push_back(idx[i].y);
  }

  int id = 0;
  TrajectoryPoolIterator piter;
  for (piter = a->begin(); piter != a->end();) {
    std::vector<int>::iterator iter = find(da.begin(), da.end(), id++);
    if (iter != da.end())
      piter = a->erase(piter);
    else
      ++piter;
  }

  id = 0;
  for (piter = b->begin(); piter != b->end();) {
    std::vector<int>::iterator iter = find(db.begin(), db.end(), id++);
    if (iter != db.end())
      piter = b->erase(piter);
    else
      ++piter;
  }
}

}  // namespace PaddleDetection
