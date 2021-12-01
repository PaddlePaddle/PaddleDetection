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
#include "include/jde_predictor.h"

using namespace paddle_infer;  // NOLINT

namespace PaddleDetection {

// Load Model and create model predictor
void JDEPredictor::LoadModel(const std::string& model_dir,
                             const std::string& run_mode) {
  paddle_infer::Config config;
  std::string prog_file = model_dir + OS_PATH_SEP + "model.pdmodel";
  std::string params_file = model_dir + OS_PATH_SEP + "model.pdiparams";
  config.SetModel(prog_file, params_file);
  if (this->device_ == "GPU") {
    config.EnableUseGpu(200, this->gpu_id_);
    config.SwitchIrOptim(true);
    // use tensorrt
    if (run_mode != "fluid") {
      auto precision = paddle_infer::Config::Precision::kFloat32;
      if (run_mode == "trt_fp32") {
        precision = paddle_infer::Config::Precision::kFloat32;
      } else if (run_mode == "trt_fp16") {
        precision = paddle_infer::Config::Precision::kHalf;
      } else if (run_mode == "trt_int8") {
        precision = paddle_infer::Config::Precision::kInt8;
      } else {
        printf(
            "run_mode should be 'fluid', 'trt_fp32', 'trt_fp16' or 'trt_int8'");
      }
      // set tensorrt
      config.EnableTensorRtEngine(1 << 30,
                                  1,
                                  this->min_subgraph_size_,
                                  precision,
                                  false,
                                  this->trt_calib_mode_);
    }
  } else if (this->device_ == "XPU") {
    config.EnableXpu(10 * 1024 * 1024);
  } else {
    config.DisableGpu();
    if (this->use_mkldnn_) {
      config.EnableMKLDNN();
      // cache 10 different shapes for mkldnn to avoid memory leak
      config.SetMkldnnCacheCapacity(10);
    }
    config.SetCpuMathLibraryNumThreads(this->cpu_math_library_num_threads_);
  }
  config.SwitchUseFeedFetchOps(false);
  config.SwitchIrOptim(true);
  config.DisableGlogInfo();
  // Memory optimization
  config.EnableMemoryOptim();
  predictor_ = std::move(CreatePredictor(config));
}

void FilterDets(const float conf_thresh,
                const cv::Mat dets,
                std::vector<int>* index) {
  for (int i = 0; i < dets.rows; ++i) {
    float score = *dets.ptr<float>(i, 4);
    if (score > conf_thresh) {
      index->push_back(i);
    }
  }
}

void JDEPredictor::Preprocess(const cv::Mat& ori_im) {
  // Clone the image : keep the original mat for postprocess
  cv::Mat im = ori_im.clone();
  preprocessor_.Run(&im, &inputs_);
}

void JDEPredictor::Postprocess(const cv::Mat dets,
                               const cv::Mat emb,
                               MOTResult* result) {
  result->clear();
  std::vector<Track> tracks;
  std::vector<int> valid;
  FilterDets(conf_thresh_, dets, &valid);
  cv::Mat new_dets, new_emb;
  for (int i = 0; i < valid.size(); ++i) {
    new_dets.push_back(dets.row(valid[i]));
    new_emb.push_back(emb.row(valid[i]));
  }
  JDETracker::instance()->update(new_dets, new_emb, &tracks);
  if (tracks.size() == 0) {
    MOTTrack mot_track;
    Rect ret = {*dets.ptr<float>(0, 0),
                *dets.ptr<float>(0, 1),
                *dets.ptr<float>(0, 2),
                *dets.ptr<float>(0, 3)};
    mot_track.ids = 1;
    mot_track.score = *dets.ptr<float>(0, 4);
    mot_track.rects = ret;
    result->push_back(mot_track);
  } else {
    std::vector<Track>::iterator titer;
    for (titer = tracks.begin(); titer != tracks.end(); ++titer) {
      if (titer->score < threshold_) {
        continue;
      } else {
        float w = titer->ltrb[2] - titer->ltrb[0];
        float h = titer->ltrb[3] - titer->ltrb[1];
        bool vertical = w / h > 1.6;
        float area = w * h;
        if (area > min_box_area_ && !vertical) {
          MOTTrack mot_track;
          Rect ret = {
              titer->ltrb[0], titer->ltrb[1], titer->ltrb[2], titer->ltrb[3]};
          mot_track.rects = ret;
          mot_track.score = titer->score;
          mot_track.ids = titer->id;
          result->push_back(mot_track);
        }
      }
    }
  }
}

void JDEPredictor::Predict(const std::vector<cv::Mat> imgs,
                           const double threshold,
                           MOTResult* result,
                           std::vector<double>* times) {
  auto preprocess_start = std::chrono::steady_clock::now();
  int batch_size = imgs.size();

  // in_data_batch
  std::vector<float> in_data_all;
  std::vector<float> im_shape_all(batch_size * 2);
  std::vector<float> scale_factor_all(batch_size * 2);

  // Preprocess image
  for (int bs_idx = 0; bs_idx < batch_size; bs_idx++) {
    cv::Mat im = imgs.at(bs_idx);
    Preprocess(im);
    im_shape_all[bs_idx * 2] = inputs_.im_shape_[0];
    im_shape_all[bs_idx * 2 + 1] = inputs_.im_shape_[1];

    scale_factor_all[bs_idx * 2] = inputs_.scale_factor_[0];
    scale_factor_all[bs_idx * 2 + 1] = inputs_.scale_factor_[1];

    in_data_all.insert(
        in_data_all.end(), inputs_.im_data_.begin(), inputs_.im_data_.end());
  }

  // Prepare input tensor
  auto input_names = predictor_->GetInputNames();
  for (const auto& tensor_name : input_names) {
    auto in_tensor = predictor_->GetInputHandle(tensor_name);
    if (tensor_name == "image") {
      int rh = inputs_.in_net_shape_[0];
      int rw = inputs_.in_net_shape_[1];
      in_tensor->Reshape({batch_size, 3, rh, rw});
      in_tensor->CopyFromCpu(in_data_all.data());
    } else if (tensor_name == "im_shape") {
      in_tensor->Reshape({batch_size, 2});
      in_tensor->CopyFromCpu(im_shape_all.data());
    } else if (tensor_name == "scale_factor") {
      in_tensor->Reshape({batch_size, 2});
      in_tensor->CopyFromCpu(scale_factor_all.data());
    }
  }

  auto preprocess_end = std::chrono::steady_clock::now();
  std::vector<int> bbox_shape;
  std::vector<int> emb_shape;

  // Run predictor
  auto inference_start = std::chrono::steady_clock::now();
  predictor_->Run();
  // Get output tensor
  auto output_names = predictor_->GetOutputNames();
  auto bbox_tensor = predictor_->GetOutputHandle(output_names[0]);
  bbox_shape = bbox_tensor->shape();
  auto emb_tensor = predictor_->GetOutputHandle(output_names[1]);
  emb_shape = emb_tensor->shape();
  // Calculate bbox length
  int bbox_size = 1;
  for (int j = 0; j < bbox_shape.size(); ++j) {
    bbox_size *= bbox_shape[j];
  }
  // Calculate emb length
  int emb_size = 1;
  for (int j = 0; j < emb_shape.size(); ++j) {
    emb_size *= emb_shape[j];
  }

  bbox_data_.resize(bbox_size);
  bbox_tensor->CopyToCpu(bbox_data_.data());

  emb_data_.resize(emb_size);
  emb_tensor->CopyToCpu(emb_data_.data());
  auto inference_end = std::chrono::steady_clock::now();

  // Postprocessing result
  auto postprocess_start = std::chrono::steady_clock::now();
  result->clear();

  cv::Mat dets(bbox_shape[0], 6, CV_32FC1, bbox_data_.data());
  cv::Mat emb(bbox_shape[0], emb_shape[1], CV_32FC1, emb_data_.data());

  Postprocess(dets, emb, result);

  auto postprocess_end = std::chrono::steady_clock::now();

  std::chrono::duration<float> preprocess_diff =
      preprocess_end - preprocess_start;
  (*times)[0] += static_cast<double>(preprocess_diff.count() * 1000);
  std::chrono::duration<float> inference_diff = inference_end - inference_start;
  (*times)[1] += static_cast<double>(inference_diff.count() * 1000);
  std::chrono::duration<float> postprocess_diff =
      postprocess_end - postprocess_start;
  (*times)[2] += static_cast<double>(postprocess_diff.count() * 1000);
}

}  // namespace PaddleDetection
