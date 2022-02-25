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
#include "include/keypoint_detector.h"

using namespace paddle_infer;

namespace PaddleDetection {

// Load Model and create model predictor
void KeyPointDetector::LoadModel(const std::string& model_dir,
                                 const int batch_size,
                                 const std::string& run_mode) {
  paddle_infer::Config config;
  std::string prog_file = model_dir + OS_PATH_SEP + "model.pdmodel";
  std::string params_file = model_dir + OS_PATH_SEP + "model.pdiparams";
  config.SetModel(prog_file, params_file);
  if (this->device_ == "GPU") {
    config.EnableUseGpu(200, this->gpu_id_);
    config.SwitchIrOptim(true);
    // use tensorrt
    if (run_mode != "paddle") {
      auto precision = paddle_infer::Config::Precision::kFloat32;
      if (run_mode == "trt_fp32") {
        precision = paddle_infer::Config::Precision::kFloat32;
      } else if (run_mode == "trt_fp16") {
        precision = paddle_infer::Config::Precision::kHalf;
      } else if (run_mode == "trt_int8") {
        precision = paddle_infer::Config::Precision::kInt8;
      } else {
        printf(
            "run_mode should be 'paddle', 'trt_fp32', 'trt_fp16' or "
            "'trt_int8'");
      }
      // set tensorrt
      config.EnableTensorRtEngine(1 << 30,
                                  batch_size,
                                  this->min_subgraph_size_,
                                  precision,
                                  false,
                                  this->trt_calib_mode_);

      // set use dynamic shape
      if (this->use_dynamic_shape_) {
        // set DynamicShsape for image tensor
        const std::vector<int> min_input_shape = {
            1, 3, this->trt_min_shape_, this->trt_min_shape_};
        const std::vector<int> max_input_shape = {
            1, 3, this->trt_max_shape_, this->trt_max_shape_};
        const std::vector<int> opt_input_shape = {
            1, 3, this->trt_opt_shape_, this->trt_opt_shape_};
        const std::map<std::string, std::vector<int>> map_min_input_shape = {
            {"image", min_input_shape}};
        const std::map<std::string, std::vector<int>> map_max_input_shape = {
            {"image", max_input_shape}};
        const std::map<std::string, std::vector<int>> map_opt_input_shape = {
            {"image", opt_input_shape}};

        config.SetTRTDynamicShapeInfo(
            map_min_input_shape, map_max_input_shape, map_opt_input_shape);
        std::cout << "TensorRT dynamic shape enabled" << std::endl;
      }
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

// Visualiztion MaskDetector results
cv::Mat VisualizeKptsResult(const cv::Mat& img,
                            const std::vector<KeyPointResult>& results,
                            const std::vector<int>& colormap) {
  const int edge[][2] = {{0, 1},
                         {0, 2},
                         {1, 3},
                         {2, 4},
                         {3, 5},
                         {4, 6},
                         {5, 7},
                         {6, 8},
                         {7, 9},
                         {8, 10},
                         {5, 11},
                         {6, 12},
                         {11, 13},
                         {12, 14},
                         {13, 15},
                         {14, 16},
                         {11, 12}};
  cv::Mat vis_img = img.clone();
  for (int batchid = 0; batchid < results.size(); batchid++) {
    for (int i = 0; i < results[batchid].num_joints; i++) {
      if (results[batchid].keypoints[i * 3] > 0.5) {
        int x_coord = int(results[batchid].keypoints[i * 3 + 1]);
        int y_coord = int(results[batchid].keypoints[i * 3 + 2]);
        cv::circle(vis_img,
                   cv::Point2d(x_coord, y_coord),
                   1,
                   cv::Scalar(0, 0, 255),
                   2);
      }
    }
    for (int i = 0; i < results[batchid].num_joints; i++) {
      int x_start = int(results[batchid].keypoints[edge[i][0] * 3 + 1]);
      int y_start = int(results[batchid].keypoints[edge[i][0] * 3 + 2]);
      int x_end = int(results[batchid].keypoints[edge[i][1] * 3 + 1]);
      int y_end = int(results[batchid].keypoints[edge[i][1] * 3 + 2]);
      cv::line(vis_img,
               cv::Point2d(x_start, y_start),
               cv::Point2d(x_end, y_end),
               colormap[i],
               1);
    }
  }
  return vis_img;
}

void KeyPointDetector::Preprocess(const cv::Mat& ori_im) {
  // Clone the image : keep the original mat for postprocess
  cv::Mat im = ori_im.clone();
  cv::cvtColor(im, im, cv::COLOR_BGR2RGB);
  preprocessor_.Run(&im, &inputs_);
}

void KeyPointDetector::Postprocess(std::vector<float>& output,
                                   std::vector<int> output_shape,
                                   std::vector<int64_t>& idxout,
                                   std::vector<int> idx_shape,
                                   std::vector<KeyPointResult>* result,
                                   std::vector<std::vector<float>>& center_bs,
                                   std::vector<std::vector<float>>& scale_bs) {
  std::vector<float> preds(output_shape[1] * 3, 0);

  for (int batchid = 0; batchid < output_shape[0]; batchid++) {
    get_final_preds(output,
                    output_shape,
                    idxout,
                    idx_shape,
                    center_bs[batchid],
                    scale_bs[batchid],
                    preds,
                    batchid,
                    this->use_dark);
    KeyPointResult result_item;
    result_item.num_joints = output_shape[1];
    result_item.keypoints.clear();
    for (int i = 0; i < output_shape[1]; i++) {
      result_item.keypoints.emplace_back(preds[i * 3]);
      result_item.keypoints.emplace_back(preds[i * 3 + 1]);
      result_item.keypoints.emplace_back(preds[i * 3 + 2]);
    }
    result->push_back(result_item);
  }
}

void KeyPointDetector::Predict(const std::vector<cv::Mat> imgs,
                               std::vector<std::vector<float>>& center_bs,
                               std::vector<std::vector<float>>& scale_bs,
                               const double threshold,
                               const int warmup,
                               const int repeats,
                               std::vector<KeyPointResult>* result,
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

    // TODO: reduce cost time
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
  std::vector<int> output_shape, idx_shape;
  // Run predictor
  // warmup
  for (int i = 0; i < warmup; i++) {
    predictor_->Run();
    // Get output tensor
    auto output_names = predictor_->GetOutputNames();
    auto out_tensor = predictor_->GetOutputHandle(output_names[0]);
    output_shape = out_tensor->shape();
    // Calculate output length
    int output_size = 1;
    for (int j = 0; j < output_shape.size(); ++j) {
      output_size *= output_shape[j];
    }
    output_data_.resize(output_size);
    out_tensor->CopyToCpu(output_data_.data());

    auto idx_tensor = predictor_->GetOutputHandle(output_names[1]);
    idx_shape = idx_tensor->shape();
    // Calculate output length
    output_size = 1;
    for (int j = 0; j < idx_shape.size(); ++j) {
      output_size *= idx_shape[j];
    }
    idx_data_.resize(output_size);
    idx_tensor->CopyToCpu(idx_data_.data());
  }

  auto inference_start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeats; i++) {
    predictor_->Run();
    // Get output tensor
    auto output_names = predictor_->GetOutputNames();
    auto out_tensor = predictor_->GetOutputHandle(output_names[0]);
    output_shape = out_tensor->shape();
    // Calculate output length
    int output_size = 1;
    for (int j = 0; j < output_shape.size(); ++j) {
      output_size *= output_shape[j];
    }
    if (output_size < 6) {
      std::cerr << "[WARNING] No object detected." << std::endl;
    }
    output_data_.resize(output_size);
    out_tensor->CopyToCpu(output_data_.data());

    auto idx_tensor = predictor_->GetOutputHandle(output_names[1]);
    idx_shape = idx_tensor->shape();
    // Calculate output length
    output_size = 1;
    for (int j = 0; j < idx_shape.size(); ++j) {
      output_size *= idx_shape[j];
    }
    idx_data_.resize(output_size);
    idx_tensor->CopyToCpu(idx_data_.data());
  }
  auto inference_end = std::chrono::steady_clock::now();
  auto postprocess_start = std::chrono::steady_clock::now();
  // Postprocessing result
  Postprocess(output_data_,
              output_shape,
              idx_data_,
              idx_shape,
              result,
              center_bs,
              scale_bs);
  auto postprocess_end = std::chrono::steady_clock::now();

  std::chrono::duration<float> preprocess_diff =
      preprocess_end - preprocess_start;
  times->push_back(double(preprocess_diff.count() * 1000));
  std::chrono::duration<float> inference_diff = inference_end - inference_start;
  times->push_back(double(inference_diff.count() / repeats * 1000));
  std::chrono::duration<float> postprocess_diff =
      postprocess_end - postprocess_start;
  times->push_back(double(postprocess_diff.count() * 1000));
}

}  // namespace PaddleDetection
