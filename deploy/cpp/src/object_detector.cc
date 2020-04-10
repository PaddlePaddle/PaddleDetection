//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# include "include/object_detector.h"

namespace PaddleDetection {

// Load Model and create model predictor
void ObjectDetector::LoadModel(const std::string& model_dir, bool use_gpu) {
  paddle::AnalysisConfig config;
  std::string prog_file = model_dir + OS_PATH_SEP + "__model__";
  std::string params_file = model_dir + OS_PATH_SEP + "__params__";
  config.SetModel(prog_file, params_file);
  if (use_gpu) {
      config.EnableUseGpu(100, 0);
  } else {
      config.DisableGpu();
  }
  config.SwitchUseFeedFetchOps(false);
  config.SwitchSpecifyInputNames(true);
  // Memory optimization
  config.EnableMemoryOptim();
  predictor_ = std::move(CreatePaddlePredictor(config));
}

// Visualiztion MaskDetector results
void VisualizeResult(const cv::Mat& img,
                     const std::vector<ObjectResult>& results,
                     cv::Mat* vis_img) {
}

void ObjectDetector::Preprocess(const cv::Mat& ori_im) {
  // Clone the image : keep the original mat for postprocess
  cv::Mat im = ori_im.clone();
  cv::cvtColor(im, im, cv::COLOR_BGR2RGB);
  preprocessor_.Run(&im, &inputs_);
}

void ObjectDetector::Postprocess(
    const cv::Mat& raw_mat,
    std::vector<ObjectResult>* result) {
  result->clear();
  int rh = 1;
  int rw = 1;
  if (config_.arch_ == "SSD") {
    rh = raw_mat.rows;
    rw = raw_mat.cols;
  }

  int total_size = output_data_.size() / 6;
  for (int j = 0; j < total_size; ++j) {
    // Class id
    int class_id = static_cast<int>(round(output_data_[0 + j * 6]));
    // Confidence score
    float score = output_data_[1 + j * 6];
    int xmin = (output_data_[2 + j * 6] * rw);
    int ymin = (output_data_[3 + j * 6] * rh);
    int xmax = (output_data_[4 + j * 6] * rw);
    int ymax = (output_data_[5 + j * 6] * rh);
    int wd = xmax - xmin;
    int hd = ymax - ymin;
    if (score > threshold_) {
      ObjectResult result_item;
      result_item.rect = {xmin, xmax, ymin, ymax};
      result_item.class_id = class_id;
      result_item.confidence = score;
      result->push_back(result_item);
      printf("class_id=%d, confidence=%.4f, rect=[%d, %d, %d, %d]\n",
          class_id, score, xmin, xmax, ymin, ymax);
    }
  }
}

void ObjectDetector::Predict(const cv::Mat& im,
                                  std::vector<ObjectResult>* result) {
  // Preprocess image
  Preprocess(im);
  // Prepare input tensor
  auto input_names = predictor_->GetInputNames();
  for (const auto& tensor_name : input_names) {
    auto in_tensor = predictor_->GetInputTensor(tensor_name);
    if (tensor_name == "image") {
      int rh = inputs_.eval_im_size_f_[0];
      int rw = inputs_.eval_im_size_f_[1];
      in_tensor->Reshape({1, 3, rh, rw});
      in_tensor->copy_from_cpu(inputs_.im_data_.data());
    } else if (tensor_name == "im_size") {
      in_tensor->Reshape({1, 2});
      in_tensor->copy_from_cpu(inputs_.ori_im_size_.data());
    } else if (tensor_name == "im_info") {
      in_tensor->Reshape({1, 3});
      in_tensor->copy_from_cpu(inputs_.eval_im_size_f_.data());
    } else if (tensor_name == "im_shape") {
      in_tensor->Reshape({1, 3});
      in_tensor->copy_from_cpu(inputs_.ori_im_size_f_.data());
    }
  }
  // Run predictor
  predictor_->ZeroCopyRun();
  // Get output tensor
  auto output_names = predictor_->GetOutputNames();
  auto out_tensor = predictor_->GetOutputTensor(output_names[0]);
  std::vector<int> output_shape = out_tensor->shape();
  // Calculate output length
  int output_size = 1;
  for (int j = 0; j < output_shape.size(); ++j) {
      output_size *= output_shape[j];
  }
  output_data_.resize(output_size);
  out_tensor->copy_to_cpu(output_data_.data());
  // Postprocessing result
  Postprocess(im,  result);
}

}  // namespace PaddleDetection
