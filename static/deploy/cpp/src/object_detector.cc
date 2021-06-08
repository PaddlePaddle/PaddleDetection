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
#include <sstream>
// for setprecision
#include <iomanip>
#include <chrono>
#include "include/object_detector.h"

namespace PaddleDetection {

// Load Model and create model predictor
void ObjectDetector::LoadModel(const std::string& model_dir,
                               const std::string& device,
                               const int min_subgraph_size,
                               const int batch_size,
                               const std::string& run_mode,
                               const int gpu_id,
                               bool trt_calib_mode) {
  paddle::AnalysisConfig config;
  std::string prog_file = model_dir + OS_PATH_SEP + "__model__";
  std::string params_file = model_dir + OS_PATH_SEP + "__params__";
  config.SetModel(prog_file, params_file);
  if (device == "GPU") {
    config.EnableUseGpu(100, gpu_id);
    config.SwitchIrOptim(true);
    if (run_mode != "fluid") {
      auto precision = paddle::AnalysisConfig::Precision::kFloat32;
      if (run_mode == "trt_fp16") {
        precision = paddle::AnalysisConfig::Precision::kHalf;
      } else if (run_mode == "trt_int8") {
        precision = paddle::AnalysisConfig::Precision::kInt8;
      } else {
        printf("run_mode should be 'fluid', 'trt_fp32', 'trt_fp16' or 'trt_int8'");
      }
      config.EnableTensorRtEngine(
          1 << 10,
          batch_size,
          min_subgraph_size,
          precision,
          false,
          trt_calib_mode);
   }
  } else if (device == "XPU"){
    config.EnableXpu(10*1024*1024);
  } else {
    config.DisableGpu();
  }
  config.SwitchUseFeedFetchOps(false);
  config.SwitchSpecifyInputNames(true);
  config.DisableGlogInfo();
  // Memory optimization
  config.EnableMemoryOptim();
  predictor_ = std::move(CreatePaddlePredictor(config));
}

// Visualiztion MaskDetector results
cv::Mat VisualizeResult(const cv::Mat& img,
                        const std::vector<ObjectResult>& results,
                        const std::vector<std::string>& lable_list,
                        const std::vector<int>& colormap) {
  cv::Mat vis_img = img.clone();
  for (int i = 0; i < results.size(); ++i) {
    int w = results[i].rect[1] - results[i].rect[0];
    int h = results[i].rect[3] - results[i].rect[2];
    cv::Rect roi = cv::Rect(results[i].rect[0], results[i].rect[2], w, h);

    // Configure color and text size
    std::ostringstream oss;
    oss << std::setiosflags(std::ios::fixed) << std::setprecision(4);
    oss << lable_list[results[i].class_id] << " ";
    oss << results[i].confidence;
    std::string text = oss.str();
    int c1 = colormap[3 * results[i].class_id + 0];
    int c2 = colormap[3 * results[i].class_id + 1];
    int c3 = colormap[3 * results[i].class_id + 2];
    cv::Scalar roi_color = cv::Scalar(c1, c2, c3);
    int font_face = cv::FONT_HERSHEY_COMPLEX_SMALL;
    double font_scale = 0.5f;
    float thickness = 0.5;
    cv::Size text_size = cv::getTextSize(text,
                                         font_face,
                                         font_scale,
                                         thickness,
                                         nullptr);
    cv::Point origin;
    origin.x = roi.x;
    origin.y = roi.y;

    // Configure text background
    cv::Rect text_back = cv::Rect(results[i].rect[0],
                                  results[i].rect[2] - text_size.height,
                                  text_size.width,
                                  text_size.height);

    // Draw roi object, text, and background
    cv::rectangle(vis_img, roi, roi_color, 2);
    cv::rectangle(vis_img, text_back, roi_color, -1);
    cv::putText(vis_img,
                text,
                origin,
                font_face,
                font_scale,
                cv::Scalar(255, 255, 255),
                thickness);
  }
  return vis_img;
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
  if (config_.arch_ == "SSD" || config_.arch_ == "Face") {
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
    if (score > threshold_ && class_id > -1) {
      ObjectResult result_item;
      result_item.rect = {xmin, xmax, ymin, ymax};
      result_item.class_id = class_id;
      result_item.confidence = score;
      result->push_back(result_item);
    }
  }
}

void ObjectDetector::Predict(const cv::Mat& im,
      const double threshold,
      const int warmup,
      const int repeats,
      const bool run_benchmark,
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
    } else if (tensor_name == "scale_factor") {
      in_tensor->Reshape({1, 4});
      in_tensor->copy_from_cpu(inputs_.scale_factor_f_.data());
    }
  }
  // Run predictor
  for (int i = 0; i < warmup; i++)
  {
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

    if (output_size < 6) {
      std::cerr << "[WARNING] No object detected." << std::endl;
    }
    output_data_.resize(output_size);
    out_tensor->copy_to_cpu(output_data_.data()); 
  }

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeats; i++)
  {
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

    if (output_size < 6) {
      std::cerr << "[WARNING] No object detected." << std::endl;
    }
    output_data_.resize(output_size);
    out_tensor->copy_to_cpu(output_data_.data()); 
  }
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<float> diff = end - start;
  float ms = diff.count() / repeats * 1000;
  printf("Inference: %f ms per batch image\n", ms);
  // Postprocessing result
  if(!run_benchmark) {
    Postprocess(im,  result);
  }
}

std::vector<int> GenerateColorMap(int num_class) {
  auto colormap = std::vector<int>(3 * num_class, 0);
  for (int i = 0; i < num_class; ++i) {
    int j = 0;
    int lab = i;
    while (lab) {
      colormap[i * 3] |= (((lab >> 0) & 1) << (7 - j));
      colormap[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j));
      colormap[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j));
      ++j;
      lab >>= 3;
    }
  }
  return colormap;
}

}  // namespace PaddleDetection
