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
#include "include/object_detector.h"

namespace PaddleDetection {

// Load Model and create model predictor
void ObjectDetector::LoadModel(std::string model_file, int num_theads) {
  MobileConfig config;
  config.set_threads(num_theads);
  config.set_model_from_file(model_file + "/model.nb");
  config.set_power_mode(LITE_POWER_HIGH);

  predictor_ = CreatePaddlePredictor<MobileConfig>(config);
}

// Visualiztion MaskDetector results
cv::Mat VisualizeResult(const cv::Mat& img,
                        const std::vector<PaddleDetection::ObjectResult>& results,
                        const std::vector<std::string>& lables,
                        const std::vector<int>& colormap,
                        const bool is_rbox = false) {
  cv::Mat vis_img = img.clone();
  for (int i = 0; i < results.size(); ++i) {
    // Configure color and text size
    std::ostringstream oss;
    oss << std::setiosflags(std::ios::fixed) << std::setprecision(4);
    oss << lables[results[i].class_id] << " ";
    oss << results[i].confidence;
    std::string text = oss.str();
    int c1 = colormap[3 * results[i].class_id + 0];
    int c2 = colormap[3 * results[i].class_id + 1];
    int c3 = colormap[3 * results[i].class_id + 2];
    cv::Scalar roi_color = cv::Scalar(c1, c2, c3);
    int font_face = cv::FONT_HERSHEY_COMPLEX_SMALL;
    double font_scale = 0.5f;
    float thickness = 0.5;
    cv::Size text_size =
        cv::getTextSize(text, font_face, font_scale, thickness, nullptr);
    cv::Point origin;

    if (is_rbox) {
      // Draw object, text, and background
      for (int k = 0; k < 4; k++) {
        cv::Point pt1 = cv::Point(results[i].rect[(k * 2) % 8],
                                  results[i].rect[(k * 2 + 1) % 8]);
        cv::Point pt2 = cv::Point(results[i].rect[(k * 2 + 2) % 8],
                                  results[i].rect[(k * 2 + 3) % 8]);
        cv::line(vis_img, pt1, pt2, roi_color, 2);
      }
    } else {
      int w = results[i].rect[2] - results[i].rect[0];
      int h = results[i].rect[3] - results[i].rect[1];
      cv::Rect roi = cv::Rect(results[i].rect[0], results[i].rect[1], w, h);
      // Draw roi object, text, and background
      cv::rectangle(vis_img, roi, roi_color, 2);
    }

    origin.x = results[i].rect[0];
    origin.y = results[i].rect[1];

    // Configure text background
    cv::Rect text_back = cv::Rect(results[i].rect[0],
                                  results[i].rect[1] - text_size.height,
                                  text_size.width,
                                  text_size.height);
    // Draw text, and background
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

void ObjectDetector::Postprocess(const std::vector<cv::Mat> mats,
                                 std::vector<PaddleDetection::ObjectResult>* result,
                                 std::vector<int> bbox_num,
                                 bool is_rbox = false) {
  result->clear();
  int start_idx = 0;
  for (int im_id = 0; im_id < mats.size(); im_id++) {
    cv::Mat raw_mat = mats[im_id];
    int rh = 1;
    int rw = 1;
    if (config_.arch_ == "Face") {
      rh = raw_mat.rows;
      rw = raw_mat.cols;
    }
    for (int j = start_idx; j < start_idx + bbox_num[im_id]; j++) {
      if (is_rbox) {
        // Class id
        int class_id = static_cast<int>(round(output_data_[0 + j * 10]));
        // Confidence score
        float score = output_data_[1 + j * 10];
        int x1 = (output_data_[2 + j * 10] * rw);
        int y1 = (output_data_[3 + j * 10] * rh);
        int x2 = (output_data_[4 + j * 10] * rw);
        int y2 = (output_data_[5 + j * 10] * rh);
        int x3 = (output_data_[6 + j * 10] * rw);
        int y3 = (output_data_[7 + j * 10] * rh);
        int x4 = (output_data_[8 + j * 10] * rw);
        int y4 = (output_data_[9 + j * 10] * rh);

        PaddleDetection::ObjectResult result_item;
        result_item.rect = {x1, y1, x2, y2, x3, y3, x4, y4};
        result_item.class_id = class_id;
        result_item.confidence = score;
        result->push_back(result_item);
      } else {
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

        PaddleDetection::ObjectResult result_item;
        result_item.rect = {xmin, ymin, xmax, ymax};
        result_item.class_id = class_id;
        result_item.confidence = score;
        result->push_back(result_item);
      }
    }
    start_idx += bbox_num[im_id];
  }
}

void ObjectDetector::Predict(const std::vector<cv::Mat>& imgs,
                             const double threshold,
                             const int warmup,
                             const int repeats,
                             std::vector<PaddleDetection::ObjectResult>* result,
                             std::vector<int>* bbox_num,
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
  auto preprocess_end = std::chrono::steady_clock::now();
  std::vector<const float *> output_data_list_;
  // Prepare input tensor

  auto input_names = predictor_->GetInputNames();
  for (const auto& tensor_name : input_names) {
    auto in_tensor = predictor_->GetInputByName(tensor_name);
    if (tensor_name == "image") {
      int rh = inputs_.in_net_shape_[0];
      int rw = inputs_.in_net_shape_[1];
      in_tensor->Resize({batch_size, 3, rh, rw});
      auto* inptr = in_tensor->mutable_data<float>();
      std::copy_n(in_data_all.data(), in_data_all.size(), inptr);
    } else if (tensor_name == "im_shape") {
      in_tensor->Resize({batch_size, 2});
      auto* inptr = in_tensor->mutable_data<float>();
      std::copy_n(im_shape_all.data(), im_shape_all.size(), inptr);
    } else if (tensor_name == "scale_factor") {
      in_tensor->Resize({batch_size, 2});
      auto* inptr = in_tensor->mutable_data<float>();
      std::copy_n(scale_factor_all.data(), scale_factor_all.size(), inptr);
    }
  }

  // Run predictor
  // warmup
  for (int i = 0; i < warmup; i++) {
    predictor_->Run();
    // Get output tensor
    auto output_names = predictor_->GetOutputNames();
    if (config_.arch_ == "PicoDet") {
      for (int j = 0; j < output_names.size(); j++) {
        auto output_tensor = predictor_->GetTensor(output_names[j]);
        const float* outptr = output_tensor->data<float>();
        std::vector<int64_t> output_shape = output_tensor->shape();
        output_data_list_.push_back(outptr);
      }
    } else {
      auto out_tensor = predictor_->GetTensor(output_names[0]);
      auto out_bbox_num = predictor_->GetTensor(output_names[1]);
    }
  }

  bool is_rbox = false;
  auto inference_start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeats; i++) {
    predictor_->Run();
  }
  auto inference_end = std::chrono::steady_clock::now();
  auto postprocess_start = std::chrono::steady_clock::now();
  // Get output tensor
  output_data_list_.clear();
  int num_class = 80;
  int reg_max = 7;
  auto output_names = predictor_->GetOutputNames();
  // TODO: Unified model output.
  if (config_.arch_ == "PicoDet") {
    for (int i = 0; i < output_names.size(); i++) {
      auto output_tensor = predictor_->GetTensor(output_names[i]);
      const float* outptr = output_tensor->data<float>();
      std::vector<int64_t> output_shape = output_tensor->shape();
      if (i == 0) {
        num_class = output_shape[2];
      }
      if (i == config_.fpn_stride_.size()) {
        reg_max = output_shape[2] / 4 - 1;
      }
      output_data_list_.push_back(outptr);
    }
  } else {
    auto output_tensor = predictor_->GetTensor(output_names[0]);
    auto output_shape = output_tensor->shape();
    auto out_bbox_num = predictor_->GetTensor(output_names[1]);
    auto out_bbox_num_shape = out_bbox_num->shape();
    // Calculate output length
    int output_size = 1;
    for (int j = 0; j < output_shape.size(); ++j) {
      output_size *= output_shape[j];
    }
    is_rbox = output_shape[output_shape.size() - 1] % 10 == 0;

    if (output_size < 6) {
      std::cerr << "[WARNING] No object detected." << std::endl;
    }
    output_data_.resize(output_size);
    std::copy_n(
        output_tensor->mutable_data<float>(), output_size, output_data_.data());

    int out_bbox_num_size = 1;
    for (int j = 0; j < out_bbox_num_shape.size(); ++j) {
      out_bbox_num_size *= out_bbox_num_shape[j];
    }
    out_bbox_num_data_.resize(out_bbox_num_size);
    std::copy_n(out_bbox_num->mutable_data<int>(),
                out_bbox_num_size,
                out_bbox_num_data_.data());
  }
  // Postprocessing result
  result->clear();
  if (config_.arch_ == "PicoDet") {
    PaddleDetection::PicoDetPostProcess(
        result, output_data_list_, config_.fpn_stride_, 
        inputs_.im_shape_, inputs_.scale_factor_,
        config_.nms_info_["score_threshold"].as<float>(), 
        config_.nms_info_["nms_threshold"].as<float>(), num_class, reg_max);
    bbox_num->push_back(result->size());
  } else {
    Postprocess(imgs, result, out_bbox_num_data_, is_rbox);
    bbox_num->clear();
    for (int k = 0; k < out_bbox_num_data_.size(); k++) {
      int tmp = out_bbox_num_data_[k];
      bbox_num->push_back(tmp);
    }
  }
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
