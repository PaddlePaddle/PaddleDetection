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
#include "include/movenet_detector.h"

namespace PaddleDetection {

// Load Model and create model predictor
void ObjectDetector::LoadModel(std::string model_file, int num_theads) {
  MobileConfig config;
  config.set_threads(num_theads);
  config.set_model_from_file(model_file + "/movenet.nb");
  config.set_power_mode(LITE_POWER_HIGH);

  predictor_ = CreatePaddlePredictor<MobileConfig>(config);
}

// Visualiztion MaskDetector results
cv::Mat VisualizeResult(const cv::Mat& img,
                        const std::vector<ObjectResult>& results,
                        const std::vector<int>& colormap,
                        float threshold) {
  cv::Mat vis_img = img.clone();
  printf("\nINFO: Detect person number: %d\n", results.size());
  if (results.size() > 1) {
    for (int i = 0; i < results.size(); ++i) {
      printf("INFO: Number {%d} rect :[ %d %d %d %d ]\n",
             i + 1,
             static_cast<int>(results[i].rect[0]),
             static_cast<int>(results[i].rect[1]),
             static_cast<int>(results[i].rect[2]),
             static_cast<int>(results[i].rect[3]));
      // Configure color and text size
      std::ostringstream oss;
      oss << std::setiosflags(std::ios::fixed) << std::setprecision(4);
      oss << results[i].confidence;
      std::string text = oss.str();
      int c1 = colormap[i * 3];
      int c2 = colormap[i * 3 + 1];
      int c3 = colormap[i * 3 + 2];
      cv::Scalar roi_color = cv::Scalar(c1, c2, c3);
      int font_face = cv::FONT_HERSHEY_COMPLEX_SMALL;
      double font_scale = 0.5f;
      float thickness = 0.5;
      cv::Size text_size =
          cv::getTextSize(text, font_face, font_scale, thickness, nullptr);
      cv::Point origin;

      int w = results[i].rect[2] - results[i].rect[0];
      int h = results[i].rect[3] - results[i].rect[1];
      cv::Rect roi = cv::Rect(results[i].rect[0], results[i].rect[1], w, h);
      // Draw roi object, text, and background
      cv::rectangle(vis_img, roi, roi_color, 2);

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
  }

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
  for (int batchid = 0; batchid < results.size(); batchid++) {
    for (int i = 0; i < 17; i++) {
      int c1 = colormap[i * 3];
      int c2 = colormap[i * 3 + 1];
      int c3 = colormap[i * 3 + 2];
      cv::Scalar roi_color = cv::Scalar(c1, c2, c3);

      if (results[batchid].kpts[i * 3] > threshold) {
        int x_coord = int(results[batchid].kpts[i * 3 + 1]);
        int y_coord = int(results[batchid].kpts[i * 3 + 2]);
        cv::circle(vis_img, cv::Point2d(x_coord, y_coord), 1, roi_color, 2);
      }
    }
    for (int i = 0; i < 17; i++) {
      int c1 = colormap[i * 3];
      int c2 = colormap[i * 3 + 1];
      int c3 = colormap[i * 3 + 2];
      cv::Scalar roi_color = cv::Scalar(c1, c2, c3);

      if (results[batchid].kpts[edge[i][0] * 3] > threshold &&
          results[batchid].kpts[edge[i][1] * 3] > threshold) {
        int x_start = int(results[batchid].kpts[edge[i][0] * 3 + 1]);
        int y_start = int(results[batchid].kpts[edge[i][0] * 3 + 2]);
        int x_end = int(results[batchid].kpts[edge[i][1] * 3 + 1]);
        int y_end = int(results[batchid].kpts[edge[i][1] * 3 + 2]);
        cv::line(vis_img,
                 cv::Point2d(x_start, y_start),
                 cv::Point2d(x_end, y_end),
                 roi_color,
                 3);
      }
    }
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
                                 std::vector<ObjectResult>* result,
                                 int personnum) {
  int h = mats[0].rows;
  int w = mats[0].cols;
  if (h > w) {
    w = h;
  }
  if (w > h) {
    h = w;
  }

  for (int i = 0; i < personnum; i++) {
    float conf = 1.;
    if (personnum > 1) {
      conf = output_data_[55 + i * 56];
      if (conf < threshold_) {
        continue;
      }
    }
    ObjectResult itemres;
    itemres.rect.resize(4);
    itemres.kpts.resize(17 * 3);
    itemres.confidence = conf;
    if (personnum > 1) {
      itemres.rect[0] = output_data_[52 + i * 56] * w;
      itemres.rect[1] = output_data_[51 + i * 56] * h;
      itemres.rect[2] = output_data_[54 + i * 56] * w;
      itemres.rect[3] = output_data_[53 + i * 56] * h;
    }
    for (int j = 0; j < 17; j++) {
      itemres.kpts[j * 3] = output_data_[j * 3 + 2 + i * 56];
      itemres.kpts[j * 3 + 1] = output_data_[j * 3 + 1 + i * 56] * w;
      itemres.kpts[j * 3 + 2] = output_data_[j * 3 + i * 56] * h;
    }
    result->emplace_back(itemres);
  }
}

void ObjectDetector::Predict(const std::vector<cv::Mat>& imgs,
                             const double threshold,
                             const int warmup,
                             const int repeats,
                             std::vector<ObjectResult>* result,
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
  // Prepare input tensor
  auto input_names = predictor_->GetInputNames();
  for (const auto& tensor_name : input_names) {
    auto in_tensor = predictor_->GetInputByName(tensor_name);
    int rh = inputs_.in_net_shape_[0];
    int rw = inputs_.in_net_shape_[1];
    in_tensor->Resize({batch_size, 3, rh, rw});
    auto* inptr = in_tensor->mutable_data<float>();
    std::copy_n(in_data_all.data(), in_data_all.size(), inptr);
  }

  // Run predictor
  // warmup
  for (int i = 0; i < warmup; i++) {
    predictor_->Run();
    // Get output tensor
    auto output_names = predictor_->GetOutputNames();
    auto out_tensor = predictor_->GetTensor(output_names[0]);
  }

  auto inference_start = std::chrono::steady_clock::now();
  int personnum = 1;
  for (int i = 0; i < repeats; i++) {
    predictor_->Run();
    // Get output tensor
    auto output_names = predictor_->GetOutputNames();
    auto output_tensor = predictor_->GetTensor(output_names[0]);
    auto output_shape = output_tensor->shape();
    // Calculate output length
    int output_size = 1;
    for (int j = 0; j < output_shape.size(); ++j) {
      output_size *= output_shape[j];
    }
    personnum = output_shape[1];

    if (output_size < 6) {
      std::cerr << "[WARNING] No object detected." << std::endl;
    }
    output_data_.resize(output_size);
    std::copy_n(
        output_tensor->mutable_data<float>(), output_size, output_data_.data());
  }
  auto inference_end = std::chrono::steady_clock::now();
  auto postprocess_start = std::chrono::steady_clock::now();
  // Postprocessing result
  result->clear();
  Postprocess(imgs, result, personnum);
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
