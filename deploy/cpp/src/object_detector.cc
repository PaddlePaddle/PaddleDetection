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
#include <chrono>
#include <iomanip>

#include "include/object_detector.h"

namespace PaddleDetection {

// Load Model and create model predictor
void ObjectDetector::LoadModel(const std::string &model_dir,
                               const int batch_size,
                               const std::string &run_mode) {
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
        printf("run_mode should be 'paddle', 'trt_fp32', 'trt_fp16' or "
               "'trt_int8'");
      }
      // set tensorrt
      config.EnableTensorRtEngine(1 << 30, batch_size, this->min_subgraph_size_,
                                  precision, false, this->trt_calib_mode_);

      // set use dynamic shape
      if (this->use_dynamic_shape_) {
        // set DynamicShape for image tensor
        const std::vector<int> min_input_shape = {
            batch_size, 3, this->trt_min_shape_, this->trt_min_shape_};
        const std::vector<int> max_input_shape = {
            batch_size, 3, this->trt_max_shape_, this->trt_max_shape_};
        const std::vector<int> opt_input_shape = {
            batch_size, 3, this->trt_opt_shape_, this->trt_opt_shape_};
        const std::map<std::string, std::vector<int>> map_min_input_shape = {
            {"image", min_input_shape}};
        const std::map<std::string, std::vector<int>> map_max_input_shape = {
            {"image", max_input_shape}};
        const std::map<std::string, std::vector<int>> map_opt_input_shape = {
            {"image", opt_input_shape}};

        config.SetTRTDynamicShapeInfo(map_min_input_shape, map_max_input_shape,
                                      map_opt_input_shape);
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
cv::Mat
VisualizeResult(const cv::Mat &img,
                const std::vector<PaddleDetection::ObjectResult> &results,
                const std::vector<std::string> &lables,
                const std::vector<int> &colormap, const bool is_rbox = false) {
  cv::Mat vis_img = img.clone();
  int img_h = vis_img.rows;
  int img_w = vis_img.cols;
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

      // Draw mask
      std::vector<int> mask_v = results[i].mask;
      if (mask_v.size() > 0) {
        cv::Mat mask = cv::Mat(img_h, img_w, CV_32S);
        std::memcpy(mask.data, mask_v.data(), mask_v.size() * sizeof(int));

        cv::Mat colored_img = vis_img.clone();

        std::vector<cv::Mat> contours;
        cv::Mat hierarchy;
        mask.convertTo(mask, CV_8U);
        cv::findContours(mask, contours, hierarchy, cv::RETR_CCOMP,
                         cv::CHAIN_APPROX_SIMPLE);
        cv::drawContours(colored_img, contours, -1, roi_color, -1, cv::LINE_8,
                         hierarchy, 100);

        cv::Mat debug_roi = vis_img;
        colored_img = 0.4 * colored_img + 0.6 * vis_img;
        colored_img.copyTo(vis_img, mask);
      }
    }

    origin.x = results[i].rect[0];
    origin.y = results[i].rect[1];

    // Configure text background
    cv::Rect text_back =
        cv::Rect(results[i].rect[0], results[i].rect[1] - text_size.height,
                 text_size.width, text_size.height);
    // Draw text, and background
    cv::rectangle(vis_img, text_back, roi_color, -1);
    cv::putText(vis_img, text, origin, font_face, font_scale,
                cv::Scalar(255, 255, 255), thickness);
  }
  return vis_img;
}

void ObjectDetector::Preprocess(const cv::Mat &ori_im) {
  // Clone the image : keep the original mat for postprocess
  cv::Mat im = ori_im.clone();
  cv::cvtColor(im, im, cv::COLOR_BGR2RGB);
  preprocessor_.Run(&im, &inputs_);
}

void ObjectDetector::Postprocess(
    const std::vector<cv::Mat> mats,
    std::vector<PaddleDetection::ObjectResult> *result,
    std::vector<int> bbox_num, std::vector<float> output_data_,
    std::vector<int> output_mask_data_, bool is_rbox = false) {
  result->clear();
  int start_idx = 0;
  int total_num = std::accumulate(bbox_num.begin(), bbox_num.end(), 0);
  int out_mask_dim = -1;
  if (config_.mask_) {
    out_mask_dim = output_mask_data_.size() / total_num;
  }

  for (int im_id = 0; im_id < mats.size(); im_id++) {
    cv::Mat raw_mat = mats[im_id];
    int rh = 1;
    int rw = 1;
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

        if (config_.mask_) {
          std::vector<int> mask;
          for (int k = 0; k < out_mask_dim; ++k) {
            if (output_mask_data_[k + j * out_mask_dim] > -1) {
              mask.push_back(output_mask_data_[k + j * out_mask_dim]);
            }
          }
          result_item.mask = mask;
        }

        result->push_back(result_item);
      }
    }
    start_idx += bbox_num[im_id];
  }
}

// This function is to convert output result from SOLOv2 to class ObjectResult
void ObjectDetector::SOLOv2Postprocess(
    const std::vector<cv::Mat> mats, std::vector<ObjectResult> *result,
    std::vector<int> *bbox_num, std::vector<int> out_bbox_num_data_,
    std::vector<int64_t> out_label_data_, std::vector<float> out_score_data_,
    std::vector<uint8_t> out_global_mask_data_, float threshold) {

  for (int im_id = 0; im_id < mats.size(); im_id++) {
    cv::Mat mat = mats[im_id];

    int valid_bbox_count = 0;
    for (int bbox_id = 0; bbox_id < out_bbox_num_data_[im_id]; ++bbox_id) {
      if (out_score_data_[bbox_id] >= threshold) {
        ObjectResult result_item;
        result_item.class_id = out_label_data_[bbox_id];
        result_item.confidence = out_score_data_[bbox_id];
        std::vector<int> global_mask;

        for (int k = 0; k < mat.rows * mat.cols; ++k) {
          global_mask.push_back(static_cast<int>(
              out_global_mask_data_[k + bbox_id * mat.rows * mat.cols]));
        }

        // find minimize bounding box from mask
        cv::Mat mask(mat.rows, mat.cols, CV_32SC1);
        std::memcpy(mask.data, global_mask.data(),
                    global_mask.size() * sizeof(int));

        cv::Mat mask_fp;
        cv::Mat rowSum;
        cv::Mat colSum;
        std::vector<float> sum_of_row(mat.rows);
        std::vector<float> sum_of_col(mat.cols);

        mask.convertTo(mask_fp, CV_32FC1);
        cv::reduce(mask_fp, colSum, 0, CV_REDUCE_SUM, CV_32FC1);
        cv::reduce(mask_fp, rowSum, 1, CV_REDUCE_SUM, CV_32FC1);

        for (int row_id = 0; row_id < mat.rows; ++row_id) {
          sum_of_row[row_id] = rowSum.at<float>(row_id, 0);
        }

        for (int col_id = 0; col_id < mat.cols; ++col_id) {
          sum_of_col[col_id] = colSum.at<float>(0, col_id);
        }

        auto it = std::find_if(sum_of_row.begin(), sum_of_row.end(),
                               [](int x) { return x > 0.5; });
        int y1 = std::distance(sum_of_row.begin(), it);

        auto it2 = std::find_if(sum_of_col.begin(), sum_of_col.end(),
                                [](int x) { return x > 0.5; });
        int x1 = std::distance(sum_of_col.begin(), it2);

        auto rit = std::find_if(sum_of_row.rbegin(), sum_of_row.rend(),
                                [](int x) { return x > 0.5; });
        int y2 = std::distance(rit, sum_of_row.rend());

        auto rit2 = std::find_if(sum_of_col.rbegin(), sum_of_col.rend(),
                                 [](int x) { return x > 0.5; });
        int x2 = std::distance(rit2, sum_of_col.rend());

        result_item.rect = {x1, y1, x2, y2};
        result_item.mask = global_mask;

        result->push_back(result_item);
        valid_bbox_count++;
      }
    }
    bbox_num->push_back(valid_bbox_count);
  }
}

void ObjectDetector::Predict(const std::vector<cv::Mat> imgs,
                             const double threshold, const int warmup,
                             const int repeats,
                             std::vector<PaddleDetection::ObjectResult> *result,
                             std::vector<int> *bbox_num,
                             std::vector<double> *times) {
  auto preprocess_start = std::chrono::steady_clock::now();
  int batch_size = imgs.size();

  // in_data_batch
  std::vector<float> in_data_all;
  std::vector<float> im_shape_all(batch_size * 2);
  std::vector<float> scale_factor_all(batch_size * 2);
  std::vector<const float *> output_data_list_;
  std::vector<int> out_bbox_num_data_;
  std::vector<int> out_mask_data_;

  // these parameters are for SOLOv2 output
  std::vector<float> out_score_data_;
  std::vector<uint8_t> out_global_mask_data_;
  std::vector<int64_t> out_label_data_;

  // in_net img for each batch
  std::vector<cv::Mat> in_net_img_all(batch_size);

  // Preprocess image
  for (int bs_idx = 0; bs_idx < batch_size; bs_idx++) {
    cv::Mat im = imgs.at(bs_idx);
    Preprocess(im);
    im_shape_all[bs_idx * 2] = inputs_.im_shape_[0];
    im_shape_all[bs_idx * 2 + 1] = inputs_.im_shape_[1];

    scale_factor_all[bs_idx * 2] = inputs_.scale_factor_[0];
    scale_factor_all[bs_idx * 2 + 1] = inputs_.scale_factor_[1];

    in_data_all.insert(in_data_all.end(), inputs_.im_data_.begin(),
                       inputs_.im_data_.end());

    // collect in_net img
    in_net_img_all[bs_idx] = inputs_.in_net_im_;
  }

  // Pad Batch if batch size > 1
  if (batch_size > 1 && CheckDynamicInput(in_net_img_all)) {
    in_data_all.clear();
    std::vector<cv::Mat> pad_img_all = PadBatch(in_net_img_all);
    int rh = pad_img_all[0].rows;
    int rw = pad_img_all[0].cols;
    int rc = pad_img_all[0].channels();

    for (int bs_idx = 0; bs_idx < batch_size; bs_idx++) {
      cv::Mat pad_img = pad_img_all[bs_idx];
      pad_img.convertTo(pad_img, CV_32FC3);
      std::vector<float> pad_data;
      pad_data.resize(rc * rh * rw);
      float *base = pad_data.data();
      for (int i = 0; i < rc; ++i) {
        cv::extractChannel(pad_img,
                           cv::Mat(rh, rw, CV_32FC1, base + i * rh * rw), i);
      }
      in_data_all.insert(in_data_all.end(), pad_data.begin(), pad_data.end());
    }
    // update in_net_shape
    inputs_.in_net_shape_ = {static_cast<float>(rh), static_cast<float>(rw)};
  }

  auto preprocess_end = std::chrono::steady_clock::now();
  // Prepare input tensor
  auto input_names = predictor_->GetInputNames();
  for (const auto &tensor_name : input_names) {
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

  // Run predictor
  std::vector<std::vector<float>> out_tensor_list;
  std::vector<std::vector<int>> output_shape_list;
  bool is_rbox = false;
  int reg_max = 7;
  int num_class = 80;

  auto inference_start = std::chrono::steady_clock::now();
  if (config_.arch_ == "SOLOv2") {
    // warmup
    for (int i = 0; i < warmup; i++) {
      predictor_->Run();
      // Get output tensor
      auto output_names = predictor_->GetOutputNames();
      for (int j = 0; j < output_names.size(); j++) {
        auto output_tensor = predictor_->GetOutputHandle(output_names[j]);
        std::vector<int> output_shape = output_tensor->shape();
        int out_num = std::accumulate(output_shape.begin(), output_shape.end(),
                                      1, std::multiplies<int>());
        if (j == 0) {
          out_bbox_num_data_.resize(out_num);
          output_tensor->CopyToCpu(out_bbox_num_data_.data());
        } else if (j == 1) {
          out_label_data_.resize(out_num);
          output_tensor->CopyToCpu(out_label_data_.data());
        } else if (j == 2) {
          out_score_data_.resize(out_num);
          output_tensor->CopyToCpu(out_score_data_.data());
        } else if (config_.mask_ && (j == 3)) {
          out_global_mask_data_.resize(out_num);
          output_tensor->CopyToCpu(out_global_mask_data_.data());
        }
      }
    }

    inference_start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; i++) {
      predictor_->Run();
      // Get output tensor
      out_tensor_list.clear();
      output_shape_list.clear();
      auto output_names = predictor_->GetOutputNames();
      for (int j = 0; j < output_names.size(); j++) {
        auto output_tensor = predictor_->GetOutputHandle(output_names[j]);
        std::vector<int> output_shape = output_tensor->shape();
        int out_num = std::accumulate(output_shape.begin(), output_shape.end(),
                                      1, std::multiplies<int>());
        output_shape_list.push_back(output_shape);
        if (j == 0) {
          out_bbox_num_data_.resize(out_num);
          output_tensor->CopyToCpu(out_bbox_num_data_.data());
        } else if (j == 1) {
          out_label_data_.resize(out_num);
          output_tensor->CopyToCpu(out_label_data_.data());
        } else if (j == 2) {
          out_score_data_.resize(out_num);
          output_tensor->CopyToCpu(out_score_data_.data());
        } else if (config_.mask_ && (j == 3)) {
          out_global_mask_data_.resize(out_num);
          output_tensor->CopyToCpu(out_global_mask_data_.data());
        }
      }
    }
  } else {
    // warmup
    for (int i = 0; i < warmup; i++) {
      predictor_->Run();
      // Get output tensor
      auto output_names = predictor_->GetOutputNames();
      for (int j = 0; j < output_names.size(); j++) {
        auto output_tensor = predictor_->GetOutputHandle(output_names[j]);
        std::vector<int> output_shape = output_tensor->shape();
        int out_num = std::accumulate(output_shape.begin(), output_shape.end(),
                                      1, std::multiplies<int>());
        if (config_.mask_ && (j == 2)) {
          out_mask_data_.resize(out_num);
          output_tensor->CopyToCpu(out_mask_data_.data());
        } else if (output_tensor->type() == paddle_infer::DataType::INT32) {
          out_bbox_num_data_.resize(out_num);
          output_tensor->CopyToCpu(out_bbox_num_data_.data());
        } else {
          std::vector<float> out_data;
          out_data.resize(out_num);
          output_tensor->CopyToCpu(out_data.data());
          out_tensor_list.push_back(out_data);
        }
      }
    }

    inference_start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; i++) {
      predictor_->Run();
      // Get output tensor
      out_tensor_list.clear();
      output_shape_list.clear();
      auto output_names = predictor_->GetOutputNames();
      for (int j = 0; j < output_names.size(); j++) {
        auto output_tensor = predictor_->GetOutputHandle(output_names[j]);
        std::vector<int> output_shape = output_tensor->shape();
        int out_num = std::accumulate(output_shape.begin(), output_shape.end(),
                                      1, std::multiplies<int>());
        output_shape_list.push_back(output_shape);
        if (config_.mask_ && (j == 2)) {
          out_mask_data_.resize(out_num);
          output_tensor->CopyToCpu(out_mask_data_.data());
        } else if (output_tensor->type() == paddle_infer::DataType::INT32) {
          out_bbox_num_data_.resize(out_num);
          output_tensor->CopyToCpu(out_bbox_num_data_.data());
        } else {
          std::vector<float> out_data;
          out_data.resize(out_num);
          output_tensor->CopyToCpu(out_data.data());
          out_tensor_list.push_back(out_data);
        }
      }
    }
  }

  auto inference_end = std::chrono::steady_clock::now();
  auto postprocess_start = std::chrono::steady_clock::now();
  // Postprocessing result
  result->clear();
  bbox_num->clear();
  if (config_.arch_ == "PicoDet") {
    for (int i = 0; i < out_tensor_list.size(); i++) {
      if (i == 0) {
        num_class = output_shape_list[i][2];
      }
      if (i == config_.fpn_stride_.size()) {
        reg_max = output_shape_list[i][2] / 4 - 1;
      }
      float *buffer = new float[out_tensor_list[i].size()];
      memcpy(buffer, &out_tensor_list[i][0],
             out_tensor_list[i].size() * sizeof(float));
      output_data_list_.push_back(buffer);
    }
    PaddleDetection::PicoDetPostProcess(
        result, output_data_list_, config_.fpn_stride_, inputs_.im_shape_,
        inputs_.scale_factor_, config_.nms_info_["score_threshold"].as<float>(),
        config_.nms_info_["nms_threshold"].as<float>(), num_class, reg_max);
    bbox_num->push_back(result->size());
  } else if (config_.arch_ == "SOLOv2") {
    SOLOv2Postprocess(imgs, result, bbox_num, out_bbox_num_data_,
                      out_label_data_, out_score_data_, out_global_mask_data_,
                      threshold);
  } else {
    is_rbox = output_shape_list[0][output_shape_list[0].size() - 1] % 10 == 0;
    Postprocess(imgs, result, out_bbox_num_data_, out_tensor_list[0],
                out_mask_data_, is_rbox);
    for (int k = 0; k < out_bbox_num_data_.size(); k++) {
      int tmp = out_bbox_num_data_[k];
      bbox_num->push_back(tmp);
    }
  }

  auto postprocess_end = std::chrono::steady_clock::now();

  std::chrono::duration<float> preprocess_diff =
      preprocess_end - preprocess_start;
  times->push_back(static_cast<double>(preprocess_diff.count() * 1000));
  std::chrono::duration<float> inference_diff = inference_end - inference_start;
  times->push_back(
      static_cast<double>(inference_diff.count() / repeats * 1000));
  std::chrono::duration<float> postprocess_diff =
      postprocess_end - postprocess_start;
  times->push_back(static_cast<double>(postprocess_diff.count() * 1000));
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

} // namespace PaddleDetection
