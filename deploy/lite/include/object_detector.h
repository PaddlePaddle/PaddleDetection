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

#pragma once

#include <ctime>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "paddle_api.h"  // NOLINT

#include "include/config_parser.h"
#include "include/preprocess_op.h"

using namespace paddle::lite_api;  // NOLINT

namespace PaddleDetection {
// Object Detection Result
struct ObjectResult {
  // Rectangle coordinates of detected object: left, right, top, down
  std::vector<int> rect;
  // Class id of detected object
  int class_id;
  // Confidence of detected object
  float confidence;
};

// Generate visualization colormap for each class
std::vector<int> GenerateColorMap(int num_class);

// Visualiztion Detection Result
cv::Mat VisualizeResult(const cv::Mat& img,
                        const std::vector<ObjectResult>& results,
                        const std::vector<std::string>& lables,
                        const std::vector<int>& colormap,
                        const bool is_rbox);

class ObjectDetector {
 public:
  explicit ObjectDetector(const std::string& model_dir,
                          int cpu_threads = 1,
                          const int batch_size = 1) {
    config_.load_config(model_dir);
    printf("config created\n");
    threshold_ = config_.draw_threshold_;
    preprocessor_.Init(config_.preprocess_info_);
    printf("before object detector\n");
    LoadModel(model_dir, cpu_threads);
    printf("create object detector\n");
  }

  // Load Paddle inference model
  void LoadModel(std::string model_file, int num_theads);

  // Run predictor
  void Predict(const std::vector<cv::Mat>& imgs,
               const double threshold = 0.5,
               const int warmup = 0,
               const int repeats = 1,
               std::vector<ObjectResult>* result = nullptr,
               std::vector<int>* bbox_num = nullptr,
               std::vector<double>* times = nullptr);

  // Get Model Label list
  const std::vector<std::string>& GetLabelList() const {
    return config_.label_list_;
  }

 private:
  // Preprocess image and copy data to input buffer
  void Preprocess(const cv::Mat& image_mat);
  // Postprocess result
  void Postprocess(const std::vector<cv::Mat> mats,
                   std::vector<ObjectResult>* result,
                   std::vector<int> bbox_num,
                   bool is_rbox);

  std::shared_ptr<PaddlePredictor> predictor_;
  Preprocessor preprocessor_;
  ImageBlob inputs_;
  std::vector<float> output_data_;
  std::vector<const float *> output_data_list_;
  std::vector<int> out_bbox_num_data_;
  float threshold_;
  ConfigPaser config_;

  int reg_max_ = 7;
  int num_class_ = 80;

  inline float fast_exp(float x)
  {
      union
      {
          uint32_t i;
          float f;
      } v{};
      v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
      return v.f;
  }

  template <typename _Tp>
  int activation_function_softmax(const _Tp *src, _Tp *dst, int length)
  {
      const _Tp alpha = *std::max_element(src, src + length);
      _Tp denominator{0};

      for (int i = 0; i < length; ++i)
      {
          dst[i] = fast_exp(src[i] - alpha);
          denominator += dst[i];
      }

      for (int i = 0; i < length; ++i)
      {
          dst[i] /= denominator;
      }

      return 0;
  }

  void nms(std::vector<ObjectResult> &input_boxes, float NMS_THRESH)
  {
      std::sort(input_boxes.begin(),
		input_boxes.end(), 
		[](ObjectResult a, ObjectResult b) { 
	return a.confidence > b.confidence; });
      std::vector<float> vArea(input_boxes.size());
      for (int i = 0; i < int(input_boxes.size()); ++i)
      {
          vArea[i] = (input_boxes.at(i).rect[2] - input_boxes.at(i).rect[0] + 1) 
		  * (input_boxes.at(i).rect[3] - input_boxes.at(i).rect[1] + 1);
      }
      for (int i = 0; i < int(input_boxes.size()); ++i)
      {
          for (int j = i + 1; j < int(input_boxes.size());)
          {
              float xx1 = (std::max)(input_boxes[i].rect[0], input_boxes[j].rect[0]);
              float yy1 = (std::max)(input_boxes[i].rect[1], input_boxes[j].rect[1]);
              float xx2 = (std::min)(input_boxes[i].rect[2], input_boxes[j].rect[2]);
              float yy2 = (std::min)(input_boxes[i].rect[3], input_boxes[j].rect[3]);
              float w = (std::max)(float(0), xx2 - xx1 + 1);
              float h = (std::max)(float(0), yy2 - yy1 + 1);
              float inter = w * h;
              float ovr = inter / (vArea[i] + vArea[j] - inter);
              if (ovr >= NMS_THRESH)
              {
                  input_boxes.erase(input_boxes.begin() + j);
                  vArea.erase(vArea.begin() + j);
              }
              else
              {
                  j++;
              }
          }
      }
  }

  // Picodet decode
  ObjectResult disPred2Bbox(const float *&dfl_det, int label, float score,
                       int x, int y, int stride)
  {
      float ct_x = (x + 0.5) * stride;
      float ct_y = (y + 0.5) * stride;
      std::vector<float> dis_pred;
      dis_pred.resize(4);
      for (int i = 0; i < 4; i++)
      {
        float dis = 0;
        float* dis_after_sm = new float[reg_max_ + 1];
        activation_function_softmax(dfl_det + i * (reg_max_ + 1), dis_after_sm, reg_max_ + 1);
        for (int j = 0; j < reg_max_ + 1; j++)
        {
            dis += j * dis_after_sm[j];
        }
        dis *= stride;
        dis_pred[i] = dis;
        delete[] dis_after_sm;
      }
      int xmin = (int)(std::max)(ct_x - dis_pred[0], .0f);
      int ymin = (int)(std::max)(ct_y - dis_pred[1], .0f);
      int xmax = (int)(std::min)(ct_x + dis_pred[2], (float)inputs_.im_shape_[0]);
      int ymax = (int)(std::min)(ct_y + dis_pred[3], (float)inputs_.im_shape_[1]);

      ObjectResult result_item;
      result_item.rect = {xmin, ymin, xmax, ymax};
      result_item.class_id = label;
      result_item.confidence = score;

      return result_item;
  }

  void picodet_postprocess(std::vector<ObjectResult>* results,
                           std::vector<const float *> outs)
  {
    std::vector<std::vector<ObjectResult>> bbox_results;
    bbox_results.resize(num_class_);
    int in_h = inputs_.im_shape_[0], in_w = inputs_.im_shape_[1];
	  for (int i = 0; i < config_.fpn_stride_.size(); ++i) {
      int feature_h = in_h / config_.fpn_stride_[i];
      int feature_w = in_w / config_.fpn_stride_[i];
      for (int idx = 0; idx < feature_h * feature_w; idx++)
      {
        const float *scores = outs[i] + (idx * num_class_);

        int row = idx / feature_w;
        int col = idx % feature_w;
        float score = 0;
        int cur_label = 0;
        for (int label = 0; label < num_class_; label++)
        {
          if (scores[label] > score)
          {
            score = scores[label];
            cur_label = label;
          }
        }
        if (score > config_.nms_info_["score_threshold"].as<float>())
        {
          const float *bbox_pred = outs[i + config_.fpn_stride_.size()]
		            + (idx * 4 * (reg_max_ + 1));
          bbox_results[cur_label].push_back(disPred2Bbox(bbox_pred, 
		            cur_label, score, col, row, config_.fpn_stride_[i]));
        }
      }
    }
    for (int i = 0; i < (int)bbox_results.size(); i++)
    {
      nms(bbox_results[i], config_.nms_info_["nms_threshold"].as<float>());

      for (auto box : bbox_results[i])
      {
          box.rect[0] = box.rect[0] / inputs_.scale_factor_[1];
          box.rect[2] = box.rect[2] / inputs_.scale_factor_[1];
          box.rect[1] = box.rect[1] / inputs_.scale_factor_[0];
          box.rect[3] = box.rect[3] / inputs_.scale_factor_[0];
          results->push_back(box);
      }
    }
  }

};

}  // namespace PaddleDetection
