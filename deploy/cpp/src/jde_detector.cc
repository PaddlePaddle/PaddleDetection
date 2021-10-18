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
#include <iomanip>
#include <chrono>
#include "include/jde_detector.h"


using namespace paddle_infer;

namespace PaddleDetection {

// Load Model and create model predictor
void JDEDetector::LoadModel(const std::string& model_dir,
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
    if (run_mode != "fluid") {
      auto precision = paddle_infer::Config::Precision::kFloat32;
      if (run_mode == "trt_fp32") {
        precision = paddle_infer::Config::Precision::kFloat32;
      }
      else if (run_mode == "trt_fp16") {
        precision = paddle_infer::Config::Precision::kHalf;
      }
      else if (run_mode == "trt_int8") {
        precision = paddle_infer::Config::Precision::kInt8;
      } else {
          printf("run_mode should be 'fluid', 'trt_fp32', 'trt_fp16' or 'trt_int8'");
      }
      // set tensorrt
      config.EnableTensorRtEngine(
          1 << 30,
          batch_size,
          this->min_subgraph_size_,
          precision,
          false,
          this->trt_calib_mode_);

      // set use dynamic shape
      if (this->use_dynamic_shape_) {
        // set DynamicShsape for image tensor
        const std::vector<int> min_input_shape = {1, 3, this->trt_min_shape_, this->trt_min_shape_};
        const std::vector<int> max_input_shape = {1, 3, this->trt_max_shape_, this->trt_max_shape_};
        const std::vector<int> opt_input_shape = {1, 3, this->trt_opt_shape_, this->trt_opt_shape_};
        const std::map<std::string, std::vector<int>> map_min_input_shape = {{"image", min_input_shape}};
        const std::map<std::string, std::vector<int>> map_max_input_shape = {{"image", max_input_shape}};
        const std::map<std::string, std::vector<int>> map_opt_input_shape = {{"image", opt_input_shape}};

        config.SetTRTDynamicShapeInfo(map_min_input_shape,
                                      map_max_input_shape,
                                      map_opt_input_shape);
        std::cout << "TensorRT dynamic shape enabled" << std::endl;
      }
    }

  } else if (this->device_ == "XPU"){
    config.EnableXpu(10*1024*1024);
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

// Visualiztion results
cv::Mat VisualizeTrackResult(const cv::Mat& img,
                        const MOT_Result& results,
                        const float fps, const int frame_id) {
  cv::Mat vis_img = img.clone();
  int im_h = img.rows;
  int im_w = img.cols;
  float text_scale = std::max(1, int(im_w / 1600.));
  float text_thickness = 2.;
  float line_thickness = std::max(1, int(im_w / 500.));

  std::ostringstream oss;
  oss << std::setiosflags(std::ios::fixed) << std::setprecision(4);
  oss << "frame: " << frame_id<<" ";
  oss << "fps: " << fps<<" ";
  oss << "num: " << results.size();
  std::string text = oss.str();

  cv::Point origin;
  origin.x = 0;
  origin.y = int(15 * text_scale);
  cv::putText(
        vis_img,
        text,
        origin,
        cv::FONT_HERSHEY_PLAIN,
        text_scale, (0, 0, 255), 2);

  for (int i = 0; i < results.size(); ++i) {
    const int obj_id = results[i].ids;
    const float score = results[i].score;
    
    cv::Scalar color = GetColor(obj_id);

    cv::Point pt1 = cv::Point(results[i].rects.left, results[i].rects.top);
    cv::Point pt2 = cv::Point(results[i].rects.right, results[i].rects.bottom);
    cv::Point id_pt = cv::Point(results[i].rects.left, results[i].rects.top + 10);
    cv::Point score_pt = cv::Point(results[i].rects.left, results[i].rects.top - 10);
    cv::rectangle(vis_img, pt1, pt2, color, line_thickness);

    std::ostringstream idoss;
    idoss << std::setiosflags(std::ios::fixed) << std::setprecision(4);
    idoss << obj_id;
    std::string id_text = idoss.str();

    cv::putText(vis_img,
                id_text,
                id_pt,
                cv::FONT_HERSHEY_PLAIN,
                text_scale,
                cv::Scalar(0, 255, 255),
                text_thickness);

    std::ostringstream soss;
    soss << std::setiosflags(std::ios::fixed) << std::setprecision(2);
    soss << score;
    std::string score_text = soss.str();

    cv::putText(vis_img,
                score_text,
                score_pt,
                cv::FONT_HERSHEY_PLAIN,
                text_scale,
                cv::Scalar(0, 255, 255),
                text_thickness);
   
  }
  return vis_img;
}


void FilterDets(const float conf_thresh, const cv::Mat dets, std::vector<int>* index) {
  for (int i = 0; i < dets.rows; ++i) {
    float score = *dets.ptr<float>(i, 4);
    if (score > conf_thresh) {
      index->push_back(i);
    }
  }
}

void JDEDetector::Preprocess(const cv::Mat& ori_im) {
  // Clone the image : keep the original mat for postprocess
  cv::Mat im = ori_im.clone();
  preprocessor_.Run(&im, &inputs_);
}

void JDEDetector::Postprocess(
    const cv::Mat dets, const cv::Mat emb,
    MOT_Result* result) {
  result->clear();
  std::vector<Track> tracks;
  std::vector<int> valid;
  FilterDets(conf_thresh_, dets, &valid);
  cv::Mat new_dets, new_emb;
  for (int i = 0; i < valid.size(); ++i) {
    new_dets.push_back(dets.row(valid[i]));
    new_emb.push_back(emb.row(valid[i]));
  }
  JDETracker::instance()->update(new_dets, new_emb, tracks);
  if (tracks.size() == 0) {
    MOT_Track mot_track;
    MOT_Rect ret = {*dets.ptr<float>(0, 0), 
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
          MOT_Track mot_track;
          MOT_Rect ret = {titer->ltrb[0],
                          titer->ltrb[1],
                          titer->ltrb[2],
                          titer->ltrb[3]};
          mot_track.rects = ret;
          mot_track.score = titer->score;
          mot_track.ids = titer->id;
          result->push_back(mot_track);
        }
      }
    } 
  }
}

void JDEDetector::Predict(const std::vector<cv::Mat> imgs,
      const double threshold,
      const int warmup,
      const int repeats,
      MOT_Result* result,
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
    in_data_all.insert(in_data_all.end(), inputs_.im_data_.begin(), inputs_.im_data_.end());
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
  // warmup
  for (int i = 0; i < warmup; i++)
  {
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
  }
  
  auto inference_start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeats; i++)
  {
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
  }
  auto inference_end = std::chrono::steady_clock::now();
  auto postprocess_start = std::chrono::steady_clock::now();
  // Postprocessing result
  result->clear();

  cv::Mat dets(bbox_shape[0], 6, CV_32FC1, bbox_data_.data());
  cv::Mat emb(bbox_shape[0], emb_shape[1], CV_32FC1, emb_data_.data());

  Postprocess(dets, emb, result);

  auto postprocess_end = std::chrono::steady_clock::now();

  std::chrono::duration<float> preprocess_diff = preprocess_end - preprocess_start;
  (*times)[0] += double(preprocess_diff.count() * 1000);
  std::chrono::duration<float> inference_diff = inference_end - inference_start;
  (*times)[1] += double(inference_diff.count() * 1000);
  std::chrono::duration<float> postprocess_diff = postprocess_end - postprocess_start;
  (*times)[2] += double(postprocess_diff.count() * 1000);
}

cv::Scalar GetColor(int idx) {
  idx = idx * 3;
  cv::Scalar color = cv::Scalar((37 * idx) % 255, 
                                (17 * idx) % 255, 
                                (29 * idx) % 255);
  return color;
}

}  // namespace PaddleDetection
