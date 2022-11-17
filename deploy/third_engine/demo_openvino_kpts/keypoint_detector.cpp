//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include "keypoint_detector.h"

namespace PaddleDetection {

// Visualiztion MaskDetector results
cv::Mat VisualizeKptsResult(const cv::Mat& img,
                            const std::vector<KeyPointResult>& results,
                            const std::vector<int>& colormap,
                            float threshold) {
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
      if (results[batchid].keypoints[i * 3] > threshold) {
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
      if (results[batchid].keypoints[edge[i][0] * 3] > threshold &&
          results[batchid].keypoints[edge[i][1] * 3] > threshold) {
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
  }
  return vis_img;
}

void KeyPointDetector::Postprocess(std::vector<float>& output,
                                   std::vector<uint64_t>& output_shape,
                                   std::vector<float>& idxout,
                                   std::vector<uint64_t>& idx_shape,
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
                    this->use_dark());
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
                               std::vector<KeyPointResult>* result) {
  int batch_size = imgs.size();
  auto insize = 3 * in_h * in_w;

  InferenceEngine::Blob::Ptr input_blob = infer_request_.GetBlob(input_name_);
  // Preprocess image
  InferenceEngine::MemoryBlob::Ptr mblob =
      InferenceEngine::as<InferenceEngine::MemoryBlob>(input_blob);
  if (!mblob) {
    THROW_IE_EXCEPTION
        << "We expect blob to be inherited from MemoryBlob in matU8ToBlob, "
        << "but by fact we were not able to cast inputBlob to MemoryBlob";
  }
  auto mblobHolder = mblob->wmap();
  float* blob_data = mblobHolder.as<float*>();

  cv::Mat resized_im;
  for (int bs_idx = 0; bs_idx < batch_size; bs_idx++) {
    cv::Mat im = imgs.at(bs_idx);

    cv::resize(im, resized_im, cv::Size(in_w, in_h));
    for (size_t c = 0; c < 3; c++) {
      for (size_t h = 0; h < in_h; h++) {
        for (size_t w = 0; w < in_w; w++) {
          blob_data[c * in_w * in_h + h * in_w + w] =
              (float)resized_im.at<cv::Vec3b>(h, w)[c];
        }
      }
    }
  }
  // Run predictor
  auto inference_start = std::chrono::steady_clock::now();
  // do inference
  infer_request_.Infer();

  InferenceEngine::Blob::Ptr output_blob =
      infer_request_.GetBlob("conv2d_441.tmp_1");
  auto output_shape = output_blob->getTensorDesc().getDims();
  InferenceEngine::MemoryBlob::Ptr moutput =
      InferenceEngine::as<InferenceEngine::MemoryBlob>(output_blob);

  if (moutput) {
    // locked memory holder should be alive all time while access to its
    // buffer happens
    auto minputHolder = moutput->rmap();

    auto data = minputHolder.as<const InferenceEngine::PrecisionTrait<
        InferenceEngine::Precision::FP32>::value_type*>();

    // Calculate output length
    int output_size = 1;
    for (int j = 0; j < output_shape.size(); ++j) {
      output_size *= output_shape[j];
    }
    
    output_data_.resize(output_size);
    std::copy_n(data, output_size, output_data_.data());

  }


  InferenceEngine::Blob::Ptr output_blob2 =
      infer_request_.GetBlob("argmax_0.tmp_0");
  auto idx_shape = output_blob2->getTensorDesc().getDims();
  InferenceEngine::MemoryBlob::Ptr moutput2 =
      InferenceEngine::as<InferenceEngine::MemoryBlob>(output_blob2);

  if (moutput2) {
    // locked memory holder should be alive all time while access to its
    // buffer happens
    auto minputHolder = moutput2->rmap();
    // Original I64 precision was converted to I32
    auto data = minputHolder.as<const InferenceEngine::PrecisionTrait<
        InferenceEngine::Precision::FP32>::value_type*>();

    // Calculate output length
    int output_size = 1;
    for (int j = 0; j < idx_shape.size(); ++j) {
      output_size *= idx_shape[j];
    }

    idx_data_.resize(output_size);
    std::copy_n(data, output_size, idx_data_.data());
  }

  auto inference_end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed = inference_end - inference_start;
  printf("keypoint inference time: %f s\n", elapsed.count());

  // Postprocessing result
  Postprocess(output_data_,
              output_shape,
              idx_data_,
              idx_shape,
              result,
              center_bs,
              scale_bs);
}

}  // namespace PaddleDetection
