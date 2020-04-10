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

#include <glog/logging.h>

#include <iostream>
#include <string>
#include <vector>

#include "include/object_detector.h"


DEFINE_string(model_dir, "", "Path of inference model");
DEFINE_string(image_path, "", "Path of input image");
DEFINE_string(video_path, "", "Path of input video");
DEFINE_bool(use_gpu, false, "Infering with GPU or CPU");

void PredictVideo(const std::string& video_path,
                  PaddleDetection::ObjectDetector* det) {
  // Open video
  cv::VideoCapture capture;
  capture.open(video_path.c_str());
  if (!capture.isOpened()) {
    printf("can not open video : %s\n", video_path.c_str());
    return;
  }

  // Get Video info : resolution, fps
  int video_width = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_WIDTH));
  int video_height = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_HEIGHT));
  int video_fps = static_cast<int>(capture.get(CV_CAP_PROP_FPS));

  // Create VideoWriter for output
  cv::VideoWriter video_out;
  std::string video_out_path = "output.avi";
  video_out.open(video_out_path.c_str(),
                 CV_FOURCC('M', 'J', 'P', 'G'),
                 video_fps,
                 cv::Size(video_width, video_height),
                 true);
  if (!video_out.isOpened()) {
    printf("create video writer failed!\n");
    return;
  }

  std::vector<PaddleDetection::ObjectResult> result;
  auto labels = det->GetLabelList();
  auto colormap = PaddleDetection::GenerateColorMap(labels.size());
  // Capture all frames and do inference
  cv::Mat frame;
  while (capture.read(frame)) {
    if (frame.empty()) {
      break;
    }
    det->Predict(frame, &result);
    cv::Mat out_im = PaddleDetection::VisualizeResult(
        frame, result, labels, colormap);
    video_out.write(out_im);
  }
  capture.release();
  video_out.release();
}

void PredictImage(const std::string& image_path,
                  PaddleDetection::ObjectDetector* det) {
  // Open input image as an opencv cv::Mat object
  cv::Mat im = cv::imread(image_path, 1);
  // Store all detected result
  std::vector<PaddleDetection::ObjectResult> result;
  det->Predict(im, &result);
  for (const auto& item : result) {
    printf("class=%d confidence=%.2f rect=[%d %d %d %d]\n",
        item.class_id,
        item.confidence,
        item.rect[0],
        item.rect[1],
        item.rect[2],
        item.rect[3]);
  }
  // Visualization result
  auto labels = det->GetLabelList();
  auto colormap = PaddleDetection::GenerateColorMap(labels.size());
  cv::Mat vis_img = PaddleDetection::VisualizeResult(
      im, result, labels, colormap);
  cv::imwrite("output.jpeg", vis_img);
  printf("Visualized output saved as output.jpeg\n");
}

int main(int argc, char** argv) {
  // Parsing command-line
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_model_dir.empty()
      || (FLAGS_image_path.empty() && FLAGS_video_path.empty())) {
      std::cout << "Usage: ./main --model_dir=/PATH/TO/INFERENCE_MODEL/ "
                << "--image_path=/PATH/TO/INPUT/IMAGE/" << std::endl;
      return -1;
  }

  // Load model and create a object detector
  PaddleDetection::ObjectDetector det(FLAGS_model_dir, FLAGS_use_gpu);
  // Do inference on input video or image
  if (!FLAGS_video_path.empty()) {
    PredictVideo(FLAGS_video_path, &det);
  } else if (!FLAGS_image_path.empty()) {
    PredictImage(FLAGS_image_path, &det);
  }
  return 0;
}
