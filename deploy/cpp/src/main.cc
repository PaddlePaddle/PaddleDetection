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

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#else  // Linux/Unix
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif


#ifdef _WIN32
#define OS_PATH_SEP "\\"
#else
#define OS_PATH_SEP "/"
#endif

DEFINE_string(model_dir, "", "Path of inference model");
DEFINE_string(image_path, "", "Path of input image");
DEFINE_string(video_path, "", "Path of input video");
DEFINE_bool(use_gpu, false, "Infering with GPU or CPU");
DEFINE_bool(use_camera, false, "Use camera or not");
DEFINE_string(run_mode, "fluid", "Mode of running(fluid/trt_fp32/trt_fp16)");
DEFINE_int32(gpu_id, 0, "Device id of GPU to execute");
DEFINE_string(output_dir, "output", "Path of saved image or video");
DEFINE_int32(camera_id, -1, "Device id of camera to predict");

std::string generate_save_path(const std::string& save_dir,
                               const std::string& file_path) {
  if (access(save_dir.c_str(), 0) < 0) {
#ifdef _WIN32
    mkdir(save_dir.c_str());
#else
    if (mkdir(save_dir.c_str(), S_IRWXU) < 0) {
      std::cerr << "Fail to create " << save_dir << "directory." << std::endl;
    }
#endif
  }
  int pos = file_path.find_last_of(OS_PATH_SEP);
  std::string image_name(file_path.substr(pos + 1));
  return save_dir + OS_PATH_SEP + image_name;
}

void PredictVideo(const std::string& video_path,
                  PaddleDetection::ObjectDetector* det) {
  // Open video
  cv::VideoCapture capture;
  if (FLAGS_camera_id != -1){
    capture.open(FLAGS_camera_id);
  }else{
    capture.open(video_path.c_str());
  }
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
  std::string video_out_path = generate_save_path(FLAGS_output_dir, "output.mp4");
  video_out.open(video_out_path.c_str(),
                 0x00000021,
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
  int frame_id = 0;
  while (capture.read(frame)) {
    if (frame.empty()) {
      break;
    }
    det->Predict(frame, &result);
    cv::Mat out_im = PaddleDetection::VisualizeResult(
        frame, result, labels, colormap);
    for (const auto& item : result) {
      printf("In frame id %d, we detect: class=%d confidence=%.2f rect=[%d %d %d %d]\n",
        frame_id,
        item.class_id,
        item.confidence,
        item.rect[0],
        item.rect[1],
        item.rect[2],
        item.rect[3]);
   }   
    video_out.write(out_im);
    frame_id += 1;
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
    printf("class=%d confidence=%.4f rect=[%d %d %d %d]\n",
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
  std::vector<int> compression_params;
  compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
  compression_params.push_back(95);
  std::string output_image_path = generate_save_path(FLAGS_output_dir, "output.jpg");
  cv::imwrite(output_image_path, vis_img, compression_params);
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
  if (!(FLAGS_run_mode == "fluid" || FLAGS_run_mode == "trt_fp32"
      || FLAGS_run_mode == "trt_fp16")) {
    std::cout << "run_mode should be 'fluid', 'trt_fp32' or 'trt_fp16'.";
    return -1;
  }

  // Load model and create a object detector
  PaddleDetection::ObjectDetector det(FLAGS_model_dir, FLAGS_use_gpu,
    FLAGS_run_mode, FLAGS_gpu_id);
  // Do inference on input video or image
  if (det.GetSuccessInit()) {
    if (!FLAGS_video_path.empty() || FLAGS_camera_id != -1) {
      PredictVideo(FLAGS_video_path, &det);
    } else if (!FLAGS_image_path.empty()) {
      PredictImage(FLAGS_image_path, &det);
    }
  }
  return 0;
}
