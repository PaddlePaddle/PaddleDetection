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
#include <ctime>

#include "include/object_detector.h"
#include <nvjpeg.h>
#include <fstream>
#include <unistd.h>


DEFINE_string(model_dir, "", "Path of inference model");
DEFINE_string(image_path, "", "Path of input image");
DEFINE_string(video_path, "", "Path of input video");
DEFINE_bool(use_gpu, false, "Infering with GPU or CPU");
DEFINE_bool(use_camera, false, "Use camera or not");
DEFINE_string(run_mode, "fluid", "Mode of running(fluid/trt_fp32/trt_fp16)");
DEFINE_int32(gpu_id, 0, "Device id of GPU to execute");
DEFINE_int32(camera_id, -1, "Device id of camera to predict");
DEFINE_bool(run_benchmark, false, "Whether to predict a image_file repeatedly for benchmark");
DEFINE_double(threshold, 0.5, "Threshold of score.");
DEFINE_string(output_dir, "output", "Directory of output visualization files.");


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
  std::string video_out_path = "output.mp4";
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
    det->Predict(frame, 0.5, 0, 1, false, &result);
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

int dev_malloc(void **p, size_t s) { return (int)cudaMalloc(p, s); }

int dev_free(void *p) { return (int)cudaFree(p); }

void PredictImage(const std::string& image_path,
                  const double threshold,
                  const bool run_benchmark,
                  PaddleDetection::ObjectDetector* det,
                  const std::string& output_dir = "output") {
  // Open input image as an opencv cv::Mat object
  clock_t imread_tic =clock();

  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  printf("stream created\n");

  nvjpegHandle_t nvjpeg_handle;
  nvjpegDevAllocator_t dev_allocator = {&dev_malloc, &dev_free};

  nvjpegStatus_t status = nvjpegCreate(NVJPEG_BACKEND_DEFAULT, &dev_allocator, &nvjpeg_handle);

  printf("handle created %d \n", status);

  nvjpegJpegState_t nvjpeg_state;
  nvjpegJpegStateCreate(nvjpeg_handle, &nvjpeg_state);

  printf("state created\n");

  std::vector<char> image_data;
  std::ifstream input(image_path.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
	std::streamsize file_size = input.tellg();
  printf("image_path %s file_size %d \n", image_path.c_str(), file_size);
	input.seekg(0, std::ios::beg);
  if (image_data.size() < file_size) {
    image_data.resize(file_size);
  }
  input.read(image_data.data(), file_size);

  printf("input created \n");

  int widths[NVJPEG_MAX_COMPONENT];
  int heights[NVJPEG_MAX_COMPONENT];
  int channels;
  nvjpegChromaSubsampling_t subsampling;
  nvjpegGetImageInfo(nvjpeg_handle, (unsigned char *)image_data.data(), image_data.size(),
                      &channels, &subsampling, widths, heights);

  printf("image info created %d %d \n", widths[0], heights[0]);

  nvjpegImage_t ibuf;
  int sz = widths[0] * heights[0];
	// alloc output buffer if required
	for (int c = 0; c < 3; c++) {
		ibuf.pitch[c] = widths[0];
    // if (ibuf.channel[c]) {
    //   cudaFree(ibuf.channel[c]);
    // }
    cudaMalloc((void **)&ibuf.channel[c], sz);
	}

  printf("buffer alloced\n");

  cudaStreamSynchronize(stream);
  status = nvjpegDecode(nvjpeg_handle, nvjpeg_state, (const unsigned char *)image_data.data(), (size_t)file_size, NVJPEG_OUTPUT_BGR, &ibuf, stream);
  printf("decode finish %d \n", status);
  clock_t imread_toc = clock();
  printf("imread time cost: %f ms\n", 1000 * (imread_toc - imread_tic) / (double)CLOCKS_PER_SEC);
  // printf("nvjpeg %d, %d, %d, %d\n", ibuf.channel[0][0], ibuf.channel[0][1], ibuf.channel[0][2], ibuf.channel[0][3]);

	// for (int c = 0; c < 3; c++) {
  //   if (ibuf.channel[c]) {
  //     cudaFree(ibuf.channel[c]);
  //   }
	// }

  clock_t memcpy_tic = clock();
  unsigned char image_data_cpu[3 * heights[0] * widths[0]];
  for (int c = 0; c < 3; c++) {
    cudaMemcpy(image_data_cpu + c * sz, ibuf.channel[c], sz, cudaMemcpyDeviceToHost);
  }
  cv::Size cv_sz(widths[0], heights[0]);
  cv::Mat im0(cv_sz, CV_8UC3, (void *)image_data_cpu);
  // cv::Mat im0(widths[0] * 3, heights[0], CV_8UC1, (void *)image_data_cpu);
  // im0 = im0.reshape(3, widths[0], heights[0]).transpose(1, 2, 0);
  printf("im0 data: %d, %d, %d, %d, %d \n", im0.data[0], im0.data[1], im0.data[2], im0.data[3], im0.data[4]);
  clock_t memcpy_toc = clock();
  printf("memcpy time cost: %f ms\n", 1000 * (memcpy_toc - memcpy_tic) / (double)CLOCKS_PER_SEC);

  cudaStreamDestroy(stream);
  nvjpegJpegStateDestroy(nvjpeg_state);
  nvjpegDestroy(nvjpeg_handle);
  // cudaDeviceReset();
  // cudaDeviceSynchronize();

  // sleep(300);

  imread_tic = clock();
  cv::Mat im = cv::imread(image_path, 1);
  printf("im data: %d, %d, %d, %d, %d \n", im.data[0], im.data[1], im.data[2], im.data[3], im.data[4]);
  imread_toc =clock();
  printf("cv imread time cost: %f ms\n", 1000 * (imread_toc - imread_tic) / (double)CLOCKS_PER_SEC);
  // Store all detected result
  clock_t predict_tic =clock();
  std::vector<PaddleDetection::ObjectResult> result;
  if (run_benchmark)
  {
    det->Predict(im, threshold, 100, 100, run_benchmark, &result);
  }else
  {
    det->Predict(im, 0.5, 0, 1, run_benchmark, &result);
    clock_t predict_toc =clock();
    printf("predict time cost: %f ms\n", 1000 * (predict_toc - predict_tic) / (double)CLOCKS_PER_SEC);
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
    cv::imwrite(output_dir + "/output.jpg", vis_img, compression_params);
    printf("Visualized output saved as output.jpg\n");
  }
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
  if (!FLAGS_video_path.empty() || FLAGS_use_camera) {
    PredictVideo(FLAGS_video_path, &det);
  } else if (!FLAGS_image_path.empty()) {
    PredictImage(FLAGS_image_path, FLAGS_threshold, FLAGS_run_benchmark, &det, FLAGS_output_dir);
  }
  return 0;
}
