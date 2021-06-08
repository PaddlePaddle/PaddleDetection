// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "paddle_api.h"  // NOLINT


using namespace paddle::lite_api;  // NOLINT
using namespace std;

struct Object {
  cv::Rect rec;
  int class_id;
  float prob;
};

// Object for storing all preprocessed data
struct ImageBlob {
  // image width and height
  std::vector<int> im_shape_;
  // Buffer for image data after preprocessing
  const float* im_data_;
  std::vector<float> mean_;
  std::vector<float> scale_;
};

void PrintBenchmarkLog(std::vector<double> det_time,
                       std::map<std::string, std::string> config,
                       int img_num) {
  std::cout << "----------------- Config info ------------------" << std::endl;
  std::cout << "runtime_device: armv8" << std::endl;
  std::cout << "precision: " << config.at("precision") << std::endl;

  std::cout << "num_threads: " << config.at("num_threads") << std::endl;
  std::cout << "---------------- Data info ---------------------" << std::endl;
  std::cout << "batch_size: " << 1 << std::endl;
  std::cout << "---------------- Model info --------------------" << std::endl;
  std::cout << "Model_name: " << config.at("model_file") << std::endl;
  std::cout << "---------------- Perf info ---------------------" << std::endl;
  std::cout << "Total number of predicted data: " << img_num
            << " and total time spent(s): "
            << std::accumulate(det_time.begin(), det_time.end(), 0) << std::endl;
  std::cout << "preproce_time(ms): " << det_time[0] / img_num
            << ", inference_time(ms): " << det_time[1] / img_num
            << ", postprocess_time(ms): " << det_time[2] << std::endl;
}

std::vector<std::string> LoadLabels(const std::string &path) {
  std::ifstream file;
  std::vector<std::string> labels;
  file.open(path);
  while (file) {
    std::string line;
    std::getline(file, line);
    std::string::size_type pos = line.find(" ");
    if (pos != std::string::npos) {
      line = line.substr(pos);
    }
    labels.push_back(line);
  }
  file.clear();
  file.close();
  return labels;
}

std::vector<std::string> ReadDict(std::string path) {
  std::ifstream in(path);
  std::string filename;
  std::string line;
  std::vector<std::string> m_vec;
  if (in) {
    while (getline(in, line)) {
      m_vec.push_back(line);
    }
  } else {
    std::cout << "no such file" << std::endl;
  }
  return m_vec;
}

std::vector<std::string> split(const std::string &str,
                               const std::string &delim) {
  std::vector<std::string> res;
  if ("" == str)
    return res;
  char *strs = new char[str.length() + 1];
  std::strcpy(strs, str.c_str());

  char *d = new char[delim.length() + 1];
  std::strcpy(d, delim.c_str());

  char *p = std::strtok(strs, d);
  while (p) {
    string s = p;
    res.push_back(s);
    p = std::strtok(NULL, d);
  }

  return res;
}

std::map<std::string, std::string> LoadConfigTxt(std::string config_path) {
  auto config = ReadDict(config_path);

  std::map<std::string, std::string> dict;
  for (int i = 0; i < config.size(); i++) {
    std::vector<std::string> res = split(config[i], " ");
    dict[res[0]] = res[1];
  }
  return dict;
}

void PrintConfig(const std::map<std::string, std::string> &config) {
  std::cout << "=======PaddleDetection lite demo config======" << std::endl;
  for (auto iter = config.begin(); iter != config.end(); iter++) {
    std::cout << iter->first << " : " << iter->second << std::endl;
  }
  std::cout << "===End of PaddleDetection lite demo config===" << std::endl;
}


// fill tensor with mean and scale and trans layout: nhwc -> nchw, neon speed up
void neon_mean_scale(const float* din,
                     float* dout,
                     int size,
                     const std::vector<float> mean,
                     const std::vector<float> scale) {
  if (mean.size() != 3 || scale.size() != 3) {
    std::cerr << "[ERROR] mean or scale size must equal to 3\n";
    exit(1);
  }
  float32x4_t vmean0 = vdupq_n_f32(mean[0]);
  float32x4_t vmean1 = vdupq_n_f32(mean[1]);
  float32x4_t vmean2 = vdupq_n_f32(mean[2]);
  float32x4_t vscale0 = vdupq_n_f32(1.f / scale[0]);
  float32x4_t vscale1 = vdupq_n_f32(1.f / scale[1]);
  float32x4_t vscale2 = vdupq_n_f32(1.f / scale[2]);
  float* dout_c0 = dout;
  float* dout_c1 = dout + size;
  float* dout_c2 = dout + size * 2;
  int i = 0;
  for (; i < size - 3; i += 4) {
    float32x4x3_t vin3 = vld3q_f32(din);
    float32x4_t vsub0 = vsubq_f32(vin3.val[0], vmean0);
    float32x4_t vsub1 = vsubq_f32(vin3.val[1], vmean1);
    float32x4_t vsub2 = vsubq_f32(vin3.val[2], vmean2);
    float32x4_t vs0 = vmulq_f32(vsub0, vscale0);
    float32x4_t vs1 = vmulq_f32(vsub1, vscale1);
    float32x4_t vs2 = vmulq_f32(vsub2, vscale2);
    vst1q_f32(dout_c0, vs0);
    vst1q_f32(dout_c1, vs1);
    vst1q_f32(dout_c2, vs2);

    din += 12;
    dout_c0 += 4;
    dout_c1 += 4;
    dout_c2 += 4;
  }
  for (; i < size; i++) {
    *(dout_c0++) = (*(din++) - mean[0]) * scale[0];
    *(dout_c0++) = (*(din++) - mean[1]) * scale[1];
    *(dout_c0++) = (*(din++) - mean[2]) * scale[2];
  }
}

std::vector<Object> visualize_result(
                        const float* data,
                        int count,
                        float thresh,
                        cv::Mat& image,
                        const std::vector<std::string> &class_names) {
  if (data == nullptr) {
    std::cerr << "[ERROR] data can not be nullptr\n";
    exit(1);
  }
  std::vector<Object> rect_out;
  for (int iw = 0; iw < count; iw++) {
    int oriw = image.cols;
    int orih = image.rows;
    if (data[1] > thresh) {
      Object obj;
      int x = static_cast<int>(data[2]);
      int y = static_cast<int>(data[3]);
      int w = static_cast<int>(data[4] - data[2] + 1);
      int h = static_cast<int>(data[5] - data[3] + 1);
      cv::Rect rec_clip =
          cv::Rect(x, y, w, h) & cv::Rect(0, 0, image.cols, image.rows);
      obj.class_id = static_cast<int>(data[0]);
      obj.prob = data[1];
      obj.rec = rec_clip;
      if (w > 0 && h > 0 && obj.prob <= 1) {
        rect_out.push_back(obj);
        cv::rectangle(image, rec_clip, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        std::string str_prob = std::to_string(obj.prob);
        std::string text = std::string(class_names[obj.class_id]) + ": " +
                           str_prob.substr(0, str_prob.find(".") + 4);
        int font_face = cv::FONT_HERSHEY_COMPLEX_SMALL;
        double font_scale = 1.f;
        int thickness = 1;
        cv::Size text_size =
            cv::getTextSize(text, font_face, font_scale, thickness, nullptr);
        float new_font_scale = w * 0.5 * font_scale / text_size.width;
        text_size = cv::getTextSize(
            text, font_face, new_font_scale, thickness, nullptr);
        cv::Point origin;
        origin.x = x + 3;
        origin.y = y + text_size.height + 3;
        cv::putText(image,
                    text,
                    origin,
                    font_face,
                    new_font_scale,
                    cv::Scalar(0, 255, 255),
                    thickness,
                    cv::LINE_AA);

        std::cout << "detection, image size: " << image.cols << ", "
                  << image.rows
                  << ", detect object: " << class_names[obj.class_id]
                  << ", score: " << obj.prob << ", location: x=" << x
                  << ", y=" << y << ", width=" << w << ", height=" << h
                  << std::endl;
      }
    }
    data += 6;
  }
  return rect_out;
}

// Load Model and create model predictor
std::shared_ptr<PaddlePredictor> LoadModel(std::string model_file,
                                           int num_theads) {
  MobileConfig config;
  config.set_threads(num_theads);
  config.set_model_from_file(model_file);

  std::shared_ptr<PaddlePredictor> predictor =
      CreatePaddlePredictor<MobileConfig>(config);
  return predictor;
}

ImageBlob prepare_imgdata(const cv::Mat& img,
                          std::map<std::string,
                          std::string> config) {
  ImageBlob img_data;
  std::vector<int> target_size_;
  std::vector<std::string> size_str = split(config.at("Resize"), ",");
  transform(size_str.begin(), size_str.end(), back_inserter(target_size_),
            [](std::string const& s){return stoi(s);});
  int width = target_size_[0];
  int height = target_size_[1];
  img_data.im_shape_ = {
      static_cast<int>(target_size_[0]),
      static_cast<int>(target_size_[1])
  };

  std::vector<float> mean_;
  std::vector<float> scale_;
  std::vector<std::string> mean_str = split(config.at("mean"), ",");
  std::vector<std::string> std_str = split(config.at("std"), ",");
  transform(mean_str.begin(), mean_str.end(), back_inserter(mean_),
            [](std::string const& s){return stof(s);});
  transform(std_str.begin(), std_str.end(), back_inserter(scale_),
            [](std::string const& s){return stof(s);});
  img_data.mean_ = mean_;
  img_data.scale_ = scale_;
  return img_data;
}


void preprocess(const cv::Mat& img, const ImageBlob img_data, float* data) {
  cv::Mat rgb_img;
  cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
  cv::resize(
      rgb_img, rgb_img, cv::Size(img_data.im_shape_[0],img_data.im_shape_[1]),
      0.f, 0.f, cv::INTER_CUBIC);
  cv::Mat imgf;
  rgb_img.convertTo(imgf, CV_32FC3, 1 / 255.f);
  const float* dimg = reinterpret_cast<const float*>(imgf.data);
  neon_mean_scale(
    dimg, data, int(img_data.im_shape_[0] * img_data.im_shape_[1]),
    img_data.mean_, img_data.scale_);
}


void RunModel(std::map<std::string, std::string> config,
              std::string img_path,
              const int repeats,
              std::vector<double>* times) {

  std::string model_file = config.at("model_file");
  std::string label_path = config.at("label_path");
  // Load Labels
  std::vector<std::string> class_names = LoadLabels(label_path);

  auto predictor = LoadModel(model_file, stoi(config.at("num_threads")));
  cv::Mat img = imread(img_path, cv::IMREAD_COLOR);
  auto img_data = prepare_imgdata(img, config);

  auto preprocess_start = std::chrono::steady_clock::now();
  // 1. Prepare input data from image
  // input 0
  std::unique_ptr<Tensor> input_tensor0(std::move(predictor->GetInput(0)));
  input_tensor0->Resize({1, 3, img_data.im_shape_[0], img_data.im_shape_[1]});
  auto* data0 = input_tensor0->mutable_data<float>();
  preprocess(img, img_data, data0);

  // input1
  std::unique_ptr<Tensor> input_tensor1(std::move(predictor->GetInput(1)));
  input_tensor1->Resize({1, 2});
  auto* data1 = input_tensor1->mutable_data<int>();
  data1[0] = img_data.im_shape_[0];
  data1[1] = img_data.im_shape_[1];

  auto preprocess_end = std::chrono::steady_clock::now();

  // 2. Run predictor
  // warm up
  for (int i = 0; i < repeats / 2; i++)
  {
    predictor->Run();
  }

  auto inference_start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeats; i++)
  {
    predictor->Run();
  }
  auto inference_end = std::chrono::steady_clock::now();
  // 3. Get output and post process
  auto postprocess_start = std::chrono::steady_clock::now();
  std::unique_ptr<const Tensor> output_tensor(
      std::move(predictor->GetOutput(0)));
  const float* outptr = output_tensor->data<float>();
  auto shape_out = output_tensor->shape();
  int64_t cnt = 1;
  for (auto& i : shape_out) {
    cnt *= i;
  }
  auto rec_out = visualize_result(
      outptr, static_cast<int>(cnt / 6), 0.5f, img, class_names);
  std::string result_name =
      img_path.substr(0, img_path.find(".")) + "_result.jpg";
  cv::imwrite(result_name, img);
  auto postprocess_end = std::chrono::steady_clock::now();
  std::chrono::duration<float> prep_diff = preprocess_end - preprocess_start;
  times->push_back(double(prep_diff.count() * 1000));
  std::chrono::duration<float> infer_diff = inference_end - inference_start;
  times->push_back(double(infer_diff.count() / repeats * 1000));
  std::chrono::duration<float> post_diff = postprocess_end - postprocess_start;
  times->push_back(double(post_diff.count() * 1000));
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "[ERROR] usage: " << argv[0] << " config_path image_path\n";
    exit(1);
  }
  std::string config_path = argv[1];
  std::string img_path = argv[2];

  // load config
  auto config = LoadConfigTxt(config_path);
  PrintConfig(config);

  bool enable_benchmark = bool(stoi(config.at("enable_benchmark")));
  int repeats = enable_benchmark ? 50 : 1;
  std::vector<double> det_times;
  RunModel(config, img_path, repeats, &det_times);
  PrintBenchmarkLog(det_times, config, 1);
  return 0;
}
