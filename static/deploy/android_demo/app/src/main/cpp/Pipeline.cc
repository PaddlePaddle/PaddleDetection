// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "Pipeline.h"

Detector::Detector(const std::string &modelDir, const std::string &labelPath,
                   const int cpuThreadNum, const std::string &cpuPowerMode,
                   int inputWidth, int inputHeight,
                   const std::vector<float> &inputMean,
                   const std::vector<float> &inputStd, float scoreThreshold)
    : inputWidth_(inputWidth), inputHeight_(inputHeight), inputMean_(inputMean),
      inputStd_(inputStd), scoreThreshold_(scoreThreshold) {
  paddle::lite_api::MobileConfig config;
  config.set_model_from_file(modelDir + "/model.nb");
  config.set_threads(cpuThreadNum);
  config.set_power_mode(ParsePowerMode(cpuPowerMode));
  predictor_ =
      paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(
          config);
  labelList_ = LoadLabelList(labelPath);
  colorMap_ = GenerateColorMap(labelList_.size());
}

std::vector<std::string> Detector::LoadLabelList(const std::string &labelPath) {
  std::ifstream file;
  std::vector<std::string> labels;
  file.open(labelPath);
  while (file) {
    std::string line;
    std::getline(file, line);
    labels.push_back(line);
  }
  file.clear();
  file.close();
  return labels;
}

std::vector<cv::Scalar> Detector::GenerateColorMap(int numOfClasses) {
  std::vector<cv::Scalar> colorMap = std::vector<cv::Scalar>(numOfClasses);
  for (int i = 0; i < numOfClasses; i++) {
    int j = 0;
    int label = i;
    int R = 0, G = 0, B = 0;
    while (label) {
      R |= (((label >> 0) & 1) << (7 - j));
      G |= (((label >> 1) & 1) << (7 - j));
      B |= (((label >> 2) & 1) << (7 - j));
      j++;
      label >>= 3;
    }
    colorMap[i] = cv::Scalar(R, G, B);
  }
  return colorMap;
}

void Detector::Preprocess(const cv::Mat &rgbaImage) {
  // Set the data of input image
  auto inputTensor = predictor_->GetInput(0);
  std::vector<int64_t> inputShape = {1, 3, inputHeight_, inputWidth_};
  inputTensor->Resize(inputShape);
  auto inputData = inputTensor->mutable_data<float>();
  cv::Mat resizedRGBAImage;
  cv::resize(rgbaImage, resizedRGBAImage,
          cv::Size(inputShape[3], inputShape[2]));
  cv::Mat resizedRGBImage;
  cv::cvtColor(resizedRGBAImage, resizedRGBImage, cv::COLOR_BGRA2RGB);
  resizedRGBImage.convertTo(resizedRGBImage, CV_32FC3, 1.0 / 255.0f);
  NHWC3ToNC3HW(reinterpret_cast<const float *>(resizedRGBImage.data), inputData,
               inputMean_.data(), inputStd_.data(), inputShape[3],
               inputShape[2]);
  // Set the size of input image
  auto sizeTensor = predictor_->GetInput(1);
  sizeTensor->Resize({1, 2});
  auto sizeData = sizeTensor->mutable_data<int32_t>();
  sizeData[0] = inputShape[3];
  sizeData[1] = inputShape[2];
}

void Detector::Postprocess(std::vector<RESULT> *results) {
  auto outputTensor = predictor_->GetOutput(0);
  auto outputData = outputTensor->data<float>();
  auto outputShape = outputTensor->shape();
  int outputSize = ShapeProduction(outputShape);
  for (int i = 0; i < outputSize; i += 6) {
    // Class id
    auto class_id = static_cast<int>(round(outputData[i]));
    // Confidence score
    auto score = outputData[i + 1];
    if (score < scoreThreshold_)
      continue;
    RESULT object;
    object.class_name = class_id >= 0 && class_id < labelList_.size()
                            ? labelList_[class_id]
                            : "Unknow";
    object.fill_color = class_id >= 0 && class_id < colorMap_.size()
                            ? colorMap_[class_id]
                            : cv::Scalar(0, 0, 0);
    object.score = score;
    object.x = outputData[i + 2] / inputWidth_;
    object.y = outputData[i + 3] / inputHeight_;
    object.w = (outputData[i + 4] - outputData[i + 2] + 1) / inputWidth_;
    object.h = (outputData[i + 5] - outputData[i + 3] + 1) / inputHeight_;
    results->push_back(object);
  }
}

void Detector::Predict(const cv::Mat &rgbaImage, std::vector<RESULT> *results,
                       double *preprocessTime, double *predictTime,
                       double *postprocessTime) {
  auto t = GetCurrentTime();

  t = GetCurrentTime();
  Preprocess(rgbaImage);
  *preprocessTime = GetElapsedTime(t);
  LOGD("Detector postprocess costs %f ms", *preprocessTime);

  t = GetCurrentTime();
  predictor_->Run();
  *predictTime = GetElapsedTime(t);
  LOGD("Detector predict costs %f ms", *predictTime);

  t = GetCurrentTime();
  Postprocess(results);
  *postprocessTime = GetElapsedTime(t);
  LOGD("Detector postprocess costs %f ms", *postprocessTime);
}

Pipeline::Pipeline(const std::string &modelDir, const std::string &labelPath,
                   const int cpuThreadNum, const std::string &cpuPowerMode,
                   int inputWidth, int inputHeight,
                   const std::vector<float> &inputMean,
                   const std::vector<float> &inputStd, float scoreThreshold) {
  detector_.reset(new Detector(modelDir, labelPath, cpuThreadNum, cpuPowerMode,
                               inputWidth, inputHeight, inputMean, inputStd,
                               scoreThreshold));
}

void Pipeline::VisualizeResults(const std::vector<RESULT> &results,
                                cv::Mat *rgbaImage) {
  int w = rgbaImage->cols;
  int h = rgbaImage->rows;
  for (int i = 0; i < results.size(); i++) {
    RESULT object = results[i];
    cv::Rect boundingBox =
        cv::Rect(object.x * w, object.y * h, object.w * w, object.h * h) &
        cv::Rect(0, 0, w - 1, h - 1);
    // Configure text size
    std::string text = object.class_name + " ";
    text += std::to_string(static_cast<int>(object.score * 100)) + "%";
    int fontFace = cv::FONT_HERSHEY_PLAIN;
    double fontScale = 1.5f;
    float fontThickness = 1.0f;
    cv::Size textSize =
        cv::getTextSize(text, fontFace, fontScale, fontThickness, nullptr);
    // Draw roi object, text, and background
    cv::rectangle(*rgbaImage, boundingBox, object.fill_color, 2);
    cv::rectangle(*rgbaImage,
                  cv::Point2d(boundingBox.x,
                              boundingBox.y - round(textSize.height * 1.25f)),
                  cv::Point2d(boundingBox.x + boundingBox.width, boundingBox.y),
                  object.fill_color, -1);
    cv::putText(*rgbaImage, text, cv::Point2d(boundingBox.x, boundingBox.y),
                fontFace, fontScale, cv::Scalar(255, 255, 255), fontThickness);
  }
}

void Pipeline::VisualizeStatus(double readGLFBOTime, double writeGLTextureTime,
                               double preprocessTime, double predictTime,
                               double postprocessTime, cv::Mat *rgbaImage) {
  char text[255];
  cv::Scalar fontColor = cv::Scalar(255, 255, 255);
  int fontFace = cv::FONT_HERSHEY_PLAIN;
  double fontScale = 1.f;
  float fontThickness = 1;
  sprintf(text, "Read GLFBO time: %.1f ms", readGLFBOTime);
  cv::Size textSize =
      cv::getTextSize(text, fontFace, fontScale, fontThickness, nullptr);
  textSize.height *= 1.25f;
  cv::Point2d offset(10, textSize.height + 15);
  cv::putText(*rgbaImage, text, offset, fontFace, fontScale, fontColor,
              fontThickness);
  sprintf(text, "Write GLTexture time: %.1f ms", writeGLTextureTime);
  offset.y += textSize.height;
  cv::putText(*rgbaImage, text, offset, fontFace, fontScale, fontColor,
              fontThickness);
  sprintf(text, "Preprocess time: %.1f ms", preprocessTime);
  offset.y += textSize.height;
  cv::putText(*rgbaImage, text, offset, fontFace, fontScale, fontColor,
              fontThickness);
  sprintf(text, "Predict time: %.1f ms", predictTime);
  offset.y += textSize.height;
  cv::putText(*rgbaImage, text, offset, fontFace, fontScale, fontColor,
              fontThickness);
  sprintf(text, "Postprocess time: %.1f ms", postprocessTime);
  offset.y += textSize.height;
  cv::putText(*rgbaImage, text, offset, fontFace, fontScale, fontColor,
              fontThickness);
}

bool Pipeline::Process(int inTexureId, int outTextureId, int textureWidth,
                       int textureHeight, std::string savedImagePath) {
  static double readGLFBOTime = 0, writeGLTextureTime = 0;
  double preprocessTime = 0, predictTime = 0, postprocessTime = 0;

  // Read pixels from FBO texture to CV image
  cv::Mat rgbaImage;
  CreateRGBAImageFromGLFBOTexture(textureWidth, textureHeight, &rgbaImage,
                                  &readGLFBOTime);

  // Feed the image, run inference and parse the results
  std::vector<RESULT> results;
  detector_->Predict(rgbaImage, &results, &preprocessTime, &predictTime,
                     &postprocessTime);

  // Visualize the objects to the origin image
  VisualizeResults(results, &rgbaImage);

  // Visualize the status(performance data) to the origin image
  VisualizeStatus(readGLFBOTime, writeGLTextureTime, preprocessTime,
                  predictTime, postprocessTime, &rgbaImage);

  // Dump modified image if savedImagePath is set
  if (!savedImagePath.empty()) {
    cv::Mat bgrImage;
    cv::cvtColor(rgbaImage, bgrImage, cv::COLOR_RGBA2BGR);
    imwrite(savedImagePath, bgrImage);
  }

  // Write back to texture2D
  WriteRGBAImageBackToGLTexture(rgbaImage, outTextureId, &writeGLTextureTime);
  return true;
}
