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

#include "include/utils.h"

namespace PaddleDetection {

void nms(std::vector<ObjectResult> &input_boxes, float nms_threshold) {
  std::sort(input_boxes.begin(),
  input_boxes.end(), 
  [](ObjectResult a, ObjectResult b) { return a.confidence > b.confidence; });
  std::vector<float> vArea(input_boxes.size());
  for (int i = 0; i < int(input_boxes.size()); ++i) {
    vArea[i] = (input_boxes.at(i).rect[2] - input_boxes.at(i).rect[0] + 1) 
            * (input_boxes.at(i).rect[3] - input_boxes.at(i).rect[1] + 1);
  }
  for (int i = 0; i < int(input_boxes.size()); ++i) {
    for (int j = i + 1; j < int(input_boxes.size());) {
      float xx1 = (std::max)(input_boxes[i].rect[0], input_boxes[j].rect[0]);
      float yy1 = (std::max)(input_boxes[i].rect[1], input_boxes[j].rect[1]);
      float xx2 = (std::min)(input_boxes[i].rect[2], input_boxes[j].rect[2]);
      float yy2 = (std::min)(input_boxes[i].rect[3], input_boxes[j].rect[3]);
      float w = (std::max)(float(0), xx2 - xx1 + 1);
      float h = (std::max)(float(0), yy2 - yy1 + 1);
      float inter = w * h;
      float ovr = inter / (vArea[i] + vArea[j] - inter);
      if (ovr >= nms_threshold) {
          input_boxes.erase(input_boxes.begin() + j);
          vArea.erase(vArea.begin() + j);
      }
      else {
          j++;
      }
    }
  }
}

}  // namespace PaddleDetection
