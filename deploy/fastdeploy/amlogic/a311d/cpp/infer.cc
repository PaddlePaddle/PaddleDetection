// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "fastdeploy/vision.h"
#ifdef WIN32
const char sep = '\\';
#else
const char sep = '/';
#endif

void InitAndInfer(const std::string& model_dir, const std::string& image_file) {
  auto model_file = model_dir + sep + "model.pdmodel";
  auto params_file = model_dir + sep + "model.pdiparams";
  auto config_file = model_dir + sep + "infer_cfg.yml";
  auto subgraph_file = model_dir + sep + "subgraph.txt";
  fastdeploy::vision::EnableFlyCV(); 
  fastdeploy::RuntimeOption option;
  option.UseTimVX();
  option.SetLiteSubgraphPartitionPath(subgraph_file);

  auto model = fastdeploy::vision::detection::PPYOLOE(model_file, params_file,
                                                      config_file, option);
  assert(model.Initialized());

  auto im = cv::imread(image_file);

  fastdeploy::vision::DetectionResult res;
  if (!model.Predict(im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }

  std::cout << res.Str() << std::endl;

  auto vis_im = fastdeploy::vision::VisDetection(im, res, 0.5);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;

}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << "Usage: infer_demo path/to/quant_model "
                 "path/to/image "
                 "e.g ./infer_demo ./PPYOLOE_L_quant ./test.jpeg"
              << std::endl;
    return -1;
  }

  std::string model_dir = argv[1];
  std::string test_image = argv[2];
  InitAndInfer(model_dir, test_image);
  return 0;
}
