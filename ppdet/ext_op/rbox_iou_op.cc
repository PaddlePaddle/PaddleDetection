/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/extension.h"

#include <vector>

std::vector<paddle::Tensor> RboxIouCPUForward(const paddle::Tensor& rbox1, const paddle::Tensor& rbox2);
std::vector<paddle::Tensor> RboxIouCUDAForward(const paddle::Tensor& rbox1, const paddle::Tensor& rbox2);


#define CHECK_INPUT_SAME(x1, x2) PD_CHECK(x1.place() == x2.place(), "input must be smae pacle.")
std::vector<paddle::Tensor> RboxIouForward(const paddle::Tensor& rbox1, const paddle::Tensor& rbox2) {
    CHECK_INPUT_SAME(rbox1, rbox2);
    if (rbox1.place() == paddle::PlaceType::kCPU) {
        return RboxIouCPUForward(rbox1, rbox2);
    }
    else if (rbox1.place() == paddle::PlaceType::kGPU) {
        return RboxIouCUDAForward(rbox1, rbox2);
    }
}

std::vector<std::vector<int64_t>> InferShape(std::vector<int64_t> rbox1_shape, std::vector<int64_t> rbox2_shape) {
    return {{rbox1_shape[0], rbox2_shape[0]}};
}

std::vector<paddle::DataType> InferDtype(paddle::DataType t1, paddle::DataType t2) {
    return {t1};
}

PD_BUILD_OP(rbox_iou)
    .Inputs({"RBOX1", "RBOX2"})
    .Outputs({"Output"})
    .SetKernelFn(PD_KERNEL(RboxIouForward))
    .SetInferShapeFn(PD_INFER_SHAPE(InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(InferDtype));
