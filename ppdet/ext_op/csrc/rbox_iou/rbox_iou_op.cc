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
//
// The code is based on https://github.com/csuhan/s2anet/blob/master/mmdet/ops/box_iou_rotated

#include "rbox_iou_op.h"
#include "paddle/extension.h"


template <typename T>
void rbox_iou_cpu_kernel(
    const int rbox1_num,
    const int rbox2_num,
    const T* rbox1_data_ptr,
    const T* rbox2_data_ptr,
    T* output_data_ptr) {

    int i, j;
    for (i = 0; i < rbox1_num; i++) {
        for (j = 0; j < rbox2_num; j++) {
		int offset = i * rbox2_num + j;
		output_data_ptr[offset] = rbox_iou_single<T>(rbox1_data_ptr + i * 5, rbox2_data_ptr + j * 5);
        }
    }
}


#define CHECK_INPUT_CPU(x) PD_CHECK(x.place() == paddle::PlaceType::kCPU, #x " must be a CPU Tensor.")

std::vector<paddle::Tensor> RboxIouCPUForward(const paddle::Tensor& rbox1, const paddle::Tensor& rbox2) {
    CHECK_INPUT_CPU(rbox1);
    CHECK_INPUT_CPU(rbox2);

    auto rbox1_num = rbox1.shape()[0];
    auto rbox2_num = rbox2.shape()[0];

    auto output = paddle::Tensor(paddle::PlaceType::kCPU, {rbox1_num, rbox2_num});

    PD_DISPATCH_FLOATING_TYPES(
        rbox1.type(),
        "rbox_iou_cpu_kernel",
        ([&] {
            rbox_iou_cpu_kernel<data_t>(
                rbox1_num,
                rbox2_num,
                rbox1.data<data_t>(),
                rbox2.data<data_t>(),
                output.mutable_data<data_t>());
        }));
    
    return {output};
}


#ifdef PADDLE_WITH_CUDA
std::vector<paddle::Tensor> RboxIouCUDAForward(const paddle::Tensor& rbox1, const paddle::Tensor& rbox2);
#endif


#define CHECK_INPUT_SAME(x1, x2) PD_CHECK(x1.place() == x2.place(), "input must be smae pacle.")

std::vector<paddle::Tensor> RboxIouForward(const paddle::Tensor& rbox1, const paddle::Tensor& rbox2) {
    CHECK_INPUT_SAME(rbox1, rbox2);
    if (rbox1.place() == paddle::PlaceType::kCPU) {
        return RboxIouCPUForward(rbox1, rbox2);
#ifdef PADDLE_WITH_CUDA
    } else if (rbox1.place() == paddle::PlaceType::kGPU) {
        return RboxIouCUDAForward(rbox1, rbox2);
#endif
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
