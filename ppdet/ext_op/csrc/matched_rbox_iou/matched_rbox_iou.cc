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
//
// The code is based on
// https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/csrc/box_iou_rotated/

#include "../rbox_iou/rbox_iou_utils.h"
#include "paddle/extension.h"

template <typename T>
void matched_rbox_iou_cpu_kernel(const int rbox_num, const T *rbox1_data_ptr,
                                 const T *rbox2_data_ptr, T *output_data_ptr) {

  int i;
  for (i = 0; i < rbox_num; i++) {
    output_data_ptr[i] =
        rbox_iou_single<T>(rbox1_data_ptr + i * 5, rbox2_data_ptr + i * 5);
  }
}

#define CHECK_INPUT_CPU(x)                                                     \
  PD_CHECK(x.is_cpu(), #x " must be a CPU Tensor.")

std::vector<paddle::Tensor>
MatchedRboxIouCPUForward(const paddle::Tensor &rbox1,
                         const paddle::Tensor &rbox2) {
  CHECK_INPUT_CPU(rbox1);
  CHECK_INPUT_CPU(rbox2);
  PD_CHECK(rbox1.shape()[0] == rbox2.shape()[0], "inputs must be same dim");

  auto rbox_num = rbox1.shape()[0];
  auto output = paddle::empty({rbox_num}, rbox1.dtype(), paddle::CPUPlace());

  PD_DISPATCH_FLOATING_TYPES(rbox1.type(), "matched_rbox_iou_cpu_kernel", ([&] {
                               matched_rbox_iou_cpu_kernel<data_t>(
                                   rbox_num, rbox1.data<data_t>(),
                                   rbox2.data<data_t>(), output.data<data_t>());
                             }));

  return {output};
}

#ifdef PADDLE_WITH_CUDA
std::vector<paddle::Tensor>
MatchedRboxIouCUDAForward(const paddle::Tensor &rbox1,
                          const paddle::Tensor &rbox2);
#endif

#define CHECK_INPUT_SAME(x1, x2)                                               \
  PD_CHECK(x1.place() == x2.place(), "input must be smae pacle.")

std::vector<paddle::Tensor> MatchedRboxIouForward(const paddle::Tensor &rbox1,
                                                  const paddle::Tensor &rbox2) {
  CHECK_INPUT_SAME(rbox1, rbox2);
  if (rbox1.is_cpu()) {
    return MatchedRboxIouCPUForward(rbox1, rbox2);
#ifdef PADDLE_WITH_CUDA
  } else if (rbox1.is_gpu()) {
    return MatchedRboxIouCUDAForward(rbox1, rbox2);
#endif
  }
}

std::vector<std::vector<int64_t>>
MatchedRboxIouInferShape(std::vector<int64_t> rbox1_shape,
                         std::vector<int64_t> rbox2_shape) {
  return {{rbox1_shape[0]}};
}

std::vector<paddle::DataType> MatchedRboxIouInferDtype(paddle::DataType t1,
                                                       paddle::DataType t2) {
  return {t1};
}

PD_BUILD_OP(matched_rbox_iou)
    .Inputs({"RBOX1", "RBOX2"})
    .Outputs({"Output"})
    .SetKernelFn(PD_KERNEL(MatchedRboxIouForward))
    .SetInferShapeFn(PD_INFER_SHAPE(MatchedRboxIouInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MatchedRboxIouInferDtype));
