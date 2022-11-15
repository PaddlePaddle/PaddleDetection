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
__global__ void
matched_rbox_iou_cuda_kernel(const int rbox_num, const T *rbox1_data_ptr,
                             const T *rbox2_data_ptr, T *output_data_ptr) {
  for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < rbox_num;
       tid += blockDim.x * gridDim.x) {
    output_data_ptr[tid] =
        rbox_iou_single<T>(rbox1_data_ptr + tid * 5, rbox2_data_ptr + tid * 5);
  }
}

#define CHECK_INPUT_GPU(x)                                                     \
  PD_CHECK(x.is_gpu(), #x " must be a GPU Tensor.")

std::vector<paddle::Tensor>
MatchedRboxIouCUDAForward(const paddle::Tensor &rbox1,
                          const paddle::Tensor &rbox2) {
  CHECK_INPUT_GPU(rbox1);
  CHECK_INPUT_GPU(rbox2);
  PD_CHECK(rbox1.shape()[0] == rbox2.shape()[0], "inputs must be same dim");

  auto rbox_num = rbox1.shape()[0];

  auto output = paddle::empty({rbox_num}, rbox1.dtype(), paddle::GPUPlace());

  const int thread_per_block = 512;
  const int block_per_grid = CeilDiv(rbox_num, thread_per_block);

  PD_DISPATCH_FLOATING_TYPES(
      rbox1.type(), "matched_rbox_iou_cuda_kernel", ([&] {
        matched_rbox_iou_cuda_kernel<
            data_t><<<block_per_grid, thread_per_block, 0, rbox1.stream()>>>(
            rbox_num, rbox1.data<data_t>(), rbox2.data<data_t>(),
            output.data<data_t>());
      }));

  return {output};
}
