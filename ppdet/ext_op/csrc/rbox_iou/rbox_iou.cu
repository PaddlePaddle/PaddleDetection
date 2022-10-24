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
// The code is based on
// https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/csrc/box_iou_rotated/

#include "paddle/extension.h"
#include "rbox_iou_utils.h"

// 2D block with 32 * 16 = 512 threads per block
const int BLOCK_DIM_X = 32;
const int BLOCK_DIM_Y = 16;

template <typename T>
__global__ void rbox_iou_cuda_kernel(const int rbox1_num, const int rbox2_num,
                                     const T *rbox1_data_ptr,
                                     const T *rbox2_data_ptr,
                                     T *output_data_ptr) {

  // get row_start and col_start
  const int rbox1_block_idx = blockIdx.x * blockDim.x;
  const int rbox2_block_idx = blockIdx.y * blockDim.y;

  const int rbox1_thread_num = min(rbox1_num - rbox1_block_idx, blockDim.x);
  const int rbox2_thread_num = min(rbox2_num - rbox2_block_idx, blockDim.y);

  __shared__ T block_boxes1[BLOCK_DIM_X * 5];
  __shared__ T block_boxes2[BLOCK_DIM_Y * 5];

  // It's safe to copy using threadIdx.x since BLOCK_DIM_X >= BLOCK_DIM_Y
  if (threadIdx.x < rbox1_thread_num && threadIdx.y == 0) {
    block_boxes1[threadIdx.x * 5 + 0] =
        rbox1_data_ptr[(rbox1_block_idx + threadIdx.x) * 5 + 0];
    block_boxes1[threadIdx.x * 5 + 1] =
        rbox1_data_ptr[(rbox1_block_idx + threadIdx.x) * 5 + 1];
    block_boxes1[threadIdx.x * 5 + 2] =
        rbox1_data_ptr[(rbox1_block_idx + threadIdx.x) * 5 + 2];
    block_boxes1[threadIdx.x * 5 + 3] =
        rbox1_data_ptr[(rbox1_block_idx + threadIdx.x) * 5 + 3];
    block_boxes1[threadIdx.x * 5 + 4] =
        rbox1_data_ptr[(rbox1_block_idx + threadIdx.x) * 5 + 4];
  }

  // threadIdx.x < BLOCK_DIM_Y=rbox2_thread_num, just use same condition as
  // above: threadIdx.y == 0
  if (threadIdx.x < rbox2_thread_num && threadIdx.y == 0) {
    block_boxes2[threadIdx.x * 5 + 0] =
        rbox2_data_ptr[(rbox2_block_idx + threadIdx.x) * 5 + 0];
    block_boxes2[threadIdx.x * 5 + 1] =
        rbox2_data_ptr[(rbox2_block_idx + threadIdx.x) * 5 + 1];
    block_boxes2[threadIdx.x * 5 + 2] =
        rbox2_data_ptr[(rbox2_block_idx + threadIdx.x) * 5 + 2];
    block_boxes2[threadIdx.x * 5 + 3] =
        rbox2_data_ptr[(rbox2_block_idx + threadIdx.x) * 5 + 3];
    block_boxes2[threadIdx.x * 5 + 4] =
        rbox2_data_ptr[(rbox2_block_idx + threadIdx.x) * 5 + 4];
  }

  // sync
  __syncthreads();

  if (threadIdx.x < rbox1_thread_num && threadIdx.y < rbox2_thread_num) {
    int offset = (rbox1_block_idx + threadIdx.x) * rbox2_num + rbox2_block_idx +
                 threadIdx.y;
    output_data_ptr[offset] = rbox_iou_single<T>(
        block_boxes1 + threadIdx.x * 5, block_boxes2 + threadIdx.y * 5);
  }
}

#define CHECK_INPUT_GPU(x)                                                     \
  PD_CHECK(x.is_gpu(), #x " must be a GPU Tensor.")

std::vector<paddle::Tensor> RboxIouCUDAForward(const paddle::Tensor &rbox1,
                                               const paddle::Tensor &rbox2) {
  CHECK_INPUT_GPU(rbox1);
  CHECK_INPUT_GPU(rbox2);

  auto rbox1_num = rbox1.shape()[0];
  auto rbox2_num = rbox2.shape()[0];

  auto output =
      paddle::empty({rbox1_num, rbox2_num}, rbox1.dtype(), paddle::GPUPlace());

  const int blocks_x = CeilDiv(rbox1_num, BLOCK_DIM_X);
  const int blocks_y = CeilDiv(rbox2_num, BLOCK_DIM_Y);

  dim3 blocks(blocks_x, blocks_y);
  dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y);

  PD_DISPATCH_FLOATING_TYPES(
      rbox1.type(), "rbox_iou_cuda_kernel", ([&] {
        rbox_iou_cuda_kernel<data_t><<<blocks, threads, 0, rbox1.stream()>>>(
            rbox1_num, rbox2_num, rbox1.data<data_t>(), rbox2.data<data_t>(),
            output.data<data_t>());
      }));

  return {output};
}
