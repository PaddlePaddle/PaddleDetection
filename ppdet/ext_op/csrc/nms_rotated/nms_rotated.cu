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

#include "../rbox_iou/rbox_iou_utils.h"
#include "paddle/extension.h"

static const int64_t threadsPerBlock = sizeof(int64_t) * 8;

template <typename T>
__global__ void
nms_rotated_cuda_kernel(const T *boxes_data, const float threshold,
                        const int64_t num_boxes, int64_t *masks) {
  auto raw_start = blockIdx.y;
  auto col_start = blockIdx.x;
  if (raw_start > col_start)
    return;
  const int raw_last_storage =
      min(num_boxes - raw_start * threadsPerBlock, threadsPerBlock);
  const int col_last_storage =
      min(num_boxes - col_start * threadsPerBlock, threadsPerBlock);
  if (threadIdx.x < raw_last_storage) {
    int64_t mask = 0;
    auto current_box_idx = raw_start * threadsPerBlock + threadIdx.x;
    const T *current_box = boxes_data + current_box_idx * 5;
    for (int i = 0; i < col_last_storage; ++i) {
      const T *target_box = boxes_data + (col_start * threadsPerBlock + i) * 5;
      if (rbox_iou_single<T>(current_box, target_box) > threshold) {
        mask |= 1ULL << i;
      }
    }
    const int blocks_per_line = CeilDiv(num_boxes, threadsPerBlock);
    masks[current_box_idx * blocks_per_line + col_start] = mask;
  }
}

#define CHECK_INPUT_GPU(x)                                                     \
  PD_CHECK(x.is_gpu(), #x " must be a GPU Tensor.")

std::vector<paddle::Tensor> NMSRotatedCUDAForward(const paddle::Tensor &boxes,
                                                  const paddle::Tensor &scores,
                                                  float threshold) {
  CHECK_INPUT_GPU(boxes);
  CHECK_INPUT_GPU(scores);

  auto num_boxes = boxes.shape()[0];
  auto order_t =
      std::get<1>(paddle::argsort(scores, /* axis=*/0, /* descending=*/true));
  auto boxes_sorted = paddle::gather(boxes, order_t, /* axis=*/0);

  const auto blocks_per_line = CeilDiv(num_boxes, threadsPerBlock);
  dim3 block(threadsPerBlock);
  dim3 grid(blocks_per_line, blocks_per_line);
  auto mask_dev = paddle::empty({num_boxes * blocks_per_line},
                                paddle::DataType::INT64, paddle::GPUPlace());

  PD_DISPATCH_FLOATING_TYPES(
      boxes.type(), "nms_rotated_cuda_kernel", ([&] {
        nms_rotated_cuda_kernel<data_t><<<grid, block, 0, boxes.stream()>>>(
            boxes_sorted.data<data_t>(), threshold, num_boxes,
            mask_dev.data<int64_t>());
      }));

  auto mask_host = mask_dev.copy_to(paddle::CPUPlace(), true);
  auto keep_host =
      paddle::empty({num_boxes}, paddle::DataType::INT64, paddle::CPUPlace());
  int64_t *keep_host_ptr = keep_host.data<int64_t>();
  int64_t *mask_host_ptr = mask_host.data<int64_t>();
  std::vector<int64_t> remv(blocks_per_line);
  int64_t last_box_num = 0;
  for (int64_t i = 0; i < num_boxes; ++i) {
    auto remv_element_id = i / threadsPerBlock;
    auto remv_bit_id = i % threadsPerBlock;
    if (!(remv[remv_element_id] & 1ULL << remv_bit_id)) {
      keep_host_ptr[last_box_num++] = i;
      int64_t *current_mask = mask_host_ptr + i * blocks_per_line;
      for (auto j = remv_element_id; j < blocks_per_line; ++j) {
        remv[j] |= current_mask[j];
      }
    }
  }

  keep_host = keep_host.slice(0, last_box_num);
  auto keep_dev = keep_host.copy_to(paddle::GPUPlace(), true);
  return {paddle::gather(order_t, keep_dev, /* axis=*/0)};
}