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

template <typename T>
void nms_rotated_cpu_kernel(const T *boxes_data, const float threshold,
                            const int64_t num_boxes, int64_t *num_keep_boxes,
                            int64_t *output_data) {

  int num_masks = CeilDiv(num_boxes, 64);
  std::vector<int64_t> masks(num_masks, 0);
  for (int64_t i = 0; i < num_boxes; ++i) {
    if (masks[i / 64] & 1ULL << (i % 64))
      continue;
    T box_1[5];
    for (int k = 0; k < 5; ++k) {
      box_1[k] = boxes_data[i * 5 + k];
    }
    for (int64_t j = i + 1; j < num_boxes; ++j) {
      if (masks[j / 64] & 1ULL << (j % 64))
        continue;
      T box_2[5];
      for (int k = 0; k < 5; ++k) {
        box_2[k] = boxes_data[j * 5 + k];
      }
      if (rbox_iou_single<T>(box_1, box_2) > threshold) {
        masks[j / 64] |= 1ULL << (j % 64);
      }
    }
  }
  int64_t output_data_idx = 0;
  for (int64_t i = 0; i < num_boxes; ++i) {
    if (masks[i / 64] & 1ULL << (i % 64))
      continue;
    output_data[output_data_idx++] = i;
  }
  *num_keep_boxes = output_data_idx;
  for (; output_data_idx < num_boxes; ++output_data_idx) {
    output_data[output_data_idx] = 0;
  }
}

#define CHECK_INPUT_CPU(x)                                                     \
  PD_CHECK(x.is_cpu(), #x " must be a CPU Tensor.")

std::vector<paddle::Tensor> NMSRotatedCPUForward(const paddle::Tensor &boxes,
                                                 const paddle::Tensor &scores,
                                                 float threshold) {
  CHECK_INPUT_CPU(boxes);
  CHECK_INPUT_CPU(scores);

  auto num_boxes = boxes.shape()[0];

  auto order_t =
      std::get<1>(paddle::argsort(scores, /* axis=*/0, /* descending=*/true));
  auto boxes_sorted = paddle::gather(boxes, order_t, /* axis=*/0);

  auto keep =
      paddle::empty({num_boxes}, paddle::DataType::INT64, paddle::CPUPlace());
  int64_t num_keep_boxes = 0;

  PD_DISPATCH_FLOATING_TYPES(boxes.type(), "nms_rotated_cpu_kernel", ([&] {
                               nms_rotated_cpu_kernel<data_t>(
                                   boxes_sorted.data<data_t>(), threshold,
                                   num_boxes, &num_keep_boxes,
                                   keep.data<int64_t>());
                             }));

  keep = keep.slice(0, num_keep_boxes);
  return {paddle::gather(order_t, keep, /* axis=*/0)};
}

#ifdef PADDLE_WITH_CUDA
std::vector<paddle::Tensor> NMSRotatedCUDAForward(const paddle::Tensor &boxes,
                                                  const paddle::Tensor &scores,
                                                  float threshold);
#endif

std::vector<paddle::Tensor> NMSRotatedForward(const paddle::Tensor &boxes,
                                              const paddle::Tensor &scores,
                                              float threshold) {
  if (boxes.is_cpu()) {
    return NMSRotatedCPUForward(boxes, scores, threshold);
#ifdef PADDLE_WITH_CUDA
  } else if (boxes.is_gpu()) {
    return NMSRotatedCUDAForward(boxes, scores, threshold);
#endif
  }
}

std::vector<std::vector<int64_t>>
NMSRotatedInferShape(std::vector<int64_t> boxes_shape,
                     std::vector<int64_t> scores_shape) {
  return {{-1}};
}

std::vector<paddle::DataType> NMSRotatedInferDtype(paddle::DataType t1,
                                                   paddle::DataType t2) {
  return {paddle::DataType::INT64};
}

PD_BUILD_OP(nms_rotated)
    .Inputs({"Boxes", "Scores"})
    .Outputs({"Output"})
    .Attrs({"threshold: float"})
    .SetKernelFn(PD_KERNEL(NMSRotatedForward))
    .SetInferShapeFn(PD_INFER_SHAPE(NMSRotatedInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(NMSRotatedInferDtype));