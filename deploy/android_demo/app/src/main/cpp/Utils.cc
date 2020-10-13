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

#include "Utils.h"
#include <arm_neon.h>

int64_t ShapeProduction(const std::vector<int64_t> &shape) {
  int64_t res = 1;
  for (auto i : shape)
    res *= i;
  return res;
}

void NHWC3ToNC3HW(const float *src, float *dst, const float *mean,
                  const float *std, int width, int height) {
  int size = height * width;
  float32x4_t vmean0 = vdupq_n_f32(mean ? mean[0] : 0.0f);
  float32x4_t vmean1 = vdupq_n_f32(mean ? mean[1] : 0.0f);
  float32x4_t vmean2 = vdupq_n_f32(mean ? mean[2] : 0.0f);
  float32x4_t vscale0 = vdupq_n_f32(std ? (1.0f / std[0]) : 1.0f);
  float32x4_t vscale1 = vdupq_n_f32(std ? (1.0f / std[1]) : 1.0f);
  float32x4_t vscale2 = vdupq_n_f32(std ? (1.0f / std[2]) : 1.0f);
  float *dst_c0 = dst;
  float *dst_c1 = dst + size;
  float *dst_c2 = dst + size * 2;
  int i = 0;
  for (; i < size - 3; i += 4) {
    float32x4x3_t vin3 = vld3q_f32(src);
    float32x4_t vsub0 = vsubq_f32(vin3.val[0], vmean0);
    float32x4_t vsub1 = vsubq_f32(vin3.val[1], vmean1);
    float32x4_t vsub2 = vsubq_f32(vin3.val[2], vmean2);
    float32x4_t vs0 = vmulq_f32(vsub0, vscale0);
    float32x4_t vs1 = vmulq_f32(vsub1, vscale1);
    float32x4_t vs2 = vmulq_f32(vsub2, vscale2);
    vst1q_f32(dst_c0, vs0);
    vst1q_f32(dst_c1, vs1);
    vst1q_f32(dst_c2, vs2);
    src += 12;
    dst_c0 += 4;
    dst_c1 += 4;
    dst_c2 += 4;
  }
  for (; i < size; i++) {
    *(dst_c0++) = (*(src++) - mean[0]) / std[0];
    *(dst_c1++) = (*(src++) - mean[1]) / std[1];
    *(dst_c2++) = (*(src++) - mean[2]) / std[2];
  }
}

void NHWC1ToNC1HW(const float *src, float *dst, const float *mean,
                  const float *std, int width, int height) {
  int size = height * width;
  float32x4_t vmean = vdupq_n_f32(mean ? mean[0] : 0.0f);
  float32x4_t vscale = vdupq_n_f32(std ? (1.0f / std[0]) : 1.0f);
  int i = 0;
  for (; i < size - 3; i += 4) {
    float32x4_t vin = vld1q_f32(src);
    float32x4_t vsub = vsubq_f32(vin, vmean);
    float32x4_t vs = vmulq_f32(vsub, vscale);
    vst1q_f32(dst, vs);
    src += 4;
    dst += 4;
  }
  for (; i < size; i++) {
    *(dst++) = (*(src++) - mean[0]) / std[0];
  }
}
