/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/memory/memory.h"
#include <vector>

namespace paddle {
namespace operators {

using framework::Tensor;

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

template <typename T>
__global__ void FillConstant(T* x, int num, int fill_num) {
  CUDA_1D_KERNEL_LOOP(i, num) {
    x[i] = static_cast<T>(fill_num);
  }
}

template <typename T>
__global__ void SliceOnAxis(const T* x, const int NC_num, const int H, const int W,
                   const int axis, const int start, const int end, 
                   T* output) {
  int HW_num = H * W;
  int length = axis == 2 ? W : H;
  int sliced_len = end - start;
  int cur_HW_num = length * sliced_len;
  // slice input on H or W (axis is 2 or 3)
  CUDA_1D_KERNEL_LOOP(i, NC_num * cur_HW_num) {
    int NC_id = i / cur_HW_num;
    int HW_id = i % cur_HW_num;
    if (axis == 2){
      output[i] = x[NC_id * HW_num + start * W + HW_id];
    } else if (axis == 3) {
      int col = HW_id % sliced_len;
      int row = HW_id / sliced_len;
      output[i] = x[NC_id * HW_num + row * W + start + col];
    }
  } 
}


template <typename T>
__global__  void CornerMaxOut(const int NC_num, const int H, const int W,
                              const int axis, bool start_with_zero,
                              T* output) {
  int HW_num = H * W;
  int len = axis == 2 ? W : H;
  int len_var = axis == 3 ? W : H;
  T cur = static_cast<T>(0.);
  T next = static_cast<T>(0.);
  T max_v = static_cast<T>(0.);
  for (int ind = 1; ind < len_var; ind <<= 1) {
    int cur_num = NC_num * len * (len_var - ind);
    int start = start_with_zero ? 0 : ind;
    int end = start_with_zero ? len_var - ind : len_var;
    int sliced_len = end - start;
    int cur_HW_num = len * sliced_len;
    // compare cur and next and assign max values to output
    CUDA_1D_KERNEL_LOOP(i, NC_num * cur_HW_num) {
      int NC_id = i / cur_HW_num;
      int HW_id = i % cur_HW_num;
      
      if (axis == 2){
        cur = output[NC_id * HW_num + start * W + HW_id];
        next = output[NC_id * HW_num + (ind - start) * W + HW_id];
        max_v = cur > next ? cur : next;
        output[NC_id * HW_num + start * W + HW_id] = max_v;
      } else if (axis == 3) {
        int col = HW_id % sliced_len;
        int row = HW_id / sliced_len;
        cur = output[NC_id * HW_num + row * W + start + col];
        next = output[NC_id * HW_num + row * W + (ind - start) + col];
        max_v = cur > next ? cur : next;
        output[NC_id * HW_num + row * W + start + col] = max_v;
      }
      __syncthreads();
    }
  }
}

template <typename T>
__global__  void UpdateMaxInfo(const T* input, const int NC_num, 
                               const int H, const int W, const int axis, 
                               const int index, T* max_val, int* max_ind) {
  int length = axis == 2 ? W : H;
  int HW_num = H * W; 
  T val = static_cast<T>(0.);
  CUDA_1D_KERNEL_LOOP(i, NC_num * length) {
    int NC_id = i / length;
    int length_id = i % length;
    if (axis == 2) {
      val = input[NC_id * HW_num + index * W + length_id];
    } else if (axis == 3) {
      val = input[NC_id * HW_num + length_id * W + index];
    }
    if (val > max_val[i]) {
      max_val[i] = val;
      max_ind[i] = index;
    }
  }
}

template <typename T>
__global__  void ScatterAddOnAxis(const T* input, const int start, const int* max_ind, const int NC_num, const int H, const int W, const int axis, T* output) {
  int length = axis == 2 ? W : H;
  int HW_num = H * W;
  CUDA_1D_KERNEL_LOOP(i, NC_num * length) { 
    int NC_id = i / length;
    int length_id = i % length;
    int id_ = max_ind[i];
    if (axis == 2) {
      output[NC_id * HW_num + id_ * W + length_id] += input[NC_id * HW_num + start * W + length_id];
    } else if (axis == 3) {
      output[NC_id * HW_num + length_id * W + id_] += input[NC_id * HW_num + length_id * W + start];
    }
  }
}

}  // namespace operators
}  // namespace paddle
