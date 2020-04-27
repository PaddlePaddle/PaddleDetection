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
  CUDA_1D_KERNEL_LOOP(i, fill_num) {
    x[i] = static_cast<T>(num);
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
__global__  void MaxOut(const T* input, const int next_ind, const int NC_num,
                        const int H, const int W, const int axis, 
                        const int start, const int end, T* output) {
  int HW_num = H * W;
  int length = axis == 2 ? W : H; 
  T cur = static_cast<T>(0.);
  T next = static_cast<T>(0.);
  T max_v = static_cast<T>(0.);
  int sliced_len = end - start;
  int cur_HW_num = length * sliced_len;
  // compare cur and next and assign max values to output
  CUDA_1D_KERNEL_LOOP(i, NC_num * cur_HW_num) {
    int NC_id = i / cur_HW_num;
    int HW_id = i % cur_HW_num;
   
    if (axis == 2){
      cur = input[NC_id * HW_num + start * W + HW_id];
      next = input[NC_id * HW_num + next_ind * W + HW_id];
      max_v = cur > next ? cur : next; 
      output[NC_id * HW_num + start * W + HW_id] = max_v;
    } else if (axis == 3) {
      int col = HW_id % sliced_len;
      int row = HW_id / sliced_len;
      cur = input[NC_id * HW_num + row * W + start + col];
      next = input[NC_id * HW_num + row * W + next_ind + col];
      max_v = cur > next ? cur : next;
      output[NC_id * HW_num + row * W + start + col] = max_v;
    }
    __syncthreads();
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
    __syncthreads();
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
      platform::CudaAtomicAdd(output + NC_id * HW_num + id_ * W + length_id, input[NC_id * HW_num + start * W + length_id]);
      //output[NC_id * HW_num + id_ * W + length_id] += input[NC_id * HW_num + start * W + length_id];
    } else if (axis == 3) {
      platform::CudaAtomicAdd(output + NC_id * HW_num + length_id * W + id_, input[NC_id * HW_num + length_id * W + start]);
      //output[NC_id * HW_num + length_id * W + id_] += input[NC_id * HW_num + length_id * W + start];
    }
    __syncthreads();
  }
}

template <typename T>
__global__ void GetMaxInfo(const T* input, const int NC_num,
                           const int H, const int W, const int axis,
                           const bool reverse, T* max_val, int* max_ind,
                           int* max_map) {
   int start = 0;
   int end = axis == 2 ? H: W;
   int s = reverse ? end-1 : start;
   int e = reverse ? start-1 : end;
   int step = reverse ? -1 : 1;
   int len = axis == 2 ? W : H;
   int loc = 0;
   T val = static_cast<T>(0.);
   for (int i = s; ; ) {
     if (i == s) {
       CUDA_1D_KERNEL_LOOP(j, NC_num * len) {
         int NC_id = j / len;
         int len_id = j % len;
         if (axis == 2) {
           loc = NC_id * H * W + i * W + len_id;
         }  else if (axis == 3){
           loc = NC_id * H * W + len_id * W + i;
         }
         max_ind[j] = i;
         max_map[loc] = max_ind[j];
         max_val[j] = input[loc];   
         __syncthreads();
       }
     } else {
       CUDA_1D_KERNEL_LOOP(j, NC_num * len) {
         int NC_id = j / len;
         int len_id = j % len;
       
         if (axis == 2) {
           loc = NC_id * H * W + i * W + len_id;
         } else if (axis == 3){
           loc = NC_id * H * W + len_id * W + i;
         }
         val = input[loc];
         T max_v = max_val[j];
         if (val > max_v) {
           max_val[j] = val;
           max_map[loc] = i;
           max_ind[j] = i;
         } else {
           max_map[loc] = max_ind[j];
         }
         __syncthreads();
       }
     }
     i += step;
     if (s < e && i >= e) break;
     if (s > e && i <= e) break;
   }
}

template <typename T>
__global__ void ScatterAddFw(const T* input, const int* max_map, const int NC_num, const int H, const int W, const int axis, T* output){
  CUDA_1D_KERNEL_LOOP(i, NC_num * H * W) {
    int loc = max_map[i];
    int NC_id = i / (H * W);
    int len_id = 0;
    if (axis == 2) {
      len_id = i % W;
      output[i] = input[NC_id * H * W + loc * W + len_id];
    } else {
      len_id = i % (H * W) / W;
      output[i] = input[NC_id * H * W + len_id * W + loc];
    }
  }
}

template <typename T>
__global__ void ScatterAddBw(const T* input, const int* max_map, const int NC_num, const int H, const int W, const int axis, T* output){
  CUDA_1D_KERNEL_LOOP(i, NC_num * H * W) {
    int loc = max_map[i];
    int NC_id = i / (H * W);
    int len_id = 0;
    int offset = 0;
    if (axis == 2) {
      len_id = i % W;
      offset = NC_id * H * W + loc * W + len_id;
    } else {
      len_id = i % (H * W) / W;
      offset = NC_id * H * W + len_id * W + loc;
    }
    platform::CudaAtomicAdd(output + offset, input[i]);
  }
}

}  // namespace operators
}  // namespace paddle
