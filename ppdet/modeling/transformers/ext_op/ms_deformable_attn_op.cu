/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#define CUDA_KERNEL_LOOP(i, n)                                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);                 \
       i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N, const int num_threads) {
  return (N + num_threads - 1) / num_threads;
}

// forward bilinear
template <typename data_t>
__device__ data_t deformable_attn_bilinear_forward(
    const data_t *&bottom_data, const int &height, const int &width,
    const int &nheads, const int &channels, const data_t &h, const data_t &w,
    const int &m, const int &c) {
  const int h_low = floor(h);
  const int w_low = floor(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const data_t lh = h - h_low;
  const data_t lw = w - w_low;
  const data_t hh = 1 - lh, hw = 1 - lw;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;

  data_t v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
  }
  data_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
  }
  data_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
  }
  data_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
  }

  const data_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  const data_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

// forward kernel
template <typename data_t>
__global__ void deformable_attn_cuda_kernel_forward(
    const int n, const data_t *data_value, const int64_t *data_spatial_shapes,
    const int64_t *data_level_start_index, const data_t *data_sampling_loc,
    const data_t *data_attn_weight, const int batch_size,
    const int value_length, const int num_heads, const int channels,
    const int num_levels, const int query_length, const int num_points,
    data_t *output_data_ptr) {
  CUDA_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % query_length;
    _temp /= query_length;
    const int b_col = _temp;

    data_t *data_ptr = output_data_ptr + index;
    int data_weight_ptr = sampling_index * num_levels * num_points;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * value_length * qid_stride;
    data_t col = 0;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const data_t *data_value_ptr = data_value + (data_value_ptr_init_offset +
                                                   level_start_id * qid_stride);
      for (int p_col = 0; p_col < num_points; ++p_col) {
        const data_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const data_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const data_t weight = data_attn_weight[data_weight_ptr];

        const data_t h_im = loc_h * spatial_h - 0.5;
        const data_t w_im = loc_w * spatial_w - 0.5;

        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          col += deformable_attn_bilinear_forward(
                     data_value_ptr, spatial_h, spatial_w, num_heads, channels,
                     h_im, w_im, m_col, c_col) *
                 weight;
        }

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
      }
    }
    *data_ptr = col;
  }
}

#define CHECK_INPUT_GPU(x) PD_CHECK(x.is_gpu(), #x " must be a GPU Tensor.")
// forward
std::vector<paddle::Tensor>
MSDeformableAttnCUDAForward(const paddle::Tensor &value,
                            const paddle::Tensor &value_spatial_shapes,
                            const paddle::Tensor &value_level_start_index,
                            const paddle::Tensor &sampling_locations,
                            const paddle::Tensor &attention_weights) {

  CHECK_INPUT_GPU(value);
  CHECK_INPUT_GPU(value_spatial_shapes);
  CHECK_INPUT_GPU(value_level_start_index);
  CHECK_INPUT_GPU(sampling_locations);
  CHECK_INPUT_GPU(attention_weights);

  const int batch_size = value.shape()[0];
  const int value_length = value.shape()[1];
  const int num_heads = value.shape()[2];
  const int channels = value.shape()[3];

  const int num_levels = value_spatial_shapes.shape()[0];
  const int query_length = sampling_locations.shape()[1];
  const int num_points = sampling_locations.shape()[4];

  auto output = paddle::full({batch_size, query_length, num_heads * channels},
                             0, value.dtype(), paddle::GPUPlace());

  const int num_kernels = batch_size * query_length * num_heads * channels;
  deformable_attn_cuda_kernel_forward<float>
      <<<GET_BLOCKS(num_kernels, CUDA_NUM_THREADS), CUDA_NUM_THREADS, 0,
         value.stream()>>>(num_kernels, value.data<float>(),
                           value_spatial_shapes.data<int64_t>(),
                           value_level_start_index.data<int64_t>(),
                           sampling_locations.data<float>(),
                           attention_weights.data<float>(), batch_size,
                           value_length, num_heads, channels, num_levels,
                           query_length, num_points, output.data<float>());
  return {output};
}

// backward bilinear
template <typename data_t>
__device__ void deformable_attn_bilinear_backward(
    const data_t *&bottom_data, const int &height, const int &width,
    const int &nheads, const int &channels, const data_t &h, const data_t &w,
    const int &m, const int &c, const data_t &top_grad,
    const data_t &attn_weight, data_t *&grad_value, data_t *grad_sampling_loc,
    data_t *grad_attn_weight) {
  const int h_low = floor(h);
  const int w_low = floor(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const data_t lh = h - h_low;
  const data_t lw = w - w_low;
  const data_t hh = 1 - lh, hw = 1 - lw;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;

  const data_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  const data_t top_grad_value = top_grad * attn_weight;
  data_t grad_h_weight = 0, grad_w_weight = 0;

  data_t v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
    grad_h_weight -= hw * v1;
    grad_w_weight -= hh * v1;
    atomicAdd(grad_value + ptr1, w1 * top_grad_value);
  }
  data_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
    grad_h_weight -= lw * v2;
    grad_w_weight += hh * v2;
    atomicAdd(grad_value + ptr2, w2 * top_grad_value);
  }
  data_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
    grad_h_weight += hw * v3;
    grad_w_weight -= lh * v3;
    atomicAdd(grad_value + ptr3, w3 * top_grad_value);
  }
  data_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
    grad_h_weight += lw * v4;
    grad_w_weight += lh * v4;
    atomicAdd(grad_value + ptr4, w4 * top_grad_value);
  }

  const data_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  *grad_attn_weight = top_grad * val;
  *grad_sampling_loc = width * grad_w_weight * top_grad_value;
  *(grad_sampling_loc + 1) = height * grad_h_weight * top_grad_value;
}

template <typename data_t>
__device__ void deformable_attn_bilinear_backward_gm(
    const data_t *&bottom_data, const int &height, const int &width,
    const int &nheads, const int &channels, const data_t &h, const data_t &w,
    const int &m, const int &c, const data_t &top_grad,
    const data_t &attn_weight, data_t *&grad_value, data_t *grad_sampling_loc,
    data_t *grad_attn_weight) {
  const int h_low = floor(h);
  const int w_low = floor(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const data_t lh = h - h_low;
  const data_t lw = w - w_low;
  const data_t hh = 1 - lh, hw = 1 - lw;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;

  const data_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  const data_t top_grad_value = top_grad * attn_weight;
  data_t grad_h_weight = 0, grad_w_weight = 0;

  data_t v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
    grad_h_weight -= hw * v1;
    grad_w_weight -= hh * v1;
    atomicAdd(grad_value + ptr1, w1 * top_grad_value);
  }
  data_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
    grad_h_weight -= lw * v2;
    grad_w_weight += hh * v2;
    atomicAdd(grad_value + ptr2, w2 * top_grad_value);
  }
  data_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
    grad_h_weight += hw * v3;
    grad_w_weight -= lh * v3;
    atomicAdd(grad_value + ptr3, w3 * top_grad_value);
  }
  data_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
    grad_h_weight += lw * v4;
    grad_w_weight += lh * v4;
    atomicAdd(grad_value + ptr4, w4 * top_grad_value);
  }

  const data_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  atomicAdd(grad_attn_weight, top_grad * val);
  atomicAdd(grad_sampling_loc, width * grad_w_weight * top_grad_value);
  atomicAdd(grad_sampling_loc + 1, height * grad_h_weight * top_grad_value);
}

// backward kernels
// channels > 1024
template <typename data_t>
__global__ void deformable_attn_cuda_kernel_backward_shm_reduce_v2_multi_blocks(
    const int n, const data_t *grad_col, const data_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const data_t *data_sampling_loc, const data_t *data_attn_weight,
    const int batch_size, const int value_length, const int num_heads,
    const int channels, const int num_levels, const int query_length,
    const int num_points, data_t *grad_value, data_t *grad_sampling_loc,
    data_t *grad_attn_weight) {
  CUDA_KERNEL_LOOP(index, n) {
    extern __shared__ int _s[];
    data_t *cache_grad_sampling_loc = (data_t *)_s;
    data_t *cache_grad_attn_weight = cache_grad_sampling_loc + 2 * blockDim.x;
    unsigned int tid = threadIdx.x;
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % query_length;
    _temp /= query_length;
    const int b_col = _temp;

    const data_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_points;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr << 1;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * value_length * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const data_t *data_value_ptr = data_value + value_ptr_offset;
      data_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_points; ++p_col) {
        const data_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const data_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const data_t weight = data_attn_weight[data_weight_ptr];

        const data_t h_im = loc_h * spatial_h - 0.5;
        const data_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc + (threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight + threadIdx.x) = 0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          deformable_attn_bilinear_backward(
              data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im,
              w_im, m_col, c_col, top_grad, weight, grad_value_ptr,
              cache_grad_sampling_loc + (threadIdx.x << 1),
              cache_grad_attn_weight + threadIdx.x);
        }

        __syncthreads();

        for (unsigned int s = blockDim.x / 2, spre = blockDim.x; s > 0;
             s >>= 1, spre >>= 1) {
          if (tid < s) {
            const unsigned int xid1 = tid << 1;
            const unsigned int xid2 = (tid + s) << 1;
            cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
            cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2];
            cache_grad_sampling_loc[xid1 + 1] +=
                cache_grad_sampling_loc[xid2 + 1];
            if (tid + (s << 1) < spre) {
              cache_grad_attn_weight[tid] +=
                  cache_grad_attn_weight[tid + (s << 1)];
              cache_grad_sampling_loc[xid1] +=
                  cache_grad_sampling_loc[xid2 + (s << 1)];
              cache_grad_sampling_loc[xid1 + 1] +=
                  cache_grad_sampling_loc[xid2 + 1 + (s << 1)];
            }
          }
          __syncthreads();
        }

        if (tid == 0) {
          atomicAdd(grad_sampling_loc, cache_grad_sampling_loc[0]);
          atomicAdd(grad_sampling_loc + 1, cache_grad_sampling_loc[1]);
          atomicAdd(grad_attn_weight, cache_grad_attn_weight[0]);
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}

template <typename data_t>
__global__ void deformable_attn_cuda_kernel_backward_gm(
    const int n, const data_t *grad_col, const data_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const data_t *data_sampling_loc, const data_t *data_attn_weight,
    const int batch_size, const int value_length, const int num_heads,
    const int channels, const int num_levels, const int query_length,
    const int num_points, data_t *grad_value, data_t *grad_sampling_loc,
    data_t *grad_attn_weight) {
  CUDA_KERNEL_LOOP(index, n) {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % query_length;
    _temp /= query_length;
    const int b_col = _temp;

    const data_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_points;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr << 1;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * value_length * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const data_t *data_value_ptr = data_value + value_ptr_offset;
      data_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_points; ++p_col) {
        const data_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const data_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const data_t weight = data_attn_weight[data_weight_ptr];

        const data_t h_im = loc_h * spatial_h - 0.5;
        const data_t w_im = loc_w * spatial_w - 0.5;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          deformable_attn_bilinear_backward_gm(
              data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im,
              w_im, m_col, c_col, top_grad, weight, grad_value_ptr,
              grad_sampling_loc, grad_attn_weight);
        }
        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}

// channels <= 1024
template <typename data_t, unsigned int blockSize>
__global__ void
deformable_attn_cuda_kernel_backward_shm_blocksize_aware_reduce_v1(
    const int n, const data_t *grad_col, const data_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const data_t *data_sampling_loc, const data_t *data_attn_weight,
    const int batch_size, const int value_length, const int num_heads,
    const int channels, const int num_levels, const int query_length,
    const int num_points, data_t *grad_value, data_t *grad_sampling_loc,
    data_t *grad_attn_weight) {
  CUDA_KERNEL_LOOP(index, n) {
    __shared__ data_t cache_grad_sampling_loc[blockSize * 2];
    __shared__ data_t cache_grad_attn_weight[blockSize];
    unsigned int tid = threadIdx.x;
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % query_length;
    _temp /= query_length;
    const int b_col = _temp;

    const data_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_points;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr << 1;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * value_length * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const data_t *data_value_ptr = data_value + value_ptr_offset;
      data_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_points; ++p_col) {
        const data_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const data_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const data_t weight = data_attn_weight[data_weight_ptr];

        const data_t h_im = loc_h * spatial_h - 0.5;
        const data_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc + (threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight + threadIdx.x) = 0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          deformable_attn_bilinear_backward(
              data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im,
              w_im, m_col, c_col, top_grad, weight, grad_value_ptr,
              cache_grad_sampling_loc + (threadIdx.x << 1),
              cache_grad_attn_weight + threadIdx.x);
        }

        __syncthreads();
        if (tid == 0) {
          data_t _grad_w = cache_grad_sampling_loc[0],
                 _grad_h = cache_grad_sampling_loc[1],
                 _grad_a = cache_grad_attn_weight[0];
          int sid = 2;
          for (unsigned int tid = 1; tid < blockSize; ++tid) {
            _grad_w += cache_grad_sampling_loc[sid];
            _grad_h += cache_grad_sampling_loc[sid + 1];
            _grad_a += cache_grad_attn_weight[tid];
            sid += 2;
          }

          *grad_sampling_loc = _grad_w;
          *(grad_sampling_loc + 1) = _grad_h;
          *grad_attn_weight = _grad_a;
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}

template <typename data_t, unsigned int blockSize>
__global__ void
deformable_attn_cuda_kernel_backward_shm_blocksize_aware_reduce_v2(
    const int n, const data_t *grad_col, const data_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const data_t *data_sampling_loc, const data_t *data_attn_weight,
    const int batch_size, const int value_length, const int num_heads,
    const int channels, const int num_levels, const int query_length,
    const int num_points, data_t *grad_value, data_t *grad_sampling_loc,
    data_t *grad_attn_weight) {
  CUDA_KERNEL_LOOP(index, n) {
    __shared__ data_t cache_grad_sampling_loc[blockSize * 2];
    __shared__ data_t cache_grad_attn_weight[blockSize];
    unsigned int tid = threadIdx.x;
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % query_length;
    _temp /= query_length;
    const int b_col = _temp;

    const data_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_points;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr << 1;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * value_length * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const data_t *data_value_ptr = data_value + value_ptr_offset;
      data_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_points; ++p_col) {
        const data_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const data_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const data_t weight = data_attn_weight[data_weight_ptr];

        const data_t h_im = loc_h * spatial_h - 0.5;
        const data_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc + (threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight + threadIdx.x) = 0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          deformable_attn_bilinear_backward(
              data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im,
              w_im, m_col, c_col, top_grad, weight, grad_value_ptr,
              cache_grad_sampling_loc + (threadIdx.x << 1),
              cache_grad_attn_weight + threadIdx.x);
        }

        __syncthreads();

        for (unsigned int s = blockSize / 2; s > 0; s >>= 1) {
          if (tid < s) {
            const unsigned int xid1 = tid << 1;
            const unsigned int xid2 = (tid + s) << 1;
            cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
            cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2];
            cache_grad_sampling_loc[xid1 + 1] +=
                cache_grad_sampling_loc[xid2 + 1];
          }
          __syncthreads();
        }

        if (tid == 0) {
          *grad_sampling_loc = cache_grad_sampling_loc[0];
          *(grad_sampling_loc + 1) = cache_grad_sampling_loc[1];
          *grad_attn_weight = cache_grad_attn_weight[0];
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}

template <typename data_t>
__global__ void deformable_attn_cuda_kernel_backward_shm_reduce_v1(
    const int n, const data_t *grad_col, const data_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const data_t *data_sampling_loc, const data_t *data_attn_weight,
    const int batch_size, const int value_length, const int num_heads,
    const int channels, const int num_levels, const int query_length,
    const int num_points, data_t *grad_value, data_t *grad_sampling_loc,
    data_t *grad_attn_weight) {
  CUDA_KERNEL_LOOP(index, n) {
    extern __shared__ int _s[];
    data_t *cache_grad_sampling_loc = (data_t *)_s;
    data_t *cache_grad_attn_weight = cache_grad_sampling_loc + 2 * blockDim.x;
    unsigned int tid = threadIdx.x;
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % query_length;
    _temp /= query_length;
    const int b_col = _temp;

    const data_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_points;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr << 1;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * value_length * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const data_t *data_value_ptr = data_value + value_ptr_offset;
      data_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_points; ++p_col) {
        const data_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const data_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const data_t weight = data_attn_weight[data_weight_ptr];

        const data_t h_im = loc_h * spatial_h - 0.5;
        const data_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc + (threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight + threadIdx.x) = 0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          deformable_attn_bilinear_backward(
              data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im,
              w_im, m_col, c_col, top_grad, weight, grad_value_ptr,
              cache_grad_sampling_loc + (threadIdx.x << 1),
              cache_grad_attn_weight + threadIdx.x);
        }

        __syncthreads();
        if (tid == 0) {
          data_t _grad_w = cache_grad_sampling_loc[0],
                 _grad_h = cache_grad_sampling_loc[1],
                 _grad_a = cache_grad_attn_weight[0];
          int sid = 2;
          for (unsigned int tid = 1; tid < blockDim.x; ++tid) {
            _grad_w += cache_grad_sampling_loc[sid];
            _grad_h += cache_grad_sampling_loc[sid + 1];
            _grad_a += cache_grad_attn_weight[tid];
            sid += 2;
          }

          *grad_sampling_loc = _grad_w;
          *(grad_sampling_loc + 1) = _grad_h;
          *grad_attn_weight = _grad_a;
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}

template <typename data_t>
__global__ void deformable_attn_cuda_kernel_backward_shm_reduce_v2(
    const int n, const data_t *grad_col, const data_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const data_t *data_sampling_loc, const data_t *data_attn_weight,
    const int batch_size, const int value_length, const int num_heads,
    const int channels, const int num_levels, const int query_length,
    const int num_points, data_t *grad_value, data_t *grad_sampling_loc,
    data_t *grad_attn_weight) {
  CUDA_KERNEL_LOOP(index, n) {
    extern __shared__ int _s[];
    data_t *cache_grad_sampling_loc = (data_t *)_s;
    data_t *cache_grad_attn_weight = cache_grad_sampling_loc + 2 * blockDim.x;
    unsigned int tid = threadIdx.x;
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % query_length;
    _temp /= query_length;
    const int b_col = _temp;

    const data_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_points;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr << 1;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 2;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * value_length * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col) {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const int value_ptr_offset =
          data_value_ptr_init_offset + level_start_id * qid_stride;
      const data_t *data_value_ptr = data_value + value_ptr_offset;
      data_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_points; ++p_col) {
        const data_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const data_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const data_t weight = data_attn_weight[data_weight_ptr];

        const data_t h_im = loc_h * spatial_h - 0.5;
        const data_t w_im = loc_w * spatial_w - 0.5;
        *(cache_grad_sampling_loc + (threadIdx.x << 1)) = 0;
        *(cache_grad_sampling_loc + ((threadIdx.x << 1) + 1)) = 0;
        *(cache_grad_attn_weight + threadIdx.x) = 0;
        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w) {
          deformable_attn_bilinear_backward(
              data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im,
              w_im, m_col, c_col, top_grad, weight, grad_value_ptr,
              cache_grad_sampling_loc + (threadIdx.x << 1),
              cache_grad_attn_weight + threadIdx.x);
        }

        __syncthreads();

        for (unsigned int s = blockDim.x / 2, spre = blockDim.x; s > 0;
             s >>= 1, spre >>= 1) {
          if (tid < s) {
            const unsigned int xid1 = tid << 1;
            const unsigned int xid2 = (tid + s) << 1;
            cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
            cache_grad_sampling_loc[xid1] += cache_grad_sampling_loc[xid2];
            cache_grad_sampling_loc[xid1 + 1] +=
                cache_grad_sampling_loc[xid2 + 1];
            if (tid + (s << 1) < spre) {
              cache_grad_attn_weight[tid] +=
                  cache_grad_attn_weight[tid + (s << 1)];
              cache_grad_sampling_loc[xid1] +=
                  cache_grad_sampling_loc[xid2 + (s << 1)];
              cache_grad_sampling_loc[xid1 + 1] +=
                  cache_grad_sampling_loc[xid2 + 1 + (s << 1)];
            }
          }
          __syncthreads();
        }

        if (tid == 0) {
          *grad_sampling_loc = cache_grad_sampling_loc[0];
          *(grad_sampling_loc + 1) = cache_grad_sampling_loc[1];
          *grad_attn_weight = cache_grad_attn_weight[0];
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}

// backward branch
template <typename data_t>
void deformable_attn_cuda_backward(
    cudaStream_t stream, const data_t *grad_out, const data_t *data_value,
    const int64_t *data_spatial_shapes, const int64_t *data_level_start_index,
    const data_t *data_sampling_loc, const data_t *data_attn_weight,
    const int batch_size, const int value_length, const int num_heads,
    const int channels, const int num_levels, const int query_length,
    const int num_points, data_t *grad_value, data_t *grad_sampling_loc,
    data_t *grad_attn_weight) {
  const int num_threads =
      (channels > CUDA_NUM_THREADS) ? CUDA_NUM_THREADS : channels;
  const int num_kernels = batch_size * query_length * num_heads * channels;
  const int num_actual_kernels =
      batch_size * query_length * num_heads * channels;
  if (channels > 1024) {
    if ((channels & 1023) == 0) {
      deformable_attn_cuda_kernel_backward_shm_reduce_v2_multi_blocks<data_t>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
             num_threads * 3 * sizeof(data_t), stream>>>(
              num_kernels, grad_out, data_value, data_spatial_shapes,
              data_level_start_index, data_sampling_loc, data_attn_weight,
              batch_size, value_length, num_heads, channels, num_levels,
              query_length, num_points, grad_value, grad_sampling_loc,
              grad_attn_weight);
    } else {
      deformable_attn_cuda_kernel_backward_gm<data_t>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
             stream>>>(num_kernels, grad_out, data_value, data_spatial_shapes,
                       data_level_start_index, data_sampling_loc,
                       data_attn_weight, batch_size, value_length, num_heads,
                       channels, num_levels, query_length, num_points,
                       grad_value, grad_sampling_loc, grad_attn_weight);
    }
  } else {
    switch (channels) {
    case 1:
      deformable_attn_cuda_kernel_backward_shm_blocksize_aware_reduce_v1<data_t,
                                                                         1>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
             stream>>>(num_kernels, grad_out, data_value, data_spatial_shapes,
                       data_level_start_index, data_sampling_loc,
                       data_attn_weight, batch_size, value_length, num_heads,
                       channels, num_levels, query_length, num_points,
                       grad_value, grad_sampling_loc, grad_attn_weight);
      break;
    case 2:
      deformable_attn_cuda_kernel_backward_shm_blocksize_aware_reduce_v1<data_t,
                                                                         2>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
             stream>>>(num_kernels, grad_out, data_value, data_spatial_shapes,
                       data_level_start_index, data_sampling_loc,
                       data_attn_weight, batch_size, value_length, num_heads,
                       channels, num_levels, query_length, num_points,
                       grad_value, grad_sampling_loc, grad_attn_weight);
      break;
    case 4:
      deformable_attn_cuda_kernel_backward_shm_blocksize_aware_reduce_v1<data_t,
                                                                         4>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
             stream>>>(num_kernels, grad_out, data_value, data_spatial_shapes,
                       data_level_start_index, data_sampling_loc,
                       data_attn_weight, batch_size, value_length, num_heads,
                       channels, num_levels, query_length, num_points,
                       grad_value, grad_sampling_loc, grad_attn_weight);
      break;
    case 8:
      deformable_attn_cuda_kernel_backward_shm_blocksize_aware_reduce_v1<data_t,
                                                                         8>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
             stream>>>(num_kernels, grad_out, data_value, data_spatial_shapes,
                       data_level_start_index, data_sampling_loc,
                       data_attn_weight, batch_size, value_length, num_heads,
                       channels, num_levels, query_length, num_points,
                       grad_value, grad_sampling_loc, grad_attn_weight);
      break;
    case 16:
      deformable_attn_cuda_kernel_backward_shm_blocksize_aware_reduce_v1<data_t,
                                                                         16>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
             stream>>>(num_kernels, grad_out, data_value, data_spatial_shapes,
                       data_level_start_index, data_sampling_loc,
                       data_attn_weight, batch_size, value_length, num_heads,
                       channels, num_levels, query_length, num_points,
                       grad_value, grad_sampling_loc, grad_attn_weight);
      break;
    case 32:
      deformable_attn_cuda_kernel_backward_shm_blocksize_aware_reduce_v1<data_t,
                                                                         32>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
             stream>>>(num_kernels, grad_out, data_value, data_spatial_shapes,
                       data_level_start_index, data_sampling_loc,
                       data_attn_weight, batch_size, value_length, num_heads,
                       channels, num_levels, query_length, num_points,
                       grad_value, grad_sampling_loc, grad_attn_weight);
      break;
    case 64:
      deformable_attn_cuda_kernel_backward_shm_blocksize_aware_reduce_v2<data_t,
                                                                         64>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
             stream>>>(num_kernels, grad_out, data_value, data_spatial_shapes,
                       data_level_start_index, data_sampling_loc,
                       data_attn_weight, batch_size, value_length, num_heads,
                       channels, num_levels, query_length, num_points,
                       grad_value, grad_sampling_loc, grad_attn_weight);
      break;
    case 128:
      deformable_attn_cuda_kernel_backward_shm_blocksize_aware_reduce_v2<data_t,
                                                                         128>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
             stream>>>(num_kernels, grad_out, data_value, data_spatial_shapes,
                       data_level_start_index, data_sampling_loc,
                       data_attn_weight, batch_size, value_length, num_heads,
                       channels, num_levels, query_length, num_points,
                       grad_value, grad_sampling_loc, grad_attn_weight);
      break;
    case 256:
      deformable_attn_cuda_kernel_backward_shm_blocksize_aware_reduce_v2<data_t,
                                                                         256>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
             stream>>>(num_kernels, grad_out, data_value, data_spatial_shapes,
                       data_level_start_index, data_sampling_loc,
                       data_attn_weight, batch_size, value_length, num_heads,
                       channels, num_levels, query_length, num_points,
                       grad_value, grad_sampling_loc, grad_attn_weight);
      break;
    case 512:
      deformable_attn_cuda_kernel_backward_shm_blocksize_aware_reduce_v2<data_t,
                                                                         512>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
             stream>>>(num_kernels, grad_out, data_value, data_spatial_shapes,
                       data_level_start_index, data_sampling_loc,
                       data_attn_weight, batch_size, value_length, num_heads,
                       channels, num_levels, query_length, num_points,
                       grad_value, grad_sampling_loc, grad_attn_weight);
      break;
    case 1024:
      deformable_attn_cuda_kernel_backward_shm_blocksize_aware_reduce_v2<data_t,
                                                                         1024>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0,
             stream>>>(num_kernels, grad_out, data_value, data_spatial_shapes,
                       data_level_start_index, data_sampling_loc,
                       data_attn_weight, batch_size, value_length, num_heads,
                       channels, num_levels, query_length, num_points,
                       grad_value, grad_sampling_loc, grad_attn_weight);
      break;
    default:
      if (channels < 64) {
        deformable_attn_cuda_kernel_backward_shm_reduce_v1<data_t>
            <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
               num_threads * 3 * sizeof(data_t), stream>>>(
                num_kernels, grad_out, data_value, data_spatial_shapes,
                data_level_start_index, data_sampling_loc, data_attn_weight,
                batch_size, value_length, num_heads, channels, num_levels,
                query_length, num_points, grad_value, grad_sampling_loc,
                grad_attn_weight);
      } else {
        deformable_attn_cuda_kernel_backward_shm_reduce_v2<data_t>
            <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
               num_threads * 3 * sizeof(data_t), stream>>>(
                num_kernels, grad_out, data_value, data_spatial_shapes,
                data_level_start_index, data_sampling_loc, data_attn_weight,
                batch_size, value_length, num_heads, channels, num_levels,
                query_length, num_points, grad_value, grad_sampling_loc,
                grad_attn_weight);
      }
    }
  }
}

// backward
std::vector<paddle::Tensor> MSDeformableAttnCUDABackward(
    const paddle::Tensor &value, const paddle::Tensor &value_spatial_shapes,
    const paddle::Tensor &value_level_start_index,
    const paddle::Tensor &sampling_locations,
    const paddle::Tensor &attention_weights, const paddle::Tensor &grad_out) {

  CHECK_INPUT_GPU(value);
  CHECK_INPUT_GPU(value_spatial_shapes);
  CHECK_INPUT_GPU(value_level_start_index);
  CHECK_INPUT_GPU(sampling_locations);
  CHECK_INPUT_GPU(attention_weights);
  CHECK_INPUT_GPU(grad_out);

  const int batch_size = value.shape()[0];
  const int value_length = value.shape()[1];
  const int num_heads = value.shape()[2];
  const int channels = value.shape()[3];

  const int num_levels = value_spatial_shapes.shape()[0];
  const int query_length = sampling_locations.shape()[1];
  const int num_points = sampling_locations.shape()[4];

  auto grad_value =
      paddle::full(value.shape(), 0, value.dtype(), paddle::GPUPlace());
  auto grad_spatial_shapes =
      paddle::full(value.shape(), 0, value.dtype(), paddle::GPUPlace());
  auto grad_level_start_index =
      paddle::full(value.shape(), 0, value.dtype(), paddle::GPUPlace());
  auto grad_sampling_locations =
      paddle::full(sampling_locations.shape(), 0, sampling_locations.dtype(),
                   paddle::GPUPlace());
  auto grad_attention_weights =
      paddle::full(attention_weights.shape(), 0, attention_weights.dtype(),
                   paddle::GPUPlace());

  deformable_attn_cuda_backward<float>(
      value.stream(), grad_out.data<float>(), value.data<float>(),
      value_spatial_shapes.data<int64_t>(),
      value_level_start_index.data<int64_t>(), sampling_locations.data<float>(),
      attention_weights.data<float>(), batch_size, value_length, num_heads,
      channels, num_levels, query_length, num_points, grad_value.data<float>(),
      grad_sampling_locations.data<float>(),
      grad_attention_weights.data<float>());

  return {grad_value, grad_spatial_shapes, grad_level_start_index,
          grad_sampling_locations, grad_attention_weights};
}
