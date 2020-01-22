/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include "util.cu.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

static constexpr int kNumCUDAThreads = 512;
static constexpr int kNumMaximumNumBlocks = 4096;

static inline int NumBlocks(const int N) {
  return std::min((N + kNumCUDAThreads - 1) / kNumCUDAThreads,
                  kNumMaximumNumBlocks);
}

template <typename T>
class RightPoolOpCUDAKernel : public framework::OpKernel<T> {
public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "This kernel only runs on GPU device.");
    auto *x = ctx.Input<Tensor>("X");
    auto *output = ctx.Output<Tensor>("Output");
    auto *x_data = x->data<T>();
    auto x_dims = x->dims();
    int NC_num = x_dims[0] * x_dims[1];
    int height = x_dims[2];
    int width = x_dims[3];
    auto& dev_ctx = ctx.cuda_device_context();

    T *output_data = output->mutable_data<T>(x_dims, dev_ctx.GetPlace());
    auto gpu_place = boost::get<platform::CUDAPlace>(dev_ctx.GetPlace());
    
    memory::Copy(gpu_place, output_data, gpu_place, x_data,
                sizeof(T) * x->numel(), dev_ctx.stream());

    int threads = kNumCUDAThreads;
    int blocks = NumBlocks(NC_num * width * height);
    CornerMaxOut<T><<<blocks, threads>>>(NC_num, height, width, 3, false, output_data);
  }
};

template <typename T>
class RightPoolGradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* out_grad = ctx.Input<Tensor>(framework::GradVarName("Output"));
    auto* in_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto x_dims = x->dims();
    
    auto& dev_ctx = ctx.cuda_device_context();
    T* in_grad_data = in_grad->mutable_data<T>(x_dims, dev_ctx.GetPlace());
    auto gpu_place = boost::get<platform::CUDAPlace>(dev_ctx.GetPlace());
    
    int threads = kNumCUDAThreads;
    int NC_num = x_dims[0] * x_dims[1];
    int height = x_dims[2];
    int width = x_dims[3];
    int grad_num = in_grad->numel();
    int grad_block = NumBlocks(grad_num);
    cudaMemset(in_grad_data, 0, grad_num*sizeof(T));

    int num = grad_num / width;
    int blocks = NumBlocks(num);

    // inital the max_value by the first row of input(x) 
    auto max_val_ptr = memory::Alloc(gpu_place, num * sizeof(T));
    T* max_val_data = reinterpret_cast<T*>(max_val_ptr->ptr());
    SliceOnAxis<T><<<blocks, threads>>>(x->data<T>(), NC_num, height, width, 3, 0, 1, max_val_data);

    // inital the max_ind by 0
    auto max_ind_ptr = memory::Alloc(gpu_place, num * sizeof(int));
    int* max_ind_data = reinterpret_cast<int*>(max_ind_ptr->ptr());
    cudaMemset(max_ind_data, 0, num*sizeof(int));

    ScatterAddOnAxis<T><<<blocks, threads>>>(out_grad->data<T>(), 0, max_ind_data, NC_num, height, width, 3, in_grad_data);

    for (int ind = 1; ind < width; ++ind) {
      UpdateMaxInfo<T><<<blocks, threads>>>(x->data<T>(), NC_num, height, width, 3, ind, max_val_data, max_ind_data);
      ScatterAddOnAxis<T><<<blocks, threads>>>(out_grad->data<T>(), ind, max_ind_data, NC_num, height, width, 3, in_grad_data); 
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(right_pool,
                        ops::RightPoolOpCUDAKernel<float>,
                        ops::RightPoolOpCUDAKernel<double>);
REGISTER_OP_CUDA_KERNEL(right_pool_grad,
                        ops::RightPoolGradOpCUDAKernel<float>,
                        ops::RightPoolGradOpCUDAKernel<double>);
