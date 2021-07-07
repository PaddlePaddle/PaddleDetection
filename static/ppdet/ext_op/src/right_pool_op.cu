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
    auto *max_map = ctx.Output<Tensor>("MaxMap");
    auto *output = ctx.Output<Tensor>("Output");
    auto *x_data = x->data<T>();
    auto x_dims = x->dims();
    int NC_num = x_dims[0] * x_dims[1];
    int height = x_dims[2];
    int width = x_dims[3];
    int num = x->numel();
    auto& dev_ctx = ctx.cuda_device_context();

    int *max_map_data = max_map->mutable_data<int>(x_dims, dev_ctx.GetPlace());
    T *output_data = output->mutable_data<T>(x_dims, dev_ctx.GetPlace());
    auto gpu_place = boost::get<platform::CUDAPlace>(dev_ctx.GetPlace());
    
    int threads = kNumCUDAThreads;
    int blocks = NumBlocks(num / width);
  
    auto max_val_ptr = memory::Alloc(gpu_place, num / width * sizeof(T));
    T* max_val_data = reinterpret_cast<T*>(max_val_ptr->ptr());
    auto max_ind_ptr = memory::Alloc(gpu_place, num / width * sizeof(int));
    int* max_ind_data = reinterpret_cast<int*>(max_ind_ptr->ptr());

    GetMaxInfo<T><<<blocks, threads, 0, dev_ctx.stream()>>>(x->data<T>(), NC_num, height, width, 3, false, max_val_data, max_ind_data, max_map_data);

    blocks = NumBlocks(num);
    ScatterAddFw<T><<<blocks, threads, 0, dev_ctx.stream()>>>(x->data<T>(), max_map_data, NC_num, height, width, 3, output_data);

  }
};

template <typename T>
class RightPoolGradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* max_map = ctx.Input<Tensor>("MaxMap");
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
    int blocks = NumBlocks(grad_num);
    FillConstant<T><<<blocks, threads, 0, dev_ctx.stream()>>>(in_grad_data, 0, grad_num);

    ScatterAddBw<T><<<blocks, threads, 0, dev_ctx.stream()>>>(out_grad->data<T>(), max_map->data<int>(), NC_num, height, width, 3, in_grad_data);
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
