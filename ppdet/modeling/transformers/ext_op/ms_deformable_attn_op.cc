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

#include <vector>

// declare GPU implementation
std::vector<paddle::Tensor>
MSDeformableAttnCUDAForward(const paddle::Tensor &value,
                            const paddle::Tensor &value_spatial_shapes,
                            const paddle::Tensor &value_level_start_index,
                            const paddle::Tensor &sampling_locations,
                            const paddle::Tensor &attention_weights);

std::vector<paddle::Tensor> MSDeformableAttnCUDABackward(
    const paddle::Tensor &value, const paddle::Tensor &value_spatial_shapes,
    const paddle::Tensor &value_level_start_index,
    const paddle::Tensor &sampling_locations,
    const paddle::Tensor &attention_weights, const paddle::Tensor &grad_out);

//// CPU not implemented

std::vector<std::vector<int64_t>>
MSDeformableAttnInferShape(std::vector<int64_t> value_shape,
                           std::vector<int64_t> value_spatial_shapes_shape,
                           std::vector<int64_t> value_level_start_index_shape,
                           std::vector<int64_t> sampling_locations_shape,
                           std::vector<int64_t> attention_weights_shape) {
  return {{value_shape[0], sampling_locations_shape[1],
           value_shape[2] * value_shape[3]}};
}

std::vector<paddle::DataType>
MSDeformableAttnInferDtype(paddle::DataType value_dtype,
                           paddle::DataType value_spatial_shapes_dtype,
                           paddle::DataType value_level_start_index_dtype,
                           paddle::DataType sampling_locations_dtype,
                           paddle::DataType attention_weights_dtype) {
  return {value_dtype};
}

PD_BUILD_OP(ms_deformable_attn)
    .Inputs({"Value", "SpatialShapes", "LevelIndex", "SamplingLocations",
             "AttentionWeights"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(MSDeformableAttnCUDAForward))
    .SetInferShapeFn(PD_INFER_SHAPE(MSDeformableAttnInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MSDeformableAttnInferDtype));

PD_BUILD_GRAD_OP(ms_deformable_attn)
    .Inputs({"Value", "SpatialShapes", "LevelIndex", "SamplingLocations",
             "AttentionWeights", paddle::Grad("Out")})
    .Outputs({paddle::Grad("Value"), paddle::Grad("SpatialShapes"),
              paddle::Grad("LevelIndex"), paddle::Grad("SamplingLocations"),
              paddle::Grad("AttentionWeights")})
    .SetKernelFn(PD_KERNEL(MSDeformableAttnCUDABackward));
