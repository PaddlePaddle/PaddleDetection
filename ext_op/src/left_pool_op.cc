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
namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class LeftPoolOp : public framework::OperatorWithKernel {
public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null");
    ctx->ShareDim("X", /*->*/ "Output");
  }

protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type =  OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

class LeftPoolOpMaker : public framework::OpProtoAndCheckerMaker {
public:
  void Make() override {
    AddInput("X",
             "Input with shape (batch, C, H, W)");
    AddOutput("Output", "output with same shape as input(X)");
    AddComment(
        R"Doc(
        
        )Doc");
  }
};

class LeftPoolOpGrad : public framework::OperatorWithKernel {
public:
  using framework::OperatorWithKernel::OperatorWithKernel;

protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Output")),
                   "Input(Output@GRAD) should not be null");
    auto out_grad_name = framework::GradVarName("Output");
    ctx->ShareDim(out_grad_name, framework::GradVarName("X"));
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        ctx.Input<Tensor>(framework::GradVarName("Output"))->type(),
        ctx.GetPlace());
  }
};

class LeftPoolGradDescMaker : public framework::SingleGradOpDescMaker {
public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType("left_pool_grad");
    op->SetInput("X", Input("X"));
    op->SetInput(framework::GradVarName("Output"), OutputGrad("Output"));
    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    op->SetAttrMap(Attrs());
    return op;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(left_pool,
                  ops::LeftPoolOp,
                  ops::LeftPoolOpMaker,
                  ops::LeftPoolGradDescMaker);
REGISTER_OPERATOR(left_pool_grad, ops::LeftPoolOpGrad);
