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

class BottomPoolOp : public framework::OperatorWithKernel {
public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null");
    ctx->ShareDim("X", /*->*/ "Output");
  }

protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("X")->type(),
                                   ctx.GetPlace());
  }
};

class BottomPoolOpMaker : public framework::OpProtoAndCheckerMaker {
public:
  void Make() override {
    AddInput("X",
             "Input with shape (batch, C, H, W)");
    AddOutput("Output", "output with same shape as input(X)");
    AddComment(
        R"Doc(
This operatio calculates the bottom pooling output based on the input.
Scan the input from top to bottom for the vertical max-pooling.
The output has the same shape with input.
        )Doc");
  }
};

class BottomPoolOpGrad : public framework::OperatorWithKernel {
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

template <typename T>
class BottomPoolGradDescMaker : public framework::SingleGradOpMaker<T> {
public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

protected:
  std::unique_ptr<T> Apply() const override {
    auto* op = new T();
    op->SetType("bottom_pool_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Output"), this->OutputGrad("Output"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
    return std::unique_ptr<T>(op);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(bottom_pool,
                  ops::BottomPoolOp,
                  ops::BottomPoolOpMaker,
                  ops::BottomPoolGradDescMaker<paddle::framework::OpDesc>,
                  ops::BottomPoolGradDescMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(bottom_pool_grad, ops::BottomPoolOpGrad);
