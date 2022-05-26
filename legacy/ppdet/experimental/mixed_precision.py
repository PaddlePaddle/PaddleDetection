# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import print_function

import six
from paddle.fluid.framework import Parameter
from paddle.fluid import layers
from paddle.fluid import core
from paddle.fluid import unique_name
import paddle.fluid.layer_helper_base as lhb
import paddle.fluid.optimizer as optim

__all__ = [
    'mixed_precision_global_state', 'mixed_precision_context',
    'StaticLossScale', 'DynamicLossScale'
]

_mixed_precision_global_state = None


def mixed_precision_global_state():
    return _mixed_precision_global_state


class LossScale(object):
    def __init__(self):
        super(LossScale, self).__init__()

    def get_loss_scale_var(self):
        return self.scale

    def increment(self):
        raise NotImplementedError()

    def decrement(self):
        raise NotImplementedError()


class StaticLossScale(LossScale):
    """
    Static (fixed) loss scale manager.

    Args:
        init_loss_scale (float): initial loss scale value.

    Examples:

        .. code-block:: python

            from paddle import fluid
            from ppdet.experimental import (mixed_precision_context,
                                            StaticLossScale)

            with mixed_precision_context(StaticLossScale(8.), True) as ctx:
                # ...
                # scale loss
                loss_scale = ctx.get_loss_scale_var()

    """

    def __init__(self, init_loss_scale=1.):
        super(StaticLossScale, self).__init__()
        self.scale = layers.create_global_var(
            name=unique_name.generate("loss_scale"),
            shape=[1],
            value=init_loss_scale,
            dtype='float32',
            persistable=True)


class DynamicLossScale(LossScale):
    """
    Dynamic loss scale manager. it works as follows:
    if gradients is valid for `increment_every` steps, loss scale values is
    increased by `factor`, otherwise loss scale values is decreased by `factor`

    Args:
        init_loss_scale (float): initial loss scale value.
        increment_every (int): minimum 'good' steps before loss scale increase.
        factor (float): increase/decrease loss scale by this much.

    Examples:

        .. code-block:: python

            from paddle import fluid
            from ppdet.experimental import (mixed_precision_context,
                                            DynamicLossScale)

            loss_scale = DynamicLossScale(8., 1000, 4.)
            with mixed_precision_context(loss_scale, True) as ctx:
                # ...
                # scale loss
                loss_scale = ctx.get_loss_scale_var()

    """

    def __init__(self, init_loss_scale=2**15, increment_every=2000, factor=2.):
        super(DynamicLossScale, self).__init__()
        self.scale = layers.create_global_var(
            name=unique_name.generate("loss_scale"),
            shape=[1],
            value=init_loss_scale,
            dtype='float32',
            persistable=True)
        self.good_steps = layers.create_global_var(
            name=unique_name.generate("good_steps"),
            shape=[1],
            value=0,
            dtype='int32',
            persistable=True)
        self.increment_every = layers.fill_constant(
            shape=[1], dtype='int32', value=increment_every)
        self.factor = factor

    def increment(self):
        enough_steps = layers.less_than(self.increment_every,
                                        self.good_steps + 1)

        def increment_step():
            layers.increment(self.good_steps)

        def maybe_update():
            new_scale = self.scale * self.factor
            scale_valid = layers.isfinite(new_scale)

            def update_scale_and_step():
                layers.assign(new_scale, self.scale)
                layers.assign(
                    layers.zeros_like(self.good_steps), self.good_steps)

            layers.cond(scale_valid, update_scale_and_step)

        layers.cond(enough_steps, maybe_update, increment_step)

    def decrement(self):
        new_scale = self.scale / self.factor
        one = layers.fill_constant(shape=[1], dtype='float32', value=1.0)
        layers.assign(layers.elementwise_max(new_scale, one), self.scale)
        layers.assign(layers.zeros_like(self.good_steps), self.good_steps)


class mixed_precision_context(object):
    """
    Context manager for mixed precision training.

    Args:
        loss_scale (float, str or obj): loss scale settings, can be:
            1. an number: use fixed loss scale.
            2. 'dynamic': use a default `DynamicLossScale`.
            3. `DynamicLossScale` or `StaticLossScale` instance.
         enabled (bool): enable mixed precision training.

    Examples:

        .. code-block:: python

            from paddle import fluid
            from ppdet.experimental import mixed_precision_context

            with mixed_precision_context('dynamic', True) as ctx:
                # cast inputs to float16
                inputs = fluid.layers.cast(inputs, "float16")
                # build model here
                logits = model(inputs)
                # use float32 for softmax
                logits = fluid.layers.cast(logits, "float32")
                softmax = fluid.layers.softmax(logits)
                loss = fluid.layers.cross_entropy(input=softmax, label=label)
                avg_loss = fluid.layers.mean(loss)
                # scale loss
                loss_scale = ctx.get_loss_scale_var()
                avg_loss *= loss_scale
                optimizer = fluid.optimizer.Momentum(...)
                optimizer.minimize(avg_loss)

    """

    def __init__(self, loss_scale=1., enabled=True):
        super(mixed_precision_context, self).__init__()
        self.enabled = enabled
        if not enabled:
            return
        monkey_patch()
        if isinstance(loss_scale, six.integer_types + (float, )):
            self.loss_scale = StaticLossScale(loss_scale)
        elif loss_scale == 'dynamic':
            self.loss_scale = DynamicLossScale()
        else:
            assert isinstance(loss_scale, LossScale), \
                "Invalid loss scale argument"
            self.loss_scale = loss_scale

    @property
    def dynamic_scaling(self):
        return isinstance(self.loss_scale, DynamicLossScale)

    def __getattr__(self, attr):
        if attr in ['get_loss_scale_var', 'increment', 'decrement']:
            return getattr(self.loss_scale, attr)

    def __enter__(self):
        if not self.enabled:
            return
        global _mixed_precision_global_state
        _mixed_precision_global_state = self
        return mixed_precision_global_state()

    def __exit__(self, *args):
        if not self.enabled:
            return
        global _mixed_precision_global_state
        _mixed_precision_global_state = None
        return mixed_precision_global_state()


def create_parameter(self,
                     attr,
                     shape,
                     dtype,
                     is_bias=False,
                     default_initializer=None):
    mp_state = mixed_precision_global_state()
    is_half = (isinstance(dtype, str) and dtype == 'float16') \
        or (isinstance(dtype, core.VarDesc.VarType)
            and dtype == core.VarDesc.VarType.FP16)

    if is_half and mp_state is not None:
        dtype = 'float32'

    param = self._create_parameter(attr, shape, dtype, is_bias,
                                   default_initializer)
    if not is_half or mp_state is None:
        return param

    param16 = self.main_program.current_block().create_var(
        name=param.name + '.fp16',
        dtype='float16',
        type=param.type,
        persistable=False)
    self.append_op(
        type='cast',
        inputs={'X': [param]},
        outputs={'Out': [param16]},
        attrs={'in_dtype': param.dtype,
               'out_dtype': param16.dtype})
    return param16


def scale_gradient(block, context):
    state = mixed_precision_global_state()
    if state is None:
        return
    scale = state.get_loss_scale_var()
    op_desc = block.desc.op(block.desc.op_size() - 1)
    op_role_attr_name = core.op_proto_and_checker_maker.kOpRoleAttrName()
    bwd_role = core.op_proto_and_checker_maker.OpRole.Backward
    for name in [n for n in op_desc.output_arg_names() if n in context]:
        fwd_var = block._var_recursive(context[name])
        if not isinstance(fwd_var, Parameter):
            continue  # TODO verify all use cases
        scale_op_desc = block.desc.append_op()
        scale_op_desc.set_type("elementwise_div")
        scale_op_desc.set_input("X", [name])
        scale_op_desc.set_input("Y", [scale.name])
        scale_op_desc.set_output("Out", [name])
        scale_op_desc._set_attr("axis", -1)
        scale_op_desc._set_attr(op_role_attr_name, bwd_role)


def update_loss_scale(grads):
    state = mixed_precision_global_state()
    if state is None or not state.dynamic_scaling:
        return
    per_grad_check = layers.stack([layers.reduce_sum(g) for g in grads])
    grad_valid = layers.isfinite(per_grad_check)
    layers.cond(grad_valid, lambda: state.increment(),
                lambda: state.decrement())
    return grad_valid


def backward(self, loss, **kwargs):
    state = mixed_precision_global_state()
    callbacks = 'callbacks' in kwargs and kwargs['callbacks'] or None
    if callbacks is None:
        from paddle.fluid.clip import error_clip_callback
        callbacks = [error_clip_callback]  # XXX what if gradient is zero?
    if state is not None:
        kwargs['callbacks'] = [scale_gradient] + callbacks
    else:
        kwargs['callbacks'] = callbacks
    param_grads = self._backward(loss, **kwargs)

    def zero_grad():
        for _, g in param_grads:
            layers.assign(layers.zeros_like(g), g)

    if state is not None:
        grad_valid = update_loss_scale(v for k, v in param_grads)
        if state.dynamic_scaling:
            layers.cond(grad_valid, None, zero_grad)

    return param_grads


mixed_precision_patched = False


# XXX this is a temporary measure, until thoroughly evaluated
def monkey_patch():
    global mixed_precision_patched
    if mixed_precision_patched:
        return
    create_parameter_orig = lhb.LayerHelperBase.create_parameter
    lhb.LayerHelperBase.create_parameter = create_parameter
    lhb.LayerHelperBase._create_parameter = create_parameter_orig
    backward_orig = optim.Optimizer.backward
    optim.Optimizer.backward = backward
    optim.Optimizer._backward = backward_orig
    mixed_precision_patched = True
