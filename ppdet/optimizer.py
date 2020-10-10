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
from __future__ import division
from __future__ import print_function

import math
import logging

from paddle import fluid

import paddle.fluid.optimizer as optimizer
import paddle.fluid.regularizer as regularizer
from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter
from paddle.fluid.layers.ops import cos
from paddle.fluid.wrapped_decorator import signature_safe_contextmanager

from ppdet.core.workspace import register, serializable

__all__ = ['LearningRate', 'OptimizerBuilder', 'ExponentialMovingAverageV5']

logger = logging.getLogger(__name__)


@serializable
class PiecewiseDecay(object):
    """
    Multi step learning rate decay

    Args:
        gamma (float | list): decay factor
        milestones (list): steps at which to decay learning rate
    """

    def __init__(self, gamma=[0.1, 0.1], milestones=[60000, 80000],
                 values=None):
        super(PiecewiseDecay, self).__init__()
        if type(gamma) is not list:
            self.gamma = []
            for i in range(len(milestones)):
                self.gamma.append(gamma / 10**i)
        else:
            self.gamma = gamma
        self.milestones = milestones
        self.values = values

    def __call__(self, base_lr=None, learning_rate=None):
        if self.values is not None:
            return fluid.layers.piecewise_decay(self.milestones, self.values)
        assert base_lr is not None, "either base LR or values should be provided"
        values = [base_lr]
        for g in self.gamma:
            new_lr = base_lr * g
            values.append(new_lr)
        return fluid.layers.piecewise_decay(self.milestones, values)


@serializable
class PolynomialDecay(object):
    """
    Applies polynomial decay to the initial learning rate.
    Args:
        max_iter (int): The learning rate decay steps. 
        end_lr (float): End learning rate.
        power (float): Polynomial attenuation coefficient
    """

    def __init__(self, max_iter=180000, end_lr=0.0001, power=1.0):
        super(PolynomialDecay).__init__()
        self.max_iter = max_iter
        self.end_lr = end_lr
        self.power = power

    def __call__(self, base_lr=None, learning_rate=None):
        assert base_lr is not None, "either base LR or values should be provided"
        lr = fluid.layers.polynomial_decay(base_lr, self.max_iter, self.end_lr,
                                           self.power)
        return lr


@serializable
class ExponentialDecay(object):
    """
    Applies exponential decay to the learning rate.
    Args:
        max_iter (int): The learning rate decay steps. 
        decay_rate (float): The learning rate decay rate. 
    """

    def __init__(self, max_iter, decay_rate):
        super(ExponentialDecay).__init__()
        self.max_iter = max_iter
        self.decay_rate = decay_rate

    def __call__(self, base_lr=None, learning_rate=None):
        assert base_lr is not None, "either base LR or values should be provided"
        lr = fluid.layers.exponential_decay(base_lr, self.max_iter,
                                            self.decay_rate)
        return lr


@serializable
class CosineDecay(object):
    """
    Cosine learning rate decay

    Args:
        max_iters (float): max iterations for the training process.
            if you commbine cosine decay with warmup, it is recommended that
            the max_iter is much larger than the warmup iter
    """

    def __init__(self, max_iters=180000, bias=0.2, step_each_epoch=1):
        self.max_iters = max_iters
        self.bias = 0.2
        self.step_each_epoch = step_each_epoch

    def __call__(self, base_lr=None, learning_rate=None):
        assert base_lr is not None, "either base LR or values should be provided"
        lr = base_lr * ((1 - self.bias) * fluid.layers.cosine_decay(
            1., self.step_each_epoch, self.max_iters) + self.bias)
        return lr


@serializable
class CosineDecayWithSkip(object):
    """
    Cosine decay, with explicit support for warm up

    Args:
        total_steps (int): total steps over which to apply the decay
        skip_steps (int): skip some steps at the beginning, e.g., warm up
    """

    def __init__(self, total_steps, skip_steps=None):
        super(CosineDecayWithSkip, self).__init__()
        assert (not skip_steps or skip_steps > 0), \
            "skip steps must be greater than zero"
        assert total_steps > 0, "total step must be greater than zero"
        assert (not skip_steps or skip_steps < total_steps), \
            "skip steps must be smaller than total steps"
        self.total_steps = total_steps
        self.skip_steps = skip_steps

    def __call__(self, base_lr=None, learning_rate=None):
        steps = _decay_step_counter()
        total = self.total_steps
        if self.skip_steps is not None:
            total -= self.skip_steps

        lr = fluid.layers.tensor.create_global_var(
            shape=[1],
            value=base_lr,
            dtype='float32',
            persistable=True,
            name="learning_rate")

        def decay():
            cos_lr = base_lr * .5 * (cos(steps * (math.pi / total)) + 1)
            fluid.layers.tensor.assign(input=cos_lr, output=lr)

        if self.skip_steps is None:
            decay()
        else:
            skipped = steps >= self.skip_steps
            fluid.layers.cond(skipped, decay)
        return lr


@serializable
class LinearWarmup(object):
    """
    Warm up learning rate linearly

    Args:
        steps (int): warm up steps
        start_factor (float): initial learning rate factor
    """

    def __init__(self, steps=500, start_factor=1. / 3):
        super(LinearWarmup, self).__init__()
        self.steps = steps
        self.start_factor = start_factor

    def __call__(self, base_lr, learning_rate):
        start_lr = base_lr * self.start_factor

        return fluid.layers.linear_lr_warmup(
            learning_rate=learning_rate,
            warmup_steps=self.steps,
            start_lr=start_lr,
            end_lr=base_lr)


@register
class LearningRate(object):
    """
    Learning Rate configuration

    Args:
        base_lr (float): base learning rate
        schedulers (list): learning rate schedulers
    """
    __category__ = 'optim'

    def __init__(self,
                 base_lr=0.01,
                 schedulers=[PiecewiseDecay(), LinearWarmup()]):
        super(LearningRate, self).__init__()
        self.base_lr = base_lr
        self.schedulers = schedulers

    def __call__(self):
        lr = None
        for sched in self.schedulers:
            lr = sched(self.base_lr, lr)
        return lr


@register
class OptimizerBuilder():
    """
    Build optimizer handles

    Args:
        regularizer (object): an `Regularizer` instance
        optimizer (object): an `Optimizer` instance
    """
    __category__ = 'optim'

    def __init__(self,
                 clip_grad_by_norm=None,
                 regularizer={'type': 'L2',
                              'factor': .0001},
                 optimizer={'type': 'Momentum',
                            'momentum': .9}):
        self.clip_grad_by_norm = clip_grad_by_norm
        self.regularizer = regularizer
        self.optimizer = optimizer

    def __call__(self, learning_rate):
        if self.clip_grad_by_norm is not None:
            fluid.clip.set_gradient_clip(
                clip=fluid.clip.GradientClipByGlobalNorm(
                    clip_norm=self.clip_grad_by_norm))
        if self.regularizer:
            reg_type = self.regularizer['type'] + 'Decay'
            reg_factor = self.regularizer['factor']
            regularization = getattr(regularizer, reg_type)(reg_factor)
        else:
            regularization = None
        optim_args = self.optimizer.copy()
        optim_type = optim_args['type']
        del optim_args['type']
        op = getattr(optimizer, optim_type)
        return op(learning_rate=learning_rate,
                  regularization=regularization,
                  **optim_args)


class ExponentialMovingAverageV5(object):
    """
    Compute the moving average of parameters with exponential decay.
    Given a parameter :math:`\\theta`, its exponential moving average (EMA)
    will be

    ..  math::

        \\text{EMA}_0 & = 0

        \\text{EMA}_t & = \\text{decay} * \\text{EMA}_{t-1} + (1 - \\text{decay}) * \\theta_t

    The average results calculated by **update()** method will be saved in 
    temporary variables which are created and maintained by the object, and can 
    be applied to parameters of current model by calling **apply()** method. And 
    the **restore()** method is used to restore the parameters.

    **Bias correction**. All EMAs are initialized to :math:`0` and hence they will be 
    zero biased, which can be corrected by divided by a factor 
    :math:`(1 - \\text{decay}^t)` , i.e., the actual EMAs applied to parameters 
    when calling **apply()** method would be 

    ..  math::
    
        \\widehat{\\text{EMA}}_t = \\frac{\\text{EMA}_t}{1 - \\text{decay}^t}

    **Decay rate scheduling**. A large decay rate very close to 1 would result 
    in that the averages move very slowly. And a better strategy is to set a 
    relative smaller decay rate in the very beginning. The argument **thres_steps**
    allows users to pass a Variable to schedule the decay rate, in this case, 
    the actual decay rate becomes
     
    ..  math::
    
        \\min(\\text{decay}, \\frac{1 + \\text{thres_steps}}{10 + \\text{thres_steps}})

    Usually **thres_steps** can be the global training steps.


    Args:
	decay (float, optional): The exponential decay rate, usually close to 1, such as 
            0.999, 0.9999, ... . Default 0.999.
        thres_steps (Variable|None): If not `None`, schedule the decay rate. 
            Default None.
        name (str|None): For detailed information, please refer to 
            :ref:`api_guide_Name`. Usually name is no need to set and None by 
            default.


    Examples:

	.. code-block:: python

	    import numpy
	    import paddle
	    import paddle.fluid as fluid

	    data = fluid.data(name='x', shape=[-1, 5], dtype='float32')
	    hidden = fluid.layers.fc(input=data, size=10)
	    cost = fluid.layers.mean(hidden)

	    test_program = fluid.default_main_program().clone(for_test=True)

	    optimizer = fluid.optimizer.Adam(learning_rate=0.001)
	    optimizer.minimize(cost)

	    global_steps = fluid.layers.autoincreased_step_counter()
	    ema = fluid.optimizer.ExponentialMovingAverage(0.999, thres_steps=global_steps)
	    ema.update()

	    place = fluid.CPUPlace()
	    exe = fluid.Executor(place)
	    exe.run(fluid.default_startup_program())

	    for pass_id in range(3):
		for batch_id in range(6):
		    data = numpy.random.random(size=(10, 5)).astype('float32')
		    exe.run(program=fluid.default_main_program(),
			feed={'x': data}, 
			fetch_list=[cost.name])

		# usage 1
		with ema.apply(exe):
		    data = numpy.random.random(size=(10, 5)).astype('float32')
		    exe.run(program=test_program,
			    feed={'x': data}, 
			    fetch_list=[hidden.name])
			    

		 # usage 2
		with ema.apply(exe, need_restore=False):
		    data = numpy.random.random(size=(10, 5)).astype('float32')
		    exe.run(program=test_program,
			    feed={'x': data}, 
			    fetch_list=[hidden.name])
		ema.restore(exe)
    """

    def __init__(self, decay=0.999, name=None):
        if framework.in_dygraph_mode():
            raise Exception(
                "In dygraph, don't support ExponentialMovingAverage.")
        self._decay = decay
        # self._thres_steps = thres_steps
        self._name = name if name is not None else ''
        # self._decay_var = self._get_ema_decay()

        self._step_counter_name = "@EMA_STEP_COUNTER@"
        self._params_tmps = []
        for param in default_main_program().global_block().all_parameters():
            if param.do_model_average != False:
                tmp = param.block.create_var(
                    name=unique_name.generate(".".join(
                        [self._name + param.name, 'ema_tmp'])),
                    dtype=param.dtype,
                    persistable=False,
                    stop_gradient=True)
                self._params_tmps.append((param, tmp))

        self._ema_vars = {}
        for param, tmp in self._params_tmps:
            with param.block.program._optimized_guard(
                [param, tmp]), name_scope('moving_average'):
                self._ema_vars[param.name] = self._create_ema_vars(param)

        self.apply_program = Program()
        block = self.apply_program.global_block()
        with program_guard(main_program=self.apply_program):
            # decay_pow, global_step = self._get_decay_pow(block)
            for param, tmp in self._params_tmps:
                param = block._clone_variable(param)
                tmp = block._clone_variable(tmp)
                ema = block._clone_variable(self._ema_vars[param.name])
                layers.assign(input=param, output=tmp)
                # bias correction
                # with layers.control_flow.Switch() as switch:
                #     with switch.case(global_step > 0):
                #         layers.assign(output=ema, input=ema / (1.0 - decay_pow))
                layers.assign(input=ema, output=param)

        self.restore_program = Program()
        block = self.restore_program.global_block()
        with program_guard(main_program=self.restore_program):
            for param, tmp in self._params_tmps:
                tmp = block._clone_variable(tmp)
                param = block._clone_variable(param)
                layers.assign(input=tmp, output=param)

    # def _get_ema_decay(self):
    #     with default_main_program()._lr_schedule_guard():
    #         decay_var = layers.tensor.create_global_var(
    #             shape=[1],
    #             value=self._decay,
    #             dtype='float32',
    #             persistable=True,
    #             name="scheduled_ema_decay_rate")

    #     return decay_var

    # def _get_decay_pow(self, block):
    #     global_step = layers.create_global_var(
    #         name=self._step_counter_name,
    #         shape=[1],
    #         value=0,
    #         dtype='int64',
    #         persistable=True)
    #     global_step = layers.cast(global_step, "float32")
    #     decay_var = block._clone_variable(self._decay_var)
    #     decay_pow_acc = layers.elementwise_pow(decay_var, global_step)
    #     return decay_pow_acc, global_step

    def _create_ema_vars(self, param):
        param_ema = layers.create_global_var(
            name=unique_name.generate(self._name + param.name + '_ema'),
            shape=param.shape,
            value=0.0,
            dtype=param.dtype,
            persistable=True)

        return param_ema

    def update(self):
        """ 
        Update Exponential Moving Average. Should only call this method in 
        train program.
        """
        global_step = layers.autoincreased_step_counter(
            counter_name=self._step_counter_name)
        global_step_t = fluid.layers.cast(global_step, "float32")
        decay_var = self._decay * (1 - fluid.layers.exp(-global_step_t / 2000.))
        param_master_emas = []
        for param, tmp in self._params_tmps:
            with param.block.program._optimized_guard(
                [param, tmp]), name_scope('moving_average'):
                param_ema = self._ema_vars[param.name]
                if param.name + '.master' in self._ema_vars:
                    master_ema = self._ema_vars[param.name + '.master']
                    param_master_emas.append([param_ema, master_ema])
                else:
                    ema_t = param_ema * decay_var + param * (1 - decay_var)
                    layers.assign(input=ema_t, output=param_ema)

        # for fp16 params
        for param_ema, master_ema in param_master_emas:
            default_main_program().global_block().append_op(
                type="cast",
                inputs={"X": master_ema},
                outputs={"Out": param_ema},
                attrs={
                    "in_dtype": master_ema.dtype,
                    "out_dtype": param_ema.dtype
                })

    @signature_safe_contextmanager
    def apply(self, executor, need_restore=True):
        """
        Apply moving average to parameters for evaluation.
        
        Args:
            executor (Executor): The Executor to execute applying.
            need_restore (bool, optional): Whether to restore parameters after 
                applying. Default True.
        """
        executor.run(self.apply_program)
        try:
            yield
        finally:
            if need_restore:
                self.restore(executor)

    def restore(self, executor):
        """Restore parameters.
        
        Args:
            executor (Executor): The Executor to execute restoring.
        """
        executor.run(self.restore_program)
