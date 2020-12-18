import os
import paddle.fluid as fluid

use_cpp = False

file_dir = os.path.dirname(os.path.abspath(__file__))
try:
    fluid.load_op_library(os.path.join(file_dir, 'src/cornerpool_lib.so'))
    use_cpp = True
except:
    print(
        'Warning: cornerpool_lib.so not found, use python version instead which may drop the inference speed. Compile in ppdet/ext_op at first if you need cpp version.'
    )

from paddle.fluid.layer_helper import LayerHelper

__all__ = [
    'bottom_pool',
    'top_pool',
    'right_pool',
    'left_pool',
]


def cornerpool_op(layer_type, input, name):
    helper = LayerHelper(layer_type, input=input, name=name)
    dtype = helper.input_dtype()
    output = helper.create_variable_for_type_inference(dtype)
    max_map = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type=layer_type,
        inputs={"X": input},
        outputs={"Output": output,
                 "MaxMap": max_map})
    return output


def bottom_pool(input, is_test=False, name=None):
    """
    This layer calculates the bottom pooling output based on the input.
    Scan the input from top to bottm for the vertical max-pooling.
    The output has the same shape with input.
    Args:
        input(Variable): This input is a Tensor with shape [N, C, H, W].
            The data type is float32 or float64.
    Returns:
        Variable(Tensor): The output of bottom_pool, with shape [N, C, H, W].
        The data type is float32 or float64.
    Examples:
        ..code-block:: python
            import paddle.fluid as fluid
            import cornerpool_lib
            input = fluid.data(
                name='input', shape=[2, 64, 10, 10], dtype='float32')
            output = corner_pool.bottom_pool(input)
    """
    if is_test:
        if use_cpp:
            output = cornerpool_op("bottom_pool", input, name)
            return output

        def cond(i, output):
            return i < H

        def body(i, output):
            cur = fluid.layers.slice(output, [2], [i], [H])
            next = fluid.layers.slice(output, [2], [0], [H - i])
            max_v = fluid.layers.elementwise_max(cur, next)
            orig = fluid.layers.slice(output, [2], [0], [i])
            output = fluid.layers.concat([orig, max_v], axis=2)
            i = i * 2
            return [i, output]

        H = fluid.layers.shape(input)[2]
        i = fluid.layers.fill_constant(shape=[1], dtype='int32', value=1)
        output = input
        output = fluid.layers.while_loop(cond, body, [i, output])
        return output[-1]

    H = input.shape[2]
    i = 1
    output = input
    while i < H:
        cur = output[:, :, i:, :]
        next = output[:, :, :H - i, :]
        max_v = fluid.layers.elementwise_max(cur, next)
        output = fluid.layers.concat([output[:, :, :i, :], max_v], axis=2)
        i *= 2

    return output


def top_pool(input, is_test=False, name=None):
    """
    This layer calculates the top pooling output based on the input.
    Scan the input from bottom to top for the vertical max-pooling.
    The output has the same shape with input.
    Args:
        input(Variable): This input is a Tensor with shape [N, C, H, W].
            The data type is float32 or float64.
    Returns:
        Variable(Tensor): The output of top_pool, with shape [N, C, H, W].
        The data type is float32 or float64.
    Examples:
        ..code-block:: python
            import paddle.fluid as fluid
            import cornerpool_lib
            input = fluid.data(
                name='input', shape=[2, 64, 10, 10], dtype='float32')
            output = corner_pool.top_pool(input)
    """
    if is_test:
        if use_cpp:
            output = cornerpool_op("top_pool", input, name)
            return output

        def cond(i, output):
            return i < H

        def body(i, output):
            cur = fluid.layers.slice(output, [2], [0], [H - i])
            next = fluid.layers.slice(output, [2], [i], [H])
            max_v = fluid.layers.elementwise_max(cur, next)
            orig = fluid.layers.slice(output, [2], [H - i], [H])
            output = fluid.layers.concat([max_v, orig], axis=2)
            i = i * 2
            return [i, output]

        H = fluid.layers.shape(input)[2]
        i = fluid.layers.fill_constant(shape=[1], dtype='int32', value=1)
        output = input
        output = fluid.layers.while_loop(cond, body, [i, output])
        return output[-1]

    H = input.shape[2]
    i = 1
    output = input
    while i < H:
        cur = output[:, :, :H - i, :]
        next = output[:, :, i:, :]
        max_v = fluid.layers.elementwise_max(cur, next)
        output = fluid.layers.concat([max_v, output[:, :, H - i:, :]], axis=2)
        i *= 2

    return output


def right_pool(input, is_test=False, name=None):
    """
    This layer calculates the right pooling output based on the input.
    Scan the input from left to right for the horizontal max-pooling.
    The output has the same shape with input.
    Args:
        input(Variable): This input is a Tensor with shape [N, C, H, W].
            The data type is float32 or float64.
    Returns:
        Variable(Tensor): The output of right_pool, with shape [N, C, H, W].
        The data type is float32 or float64.
    Examples:
        ..code-block:: python
            import paddle.fluid as fluid
            import cornerpool_lib
            input = fluid.data(
                name='input', shape=[2, 64, 10, 10], dtype='float32')
            output = corner_pool.right_pool(input)
    """
    if is_test:
        if use_cpp:
            output = cornerpool_op("right_pool", input, name)
            return output

        def cond(i, output):
            return i < W

        def body(i, output):
            cur = fluid.layers.slice(output, [3], [i], [W])
            next = fluid.layers.slice(output, [3], [0], [W - i])
            max_v = fluid.layers.elementwise_max(cur, next)
            orig = fluid.layers.slice(output, [3], [0], [i])
            output = fluid.layers.concat([orig, max_v], axis=-1)
            i = i * 2
            return [i, output]

        W = fluid.layers.shape(input)[3]
        i = fluid.layers.fill_constant(shape=[1], dtype='int32', value=1)
        output = input
        output = fluid.layers.while_loop(cond, body, [i, output])
        return output[-1]

    W = input.shape[3]
    i = 1
    output = input
    while i < W:
        cur = output[:, :, :, i:]
        next = output[:, :, :, :W - i]
        max_v = fluid.layers.elementwise_max(cur, next)
        output = fluid.layers.concat([output[:, :, :, :i], max_v], axis=-1)
        i *= 2

    return output


def left_pool(input, is_test=False, name=None):
    """
    This layer calculates the left pooling output based on the input.
    Scan the input from right to left for the horizontal max-pooling.
    The output has the same shape with input.
    Args:
        input(Variable): This input is a Tensor with shape [N, C, H, W].
            The data type is float32 or float64.
    Returns:
        Variable(Tensor): The output of left_pool, with shape [N, C, H, W].
        The data type is float32 or float64.
    Examples:
        ..code-block:: python
            import paddle.fluid as fluid
            import cornerpool_lib
            input = fluid.data(
                name='input', shape=[2, 64, 10, 10], dtype='float32')
            output = corner_pool.left_pool(input)
    """
    if is_test:
        if use_cpp:
            output = cornerpool_op("left_pool", input, name)
            return output

        def cond(i, output):
            return i < W

        def body(i, output):
            cur = fluid.layers.slice(output, [3], [0], [W - i])
            next = fluid.layers.slice(output, [3], [i], [W])
            max_v = fluid.layers.elementwise_max(cur, next)
            orig = fluid.layers.slice(output, [3], [W - i], [W])
            output = fluid.layers.concat([max_v, orig], axis=-1)
            i = i * 2
            return [i, output]

        W = fluid.layers.shape(input)[3]
        i = fluid.layers.fill_constant(shape=[1], dtype='int32', value=1)
        output = input
        output = fluid.layers.while_loop(cond, body, [i, output])
        return output[-1]

    W = input.shape[3]
    i = 1
    output = input
    while i < W:
        cur = output[:, :, :, :W - i]
        next = output[:, :, :, i:]
        max_v = fluid.layers.elementwise_max(cur, next)
        output = fluid.layers.concat([max_v, output[:, :, :, W - i:]], axis=-1)
        i *= 2

    return output
