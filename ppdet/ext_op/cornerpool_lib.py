import os
import paddle.fluid as fluid

file_dir = os.path.dirname(os.path.abspath(__file__))
fluid.load_op_library(os.path.join(file_dir, 'src/cornerpool_lib.so'))

from paddle.fluid.layer_helper import LayerHelper

__all__ = [
    'bottom_pool',
    'top_pool',
    'right_pool',
    'left_pool',
]


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
        helper = LayerHelper('bottom_pool', **locals())
        dtype = helper.input_dtype()
        output = helper.create_variable_for_type_inference(dtype)
        max_map = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type="bottom_pool",
            inputs={"X": input},
            outputs={"Output": output,
                     "MaxMap": max_map})
        return output
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
        helper = LayerHelper('top_pool', **locals())
        dtype = helper.input_dtype()
        output = helper.create_variable_for_type_inference(dtype)
        max_map = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type="top_pool",
            inputs={"X": input},
            outputs={"Output": output,
                     "MaxMap": max_map})
        return output

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
        helper = LayerHelper('right_pool', **locals())
        dtype = helper.input_dtype()
        output = helper.create_variable_for_type_inference(dtype)
        max_map = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type="right_pool",
            inputs={"X": input},
            outputs={"Output": output,
                     "MaxMap": max_map})
        return output

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
        helper = LayerHelper('left_pool', **locals())
        dtype = helper.input_dtype()
        output = helper.create_variable_for_type_inference(dtype)
        max_map = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type="left_pool",
            inputs={"X": input},
            outputs={"Output": output,
                     "MaxMap": max_map})
        return output

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
