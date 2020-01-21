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


def bottom_pool(input, name=None):
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
    helper = LayerHelper('bottom_pool', **locals())
    dtype = helper.input_dtype()
    output = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="bottom_pool", inputs={"X": input}, outputs={"Output": output})
    return output


def top_pool(input, name=None):
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
    helper = LayerHelper('top_pool', **locals())
    dtype = helper.input_dtype()
    output = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="top_pool", inputs={"X": input}, outputs={"Output": output})
    return output


def right_pool(input, name=None):
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
    helper = LayerHelper('right_pool', **locals())
    dtype = helper.input_dtype()
    output = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="right_pool", inputs={"X": input}, outputs={"Output": output})
    return output


def left_pool(input, name=None):
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
    helper = LayerHelper('left_pool', **locals())
    dtype = helper.input_dtype()
    output = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="left_pool", inputs={"X": input}, outputs={"Output": output})
    return output
