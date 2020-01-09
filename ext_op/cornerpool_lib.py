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
    """
    helper = LayerHelper('bottom_pool', **locals())
    dtype = helper.input_dtype()
    output = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="bottom_pool", inputs={"X": input}, outputs={"Output": output})
    return output


def top_pool(input, name=None):
    """
    """
    helper = LayerHelper('top_pool', **locals())
    dtype = helper.input_dtype()
    output = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="top_pool", inputs={"X": input}, outputs={"Output": output})
    return output


def right_pool(input, name=None):
    """
    """
    helper = LayerHelper('right_pool', **locals())
    dtype = helper.input_dtype()
    output = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="right_pool", inputs={"X": input}, outputs={"Output": output})
    return output


def left_pool(input, name=None):
    """
    """
    helper = LayerHelper('left_pool', **locals())
    dtype = helper.input_dtype()
    output = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="left_pool", inputs={"X": input}, outputs={"Output": output})
    return output
