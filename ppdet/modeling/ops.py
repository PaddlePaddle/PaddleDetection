#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle

from paddle.fluid.framework import Variable, in_dygraph_mode
from paddle.fluid import core
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph import layers
from paddle.fluid.data_feeder import check_variable_and_dtype, check_type, check_dtype, convert_dtype
import math
import six
import numpy as np
from functools import reduce

__all__ = [
    'roi_pool',
    'roi_align',
    #'prior_box',
    #'anchor_generator',
    #'generate_proposals',
    #'iou_similarity',
    #'box_coder',
    #'yolo_box',
    #'multiclass_nms',
    'distribute_fpn_proposals',
    'collect_fpn_proposals',
    #'matrix_nms',
]


def roi_pool(input,
             rois,
             output_size,
             spatial_scale=1.0,
             rois_num=None,
             name=None):
    """

    This operator implements the roi_pooling layer.
    Region of interest pooling (also known as RoI pooling) is to perform max pooling on inputs of nonuniform sizes to obtain fixed-size feature maps (e.g. 7*7).

    The operator has three steps:

        1. Dividing each region proposal into equal-sized sections with output_size(h, w);
        2. Finding the largest value in each section;
        3. Copying these max values to the output buffer.

    For more information, please refer to https://stackoverflow.com/questions/43430056/what-is-roi-layer-in-fast-rcnn

    Args:
        input (Tensor): Input feature, 4D-Tensor with the shape of [N,C,H,W], 
            where N is the batch size, C is the input channel, H is Height, W is weight. 
            The data type is float32 or float64.
        rois (Tensor): ROIs (Regions of Interest) to pool over. 
            2D-Tensor or 2D-LoDTensor with the shape of [num_rois,4], the lod level is 1. 
            Given as [[x1, y1, x2, y2], ...], (x1, y1) is the top left coordinates, 
            and (x2, y2) is the bottom right coordinates.
        output_size (int or tuple[int, int]): The pooled output size(h, w), data type is int32. If int, h and w are both equal to output_size.
        spatial_scale (float, optional): Multiplicative spatial scale factor to translate ROI coords from their input scale to the scale used when pooling. Default: 1.0
        rois_num (Tensor): The number of RoIs in each image. Default: None
        name(str, optional): For detailed information, please refer
            to :ref:`api_guide_Name`. Usually name is no need to set and
            None by default.


    Returns:
        Tensor: The pooled feature, 4D-Tensor with the shape of [num_rois, C, output_size[0], output_size[1]].


    Examples:

    ..  code-block:: python

        import paddle
        paddle.enable_static()

        x = paddle.static.data(
                name='data', shape=[None, 256, 32, 32], dtype='float32')
        rois = paddle.static.data(
                name='rois', shape=[None, 4], dtype='float32')
        rois_num = paddle.static.data(name='rois_num', shape=[None], dtype='int32')

        pool_out = ops.roi_pool(
                input=x,
                rois=rois,
                output_size=(1, 1),
                spatial_scale=1.0,
                rois_num=rois_num)
    """
    check_type(output_size, 'output_size', (int, tuple), 'roi_pool')
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    pooled_height, pooled_width = output_size
    if in_dygraph_mode():
        assert rois_num is not None, "rois_num should not be None in dygraph mode."
        pool_out, argmaxes = core.ops.roi_pool(
            input, rois, rois_num, "pooled_height", pooled_height,
            "pooled_width", pooled_width, "spatial_scale", spatial_scale)
        return pool_out, argmaxes

    check_variable_and_dtype(input, 'input', ['float32'], 'roi_pool')
    check_variable_and_dtype(rois, 'rois', ['float32'], 'roi_pool')
    helper = LayerHelper('roi_pool', **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_variable_for_type_inference(dtype)
    argmaxes = helper.create_variable_for_type_inference(dtype='int32')

    inputs = {
        "X": input,
        "ROIs": rois,
    }
    if rois_num is not None:
        inputs['RoisNum'] = rois_num
    helper.append_op(
        type="roi_pool",
        inputs=inputs,
        outputs={"Out": pool_out,
                 "Argmax": argmaxes},
        attrs={
            "pooled_height": pooled_height,
            "pooled_width": pooled_width,
            "spatial_scale": spatial_scale
        })
    return pool_out, argmaxes


def roi_align(input,
              rois,
              output_size,
              spatial_scale=1.0,
              sampling_ratio=-1,
              rois_num=None,
              name=None):
    """

    Region of interest align (also known as RoI align) is to perform
    bilinear interpolation on inputs of nonuniform sizes to obtain 
    fixed-size feature maps (e.g. 7*7)

    Dividing each region proposal into equal-sized sections with
    the pooled_width and pooled_height. Location remains the origin
    result.

    In each ROI bin, the value of the four regularly sampled locations 
    are computed directly through bilinear interpolation. The output is
    the mean of four locations.
    Thus avoid the misaligned problem. 

    Args:
        input (Tensor): Input feature, 4D-Tensor with the shape of [N,C,H,W], 
            where N is the batch size, C is the input channel, H is Height, W is weight. 
            The data type is float32 or float64.
        rois (Tensor): ROIs (Regions of Interest) to pool over.It should be
            a 2-D Tensor or 2-D LoDTensor of shape (num_rois, 4), the lod level is 1. 
            The data type is float32 or float64. Given as [[x1, y1, x2, y2], ...],
            (x1, y1) is the top left coordinates, and (x2, y2) is the bottom right coordinates.
        output_size (int or tuple[int, int]): The pooled output size(h, w), data type is int32. If int, h and w are both equal to output_size.
        spatial_scale (float32, optional): ${spatial_scale_comment} Default: 1.0
        sampling_ratio(int32, optional): ${sampling_ratio_comment} Default: -1
        rois_num (Tensor): The number of RoIs in each image. Default: None
        name(str, optional): For detailed information, please refer
            to :ref:`api_guide_Name`. Usually name is no need to set and
            None by default.

    Returns:
        Tensor:

        Output: The output of ROIAlignOp is a 4-D tensor with shape (num_rois, channels, pooled_h, pooled_w). The data type is float32 or float64.


    Examples:
        .. code-block:: python

            import paddle
            paddle.enable_static()

            x = paddle.static.data(
                name='data', shape=[None, 256, 32, 32], dtype='float32')
            rois = paddle.static.data(
                name='rois', shape=[None, 4], dtype='float32')
            rois_num = paddle.static.data(name='rois_num', shape=[None], dtype='int32')
            align_out = ops.roi_align(input=x,
                                               rois=rois,
                                               ouput_size=(7, 7),
                                               spatial_scale=0.5,
                                               sampling_ratio=-1,
                                               rois_num=rois_num)
    """
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    pooled_height, pooled_width = output_size

    if in_dygraph_mode():
        assert rois_num is not None, "rois_num should not be None in dygraph mode."
        align_out = core.ops.roi_align(
            input, rois, rois_num, "pooled_height", pooled_height,
            "pooled_width", pooled_width, "spatial_scale", spatial_scale,
            "sampling_ratio", sampling_ratio)
        return align_out

    check_variable_and_dtype(input, 'input', ['float32', 'float64'],
                             'roi_align')
    check_variable_and_dtype(rois, 'rois', ['float32', 'float64'], 'roi_align')
    helper = LayerHelper('roi_align', **locals())
    dtype = helper.input_dtype()
    align_out = helper.create_variable_for_type_inference(dtype)
    inputs = {
        "X": input,
        "ROIs": rois,
    }
    if rois_num is not None:
        inputs['RoisNum'] = rois_num
    helper.append_op(
        type="roi_align",
        inputs=inputs,
        outputs={"Out": align_out},
        attrs={
            "pooled_height": pooled_height,
            "pooled_width": pooled_width,
            "spatial_scale": spatial_scale,
            "sampling_ratio": sampling_ratio
        })
    return align_out


def collect_fpn_proposals(multi_rois,
                          multi_scores,
                          min_level,
                          max_level,
                          post_nms_top_n,
                          rois_num_per_level=None,
                          name=None):
    """
    
    **This OP only supports LoDTensor as input**. Concat multi-level RoIs 
    (Region of Interest) and select N RoIs with respect to multi_scores. 
    This operation performs the following steps:

    1. Choose num_level RoIs and scores as input: num_level = max_level - min_level
    2. Concat multi-level RoIs and scores
    3. Sort scores and select post_nms_top_n scores
    4. Gather RoIs by selected indices from scores
    5. Re-sort RoIs by corresponding batch_id

    Args:
        multi_rois(list): List of RoIs to collect. Element in list is 2-D 
            LoDTensor with shape [N, 4] and data type is float32 or float64, 
            N is the number of RoIs.
        multi_scores(list): List of scores of RoIs to collect. Element in list 
            is 2-D LoDTensor with shape [N, 1] and data type is float32 or
            float64, N is the number of RoIs.
        min_level(int): The lowest level of FPN layer to collect
        max_level(int): The highest level of FPN layer to collect
        post_nms_top_n(int): The number of selected RoIs
        rois_num_per_level(list, optional): The List of RoIs' numbers. 
            Each element is 1-D Tensor which contains the RoIs' number of each 
            image on each level and the shape is [B] and data type is 
            int32, B is the number of images. If it is not None then return 
            a 1-D Tensor contains the output RoIs' number of each image and 
            the shape is [B]. Default: None
        name(str, optional): For detailed information, please refer 
            to :ref:`api_guide_Name`. Usually name is no need to set and 
            None by default.

    Returns:
        Variable:

        fpn_rois(Variable): 2-D LoDTensor with shape [N, 4] and data type is 
        float32 or float64. Selected RoIs. 

        rois_num(Tensor): 1-D Tensor contains the RoIs's number of each 
        image. The shape is [B] and data type is int32. B is the number of 
        images. 

    Examples:
        .. code-block:: python
           
            import paddle.fluid as fluid
            import paddle
            paddle.enable_static()
            multi_rois = []
            multi_scores = []
            for i in range(4):
                multi_rois.append(paddle.static.data(
                    name='roi_'+str(i), shape=[None, 4], dtype='float32', lod_level=1))
            for i in range(4):
                multi_scores.append(paddle.static.data(
                    name='score_'+str(i), shape=[None, 1], dtype='float32', lod_level=1))

            fpn_rois = fluid.layers.collect_fpn_proposals(
                multi_rois=multi_rois, 
                multi_scores=multi_scores,
                min_level=2, 
                max_level=5, 
                post_nms_top_n=2000)
    """
    check_type(multi_rois, 'multi_rois', list, 'collect_fpn_proposals')
    check_type(multi_scores, 'multi_scores', list, 'collect_fpn_proposals')
    num_lvl = max_level - min_level + 1
    input_rois = multi_rois[:num_lvl]
    input_scores = multi_scores[:num_lvl]

    if in_dygraph_mode():
        assert rois_num_per_level is not None, "rois_num_per_level should not be None in dygraph mode."
        attrs = ('post_nms_topN', post_nms_top_n)
        output_rois, rois_num = core.ops.collect_fpn_proposals(
            input_rois, input_scores, rois_num_per_level, *attrs)

    helper = LayerHelper('collect_fpn_proposals', **locals())
    dtype = helper.input_dtype('multi_rois')
    check_dtype(dtype, 'multi_rois', ['float32', 'float64'],
                'collect_fpn_proposals')
    output_rois = helper.create_variable_for_type_inference(dtype)
    output_rois.stop_gradient = True

    inputs = {
        'MultiLevelRois': input_rois,
        'MultiLevelScores': input_scores,
    }
    outputs = {'FpnRois': output_rois}
    if rois_num_per_level is not None:
        inputs['MultiLevelRoIsNum'] = rois_num_per_level
        rois_num = helper.create_variable_for_type_inference(dtype='int32')
        rois_num.stop_gradient = True
        outputs['RoisNum'] = rois_num
    helper.append_op(
        type='collect_fpn_proposals',
        inputs=inputs,
        outputs=outputs,
        attrs={'post_nms_topN': post_nms_top_n})
    if rois_num_per_level is not None:
        return output_rois, rois_num
    return output_rois


def distribute_fpn_proposals(fpn_rois,
                             min_level,
                             max_level,
                             refer_level,
                             refer_scale,
                             rois_num=None,
                             name=None):
    """
    
    **This op only takes LoDTensor as input.** In Feature Pyramid Networks 
    (FPN) models, it is needed to distribute all proposals into different FPN 
    level, with respect to scale of the proposals, the referring scale and the 
    referring level. Besides, to restore the order of proposals, we return an 
    array which indicates the original index of rois in current proposals. 
    To compute FPN level for each roi, the formula is given as follows:
    
    .. math::

        roi\_scale &= \sqrt{BBoxArea(fpn\_roi)}

        level = floor(&\log(\\frac{roi\_scale}{refer\_scale}) + refer\_level)

    where BBoxArea is a function to compute the area of each roi.

    Args:

        fpn_rois(Variable): 2-D Tensor with shape [N, 4] and data type is 
            float32 or float64. The input fpn_rois.
        min_level(int32): The lowest level of FPN layer where the proposals come 
            from.
        max_level(int32): The highest level of FPN layer where the proposals
            come from.
        refer_level(int32): The referring level of FPN layer with specified scale.
        refer_scale(int32): The referring scale of FPN layer with specified level.
        rois_num(Tensor): 1-D Tensor contains the number of RoIs in each image. 
            The shape is [B] and data type is int32. B is the number of images.
            If it is not None then return a list of 1-D Tensor. Each element 
            is the output RoIs' number of each image on the corresponding level
            and the shape is [B]. None by default.
        name(str, optional): For detailed information, please refer 
            to :ref:`api_guide_Name`. Usually name is no need to set and 
            None by default. 

    Returns:
        Tuple:

        multi_rois(List) : A list of 2-D LoDTensor with shape [M, 4] 
        and data type of float32 and float64. The length is 
        max_level-min_level+1. The proposals in each FPN level.

        restore_ind(Variable): A 2-D Tensor with shape [N, 1], N is 
        the number of total rois. The data type is int32. It is
        used to restore the order of fpn_rois.

        rois_num_per_level(List): A list of 1-D Tensor and each Tensor is 
        the RoIs' number in each image on the corresponding level. The shape 
        is [B] and data type of int32. B is the number of images


    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import paddle
            paddle.enable_static()
            fpn_rois = paddle.static.data(
                name='data', shape=[None, 4], dtype='float32', lod_level=1)
            multi_rois, restore_ind = fluid.layers.distribute_fpn_proposals(
                fpn_rois=fpn_rois,
                min_level=2,
                max_level=5,
                refer_level=4,
                refer_scale=224)
    """
    num_lvl = max_level - min_level + 1

    if in_dygraph_mode():
        assert rois_num is not None, "rois_num should not be None in dygraph mode."
        attrs = ('min_level', min_level, 'max_level', max_level, 'refer_level',
                 refer_level, 'refer_scale', refer_scale)
        multi_rois, restore_ind, rois_num_per_level = core.ops.distribute_fpn_proposals(
            fpn_rois, rois_num, num_lvl, num_lvl, *attrs)
        return multi_rois, restore_ind, rois_num_per_level

    check_variable_and_dtype(fpn_rois, 'fpn_rois', ['float32', 'float64'],
                             'distribute_fpn_proposals')
    helper = LayerHelper('distribute_fpn_proposals', **locals())
    dtype = helper.input_dtype('fpn_rois')
    multi_rois = [
        helper.create_variable_for_type_inference(dtype) for i in range(num_lvl)
    ]

    restore_ind = helper.create_variable_for_type_inference(dtype='int32')

    inputs = {'FpnRois': fpn_rois}
    outputs = {
        'MultiFpnRois': multi_rois,
        'RestoreIndex': restore_ind,
    }

    if rois_num is not None:
        inputs['RoisNum'] = rois_num
        rois_num_per_level = [
            helper.create_variable_for_type_inference(dtype='int32')
            for i in range(num_lvl)
        ]
        outputs['MultiLevelRoIsNum'] = rois_num_per_level

    helper.append_op(
        type='distribute_fpn_proposals',
        inputs=inputs,
        outputs=outputs,
        attrs={
            'min_level': min_level,
            'max_level': max_level,
            'refer_level': refer_level,
            'refer_scale': refer_scale
        })
    if rois_num is not None:
        return multi_rois, restore_ind, rois_num_per_level
    return multi_rois, restore_ind
