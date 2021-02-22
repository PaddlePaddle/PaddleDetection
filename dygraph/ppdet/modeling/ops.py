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
import paddle.nn.functional as F
import paddle.nn as nn
from paddle import ParamAttr
from paddle.regularizer import L2Decay

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
    'prior_box',
    'generate_proposals',
    'iou_similarity',
    'box_coder',
    'yolo_box',
    'multiclass_nms',
    'distribute_fpn_proposals',
    'collect_fpn_proposals',
    'matrix_nms',
    'batch_norm',
]


def batch_norm(ch, norm_type='bn', norm_decay=0., initializer=None, name=None):
    bn_name = name + '.bn'
    if norm_type == 'sync_bn':
        batch_norm = nn.SyncBatchNorm
    else:
        batch_norm = nn.BatchNorm2D

    return batch_norm(
        ch,
        weight_attr=ParamAttr(
            name=bn_name + '.scale',
            initializer=initializer,
            regularizer=L2Decay(norm_decay)),
        bias_attr=ParamAttr(
            name=bn_name + '.offset', regularizer=L2Decay(norm_decay)))


@paddle.jit.not_to_static
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
        from ppdet.modeling import ops
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

    else:
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


@paddle.jit.not_to_static
def roi_align(input,
              rois,
              output_size,
              spatial_scale=1.0,
              sampling_ratio=-1,
              rois_num=None,
              aligned=True,
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
        spatial_scale (float32, optional): Multiplicative spatial scale factor to translate ROI coords 
            from their input scale to the scale used when pooling. Default: 1.0
        sampling_ratio(int32, optional): number of sampling points in the interpolation grid. 
            If <=0, then grid points are adaptive to roi_width and pooled_w, likewise for height. Default: -1
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
            from ppdet.modeling import ops
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
    check_type(output_size, 'output_size', (int, tuple), 'roi_align')
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    pooled_height, pooled_width = output_size

    if in_dygraph_mode():
        assert rois_num is not None, "rois_num should not be None in dygraph mode."
        align_out = core.ops.roi_align(
            input, rois, rois_num, "pooled_height", pooled_height,
            "pooled_width", pooled_width, "spatial_scale", spatial_scale,
            "sampling_ratio", sampling_ratio, "aligned", aligned)
        return align_out

    else:
        check_variable_and_dtype(input, 'input', ['float32', 'float64'],
                                 'roi_align')
        check_variable_and_dtype(rois, 'rois', ['float32', 'float64'],
                                 'roi_align')
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
                "sampling_ratio": sampling_ratio,
                "aligned": aligned,
            })
        return align_out


@paddle.jit.not_to_static
def iou_similarity(x, y, box_normalized=True, name=None):
    """
    Computes intersection-over-union (IOU) between two box lists.
    Box list 'X' should be a LoDTensor and 'Y' is a common Tensor,
    boxes in 'Y' are shared by all instance of the batched inputs of X.
    Given two boxes A and B, the calculation of IOU is as follows:

    $$
    IOU(A, B) = 
    \\frac{area(A\\cap B)}{area(A)+area(B)-area(A\\cap B)}
    $$

    Args:
        x (Tensor): Box list X is a 2-D Tensor with shape [N, 4] holds N 
             boxes, each box is represented as [xmin, ymin, xmax, ymax], 
             the shape of X is [N, 4]. [xmin, ymin] is the left top 
             coordinate of the box if the input is image feature map, they
             are close to the origin of the coordinate system. 
             [xmax, ymax] is the right bottom coordinate of the box.
             The data type is float32 or float64.
        y (Tensor): Box list Y holds M boxes, each box is represented as 
             [xmin, ymin, xmax, ymax], the shape of X is [N, 4]. 
             [xmin, ymin] is the left top coordinate of the box if the 
             input is image feature map, and [xmax, ymax] is the right 
             bottom coordinate of the box. The data type is float32 or float64.
        box_normalized(bool): Whether treat the priorbox as a normalized box.
            Set true by default.
        name(str, optional): For detailed information, please refer 
            to :ref:`api_guide_Name`. Usually name is no need to set and 
            None by default. 

    Returns:
        Tensor: The output of iou_similarity op, a tensor with shape [N, M] 
              representing pairwise iou scores. The data type is same with x.

    Examples:
        .. code-block:: python

            import paddle
            from ppdet.modeling import ops
            paddle.enable_static()

            x = paddle.static.data(name='x', shape=[None, 4], dtype='float32')
            y = paddle.static.data(name='y', shape=[None, 4], dtype='float32')
            iou = ops.iou_similarity(x=x, y=y)
    """

    if in_dygraph_mode():
        out = core.ops.iou_similarity(x, y, 'box_normalized', box_normalized)
        return out
    else:
        helper = LayerHelper("iou_similarity", **locals())
        out = helper.create_variable_for_type_inference(dtype=x.dtype)

        helper.append_op(
            type="iou_similarity",
            inputs={"X": x,
                    "Y": y},
            attrs={"box_normalized": box_normalized},
            outputs={"Out": out})
        return out


@paddle.jit.not_to_static
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
           
            import paddle
            from ppdet.modeling import ops
            paddle.enable_static()
            multi_rois = []
            multi_scores = []
            for i in range(4):
                multi_rois.append(paddle.static.data(
                    name='roi_'+str(i), shape=[None, 4], dtype='float32', lod_level=1))
            for i in range(4):
                multi_scores.append(paddle.static.data(
                    name='score_'+str(i), shape=[None, 1], dtype='float32', lod_level=1))

            fpn_rois = ops.collect_fpn_proposals(
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
        return output_rois, rois_num

    else:
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
        return output_rois, rois_num


@paddle.jit.not_to_static
def distribute_fpn_proposals(fpn_rois,
                             min_level,
                             max_level,
                             refer_level,
                             refer_scale,
                             pixel_offset=False,
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

            import paddle
            from ppdet.modeling import ops
            paddle.enable_static()
            fpn_rois = paddle.static.data(
                name='data', shape=[None, 4], dtype='float32', lod_level=1)
            multi_rois, restore_ind = ops.distribute_fpn_proposals(
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
                 refer_level, 'refer_scale', refer_scale, 'pixel_offset',
                 pixel_offset)
        multi_rois, restore_ind, rois_num_per_level = core.ops.distribute_fpn_proposals(
            fpn_rois, rois_num, num_lvl, num_lvl, *attrs)
        return multi_rois, restore_ind, rois_num_per_level

    else:
        check_variable_and_dtype(fpn_rois, 'fpn_rois', ['float32', 'float64'],
                                 'distribute_fpn_proposals')
        helper = LayerHelper('distribute_fpn_proposals', **locals())
        dtype = helper.input_dtype('fpn_rois')
        multi_rois = [
            helper.create_variable_for_type_inference(dtype)
            for i in range(num_lvl)
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
                'refer_scale': refer_scale,
                'pixel_offset': pixel_offset
            })
        return multi_rois, restore_ind, rois_num_per_level


@paddle.jit.not_to_static
def yolo_box(
        x,
        origin_shape,
        anchors,
        class_num,
        conf_thresh,
        downsample_ratio,
        clip_bbox=True,
        scale_x_y=1.,
        name=None, ):
    """

    This operator generates YOLO detection boxes from output of YOLOv3 network.
     
     The output of previous network is in shape [N, C, H, W], while H and W
     should be the same, H and W specify the grid size, each grid point predict
     given number boxes, this given number, which following will be represented as S,
     is specified by the number of anchors. In the second dimension(the channel
     dimension), C should be equal to S * (5 + class_num), class_num is the object
     category number of source dataset(such as 80 in coco dataset), so the
     second(channel) dimension, apart from 4 box location coordinates x, y, w, h,
     also includes confidence score of the box and class one-hot key of each anchor
     box.
     Assume the 4 location coordinates are :math:`t_x, t_y, t_w, t_h`, the box
     predictions should be as follows:
     $$
     b_x = \\sigma(t_x) + c_x
     $$
     $$
     b_y = \\sigma(t_y) + c_y
     $$
     $$
     b_w = p_w e^{t_w}
     $$
     $$
     b_h = p_h e^{t_h}
     $$
     in the equation above, :math:`c_x, c_y` is the left top corner of current grid
     and :math:`p_w, p_h` is specified by anchors.
     The logistic regression value of the 5th channel of each anchor prediction boxes
     represents the confidence score of each prediction box, and the logistic
     regression value of the last :attr:`class_num` channels of each anchor prediction
     boxes represents the classifcation scores. Boxes with confidence scores less than
     :attr:`conf_thresh` should be ignored, and box final scores is the product of
     confidence scores and classification scores.
     $$
     score_{pred} = score_{conf} * score_{class}
     $$

    Args:
        x (Tensor): The input tensor of YoloBox operator is a 4-D tensor with shape of [N, C, H, W].
                    The second dimension(C) stores box locations, confidence score and
                    classification one-hot keys of each anchor box. Generally, X should be the output of YOLOv3 network.
                    The data type is float32 or float64.
        origin_shape (Tensor): The image size tensor of YoloBox operator, This is a 2-D tensor with shape of [N, 2].
                    This tensor holds height and width of each input image used for resizing output box in input image
                    scale. The data type is int32.
        anchors (list|tuple): The anchor width and height, it will be parsed pair by pair.
        class_num (int): The number of classes to predict.
        conf_thresh (float): The confidence scores threshold of detection boxes. Boxes with confidence scores
                    under threshold should be ignored.
        downsample_ratio (int): The downsample ratio from network input to YoloBox operator input,
                    so 32, 16, 8 should be set for the first, second, and thrid YoloBox operators.
        clip_bbox (bool): Whether clip output bonding box in Input(ImgSize) boundary. Default true.
        scale_x_y (float): Scale the center point of decoded bounding box. Default 1.0.
        name (string): The default value is None.  Normally there is no need
                       for user to set this property.  For more information,
                       please refer to :ref:`api_guide_Name`

    Returns:
        boxes Tensor: A 3-D tensor with shape [N, M, 4], the coordinates of boxes,  N is the batch num,
                    M is output box number, and the 3rd dimension stores [xmin, ymin, xmax, ymax] coordinates of boxes.
        scores Tensor: A 3-D tensor with shape [N, M, :attr:`class_num`], the coordinates of boxes,  N is the batch num,
                    M is output box number.
                    
    Raises:
        TypeError: Attr anchors of yolo box must be list or tuple
        TypeError: Attr class_num of yolo box must be an integer
        TypeError: Attr conf_thresh of yolo box must be a float number

    Examples:

    .. code-block:: python

        import paddle
        from ppdet.modeling import ops
        
        paddle.enable_static()
        x = paddle.static.data(name='x', shape=[None, 255, 13, 13], dtype='float32')
        img_size = paddle.static.data(name='img_size',shape=[None, 2],dtype='int64')
        anchors = [10, 13, 16, 30, 33, 23]
        boxes,scores = ops.yolo_box(x=x, img_size=img_size, class_num=80, anchors=anchors,
                                        conf_thresh=0.01, downsample_ratio=32)
    """
    helper = LayerHelper('yolo_box', **locals())

    if not isinstance(anchors, list) and not isinstance(anchors, tuple):
        raise TypeError("Attr anchors of yolo_box must be list or tuple")
    if not isinstance(class_num, int):
        raise TypeError("Attr class_num of yolo_box must be an integer")
    if not isinstance(conf_thresh, float):
        raise TypeError("Attr ignore_thresh of yolo_box must be a float number")

    if in_dygraph_mode():
        attrs = ('anchors', anchors, 'class_num', class_num, 'conf_thresh',
                 conf_thresh, 'downsample_ratio', downsample_ratio, 'clip_bbox',
                 clip_bbox, 'scale_x_y', scale_x_y)
        boxes, scores = core.ops.yolo_box(x, origin_shape, *attrs)
        return boxes, scores
    else:
        boxes = helper.create_variable_for_type_inference(dtype=x.dtype)
        scores = helper.create_variable_for_type_inference(dtype=x.dtype)

        attrs = {
            "anchors": anchors,
            "class_num": class_num,
            "conf_thresh": conf_thresh,
            "downsample_ratio": downsample_ratio,
            "clip_bbox": clip_bbox,
            "scale_x_y": scale_x_y,
        }

        helper.append_op(
            type='yolo_box',
            inputs={
                "X": x,
                "ImgSize": origin_shape,
            },
            outputs={
                'Boxes': boxes,
                'Scores': scores,
            },
            attrs=attrs)
        return boxes, scores


@paddle.jit.not_to_static
def prior_box(input,
              image,
              min_sizes,
              max_sizes=None,
              aspect_ratios=[1.],
              variance=[0.1, 0.1, 0.2, 0.2],
              flip=False,
              clip=False,
              steps=[0.0, 0.0],
              offset=0.5,
              min_max_aspect_ratios_order=False,
              name=None):
    """

    This op generates prior boxes for SSD(Single Shot MultiBox Detector) algorithm.
    Each position of the input produce N prior boxes, N is determined by
    the count of min_sizes, max_sizes and aspect_ratios, The size of the
    box is in range(min_size, max_size) interval, which is generated in
    sequence according to the aspect_ratios.

    Parameters:
       input(Tensor): 4-D tensor(NCHW), the data type should be float32 or float64.
       image(Tensor): 4-D tensor(NCHW), the input image data of PriorBoxOp,
            the data type should be float32 or float64.
       min_sizes(list|tuple|float): the min sizes of generated prior boxes.
       max_sizes(list|tuple|None): the max sizes of generated prior boxes.
            Default: None.
       aspect_ratios(list|tuple|float): the aspect ratios of generated
            prior boxes. Default: [1.].
       variance(list|tuple): the variances to be encoded in prior boxes.
            Default:[0.1, 0.1, 0.2, 0.2].
       flip(bool): Whether to flip aspect ratios. Default:False.
       clip(bool): Whether to clip out-of-boundary boxes. Default: False.
       step(list|tuple): Prior boxes step across width and height, If
            step[0] equals to 0.0 or step[1] equals to 0.0, the prior boxes step across
            height or weight of the input will be automatically calculated.
            Default: [0., 0.]
       offset(float): Prior boxes center offset. Default: 0.5
       min_max_aspect_ratios_order(bool): If set True, the output prior box is
            in order of [min, max, aspect_ratios], which is consistent with
            Caffe. Please note, this order affects the weights order of
            convolution layer followed by and does not affect the final
            detection results. Default: False.
       name(str, optional): The default value is None.  Normally there is no need for 
            user to set this property. For more information, please refer to :ref:`api_guide_Name`

    Returns:
        Tuple: A tuple with two Variable (boxes, variances)

        boxes(Tensor): the output prior boxes of PriorBox.
        4-D tensor, the layout is [H, W, num_priors, 4].
        H is the height of input, W is the width of input,
        num_priors is the total box count of each position of input.

        variances(Tensor): the expanded variances of PriorBox.
        4-D tensor, the layput is [H, W, num_priors, 4].
        H is the height of input, W is the width of input
        num_priors is the total box count of each position of input

    Examples:
        .. code-block:: python

        import paddle
        from ppdet.modeling import ops

        paddle.enable_static()
        input = paddle.static.data(name="input", shape=[None,3,6,9])
        image = paddle.static.data(name="image", shape=[None,3,9,12])
        box, var = ops.prior_box(
                    input=input,
                    image=image,
                    min_sizes=[100.],
                    clip=True,
                    flip=True)
    """
    helper = LayerHelper("prior_box", **locals())
    dtype = helper.input_dtype()
    check_variable_and_dtype(
        input, 'input', ['uint8', 'int8', 'float32', 'float64'], 'prior_box')

    def _is_list_or_tuple_(data):
        return (isinstance(data, list) or isinstance(data, tuple))

    if not _is_list_or_tuple_(min_sizes):
        min_sizes = [min_sizes]
    if not _is_list_or_tuple_(aspect_ratios):
        aspect_ratios = [aspect_ratios]
    if not (_is_list_or_tuple_(steps) and len(steps) == 2):
        raise ValueError('steps should be a list or tuple ',
                         'with length 2, (step_width, step_height).')

    min_sizes = list(map(float, min_sizes))
    aspect_ratios = list(map(float, aspect_ratios))
    steps = list(map(float, steps))

    cur_max_sizes = None
    if max_sizes is not None and len(max_sizes) > 0 and max_sizes[0] > 0:
        if not _is_list_or_tuple_(max_sizes):
            max_sizes = [max_sizes]
        cur_max_sizes = max_sizes

    if in_dygraph_mode():
        attrs = ('min_sizes', min_sizes, 'aspect_ratios', aspect_ratios,
                 'variances', variance, 'flip', flip, 'clip', clip, 'step_w',
                 steps[0], 'step_h', steps[1], 'offset', offset,
                 'min_max_aspect_ratios_order', min_max_aspect_ratios_order)
        if cur_max_sizes is not None:
            attrs += ('max_sizes', cur_max_sizes)
        box, var = core.ops.prior_box(input, image, *attrs)
        return box, var
    else:
        attrs = {
            'min_sizes': min_sizes,
            'aspect_ratios': aspect_ratios,
            'variances': variance,
            'flip': flip,
            'clip': clip,
            'step_w': steps[0],
            'step_h': steps[1],
            'offset': offset,
            'min_max_aspect_ratios_order': min_max_aspect_ratios_order
        }

        if cur_max_sizes is not None:
            attrs['max_sizes'] = cur_max_sizes

        box = helper.create_variable_for_type_inference(dtype)
        var = helper.create_variable_for_type_inference(dtype)
        helper.append_op(
            type="prior_box",
            inputs={"Input": input,
                    "Image": image},
            outputs={"Boxes": box,
                     "Variances": var},
            attrs=attrs, )
        box.stop_gradient = True
        var.stop_gradient = True
        return box, var


@paddle.jit.not_to_static
def multiclass_nms(bboxes,
                   scores,
                   score_threshold,
                   nms_top_k,
                   keep_top_k,
                   nms_threshold=0.3,
                   normalized=True,
                   nms_eta=1.,
                   background_label=-1,
                   return_index=False,
                   return_rois_num=True,
                   rois_num=None,
                   name=None):
    """
    This operator is to do multi-class non maximum suppression (NMS) on
    boxes and scores.
    In the NMS step, this operator greedily selects a subset of detection bounding
    boxes that have high scores larger than score_threshold, if providing this
    threshold, then selects the largest nms_top_k confidences scores if nms_top_k
    is larger than -1. Then this operator pruns away boxes that have high IOU
    (intersection over union) overlap with already selected boxes by adaptive
    threshold NMS based on parameters of nms_threshold and nms_eta.
    Aftern NMS step, at most keep_top_k number of total bboxes are to be kept
    per image if keep_top_k is larger than -1.
    Args:
        bboxes (Tensor): Two types of bboxes are supported:
                           1. (Tensor) A 3-D Tensor with shape
                           [N, M, 4 or 8 16 24 32] represents the
                           predicted locations of M bounding bboxes,
                           N is the batch size. Each bounding box has four
                           coordinate values and the layout is
                           [xmin, ymin, xmax, ymax], when box size equals to 4.
                           2. (LoDTensor) A 3-D Tensor with shape [M, C, 4]
                           M is the number of bounding boxes, C is the
                           class number
        scores (Tensor): Two types of scores are supported:
                           1. (Tensor) A 3-D Tensor with shape [N, C, M]
                           represents the predicted confidence predictions.
                           N is the batch size, C is the class number, M is
                           number of bounding boxes. For each category there
                           are total M scores which corresponding M bounding
                           boxes. Please note, M is equal to the 2nd dimension
                           of BBoxes.
                           2. (LoDTensor) A 2-D LoDTensor with shape [M, C].
                           M is the number of bbox, C is the class number.
                           In this case, input BBoxes should be the second
                           case with shape [M, C, 4].
        background_label (int): The index of background label, the background
                                label will be ignored. If set to -1, then all
                                categories will be considered. Default: 0
        score_threshold (float): Threshold to filter out bounding boxes with
                                 low confidence score. If not provided,
                                 consider all boxes.
        nms_top_k (int): Maximum number of detections to be kept according to
                         the confidences after the filtering detections based
                         on score_threshold.
        nms_threshold (float): The threshold to be used in NMS. Default: 0.3
        nms_eta (float): The threshold to be used in NMS. Default: 1.0
        keep_top_k (int): Number of total bboxes to be kept per image after NMS
                          step. -1 means keeping all bboxes after NMS step.
        normalized (bool): Whether detections are normalized. Default: True
        return_index(bool): Whether return selected index. Default: False
        rois_num(Tensor): 1-D Tensor contains the number of RoIs in each image. 
            The shape is [B] and data type is int32. B is the number of images.
            If it is not None then return a list of 1-D Tensor. Each element 
            is the output RoIs' number of each image on the corresponding level
            and the shape is [B]. None by default.
        name(str): Name of the multiclass nms op. Default: None.
    Returns:
        A tuple with two Variables: (Out, Index) if return_index is True,
        otherwise, a tuple with one Variable(Out) is returned.
        Out: A 2-D LoDTensor with shape [No, 6] represents the detections.
        Each row has 6 values: [label, confidence, xmin, ymin, xmax, ymax]
        or A 2-D LoDTensor with shape [No, 10] represents the detections.
        Each row has 10 values: [label, confidence, x1, y1, x2, y2, x3, y3,
        x4, y4]. No is the total number of detections.
        If all images have not detected results, all elements in LoD will be
        0, and output tensor is empty (None).
        Index: Only return when return_index is True. A 2-D LoDTensor with
        shape [No, 1] represents the selected index which type is Integer.
        The index is the absolute value cross batches. No is the same number
        as Out. If the index is used to gather other attribute such as age,
        one needs to reshape the input(N, M, 1) to (N * M, 1) as first, where
        N is the batch size and M is the number of boxes.
    Examples:
        .. code-block:: python

            import paddle
            from ppdet.modeling import ops
            boxes = paddle.static.data(name='bboxes', shape=[81, 4],
                                      dtype='float32', lod_level=1)
            scores = paddle.static.data(name='scores', shape=[81],
                                      dtype='float32', lod_level=1)
            out, index = ops.multiclass_nms(bboxes=boxes,
                                            scores=scores,
                                            background_label=0,
                                            score_threshold=0.5,
                                            nms_top_k=400,
                                            nms_threshold=0.3,
                                            keep_top_k=200,
                                            normalized=False,
                                            return_index=True)
    """
    helper = LayerHelper('multiclass_nms3', **locals())

    if in_dygraph_mode():
        attrs = ('background_label', background_label, 'score_threshold',
                 score_threshold, 'nms_top_k', nms_top_k, 'nms_threshold',
                 nms_threshold, 'keep_top_k', keep_top_k, 'nms_eta', nms_eta,
                 'normalized', normalized)
        output, index, nms_rois_num = core.ops.multiclass_nms3(bboxes, scores,
                                                               rois_num, *attrs)
        if return_index:
            index = None
        return output, nms_rois_num, index

    else:
        output = helper.create_variable_for_type_inference(dtype=bboxes.dtype)
        index = helper.create_variable_for_type_inference(dtype='int')

        inputs = {'BBoxes': bboxes, 'Scores': scores}
        outputs = {'Out': output, 'Index': index}

        if rois_num is not None:
            inputs['RoisNum'] = rois_num

        if return_rois_num:
            nms_rois_num = helper.create_variable_for_type_inference(
                dtype='int32')
            outputs['NmsRoisNum'] = nms_rois_num

        helper.append_op(
            type="multiclass_nms3",
            inputs=inputs,
            attrs={
                'background_label': background_label,
                'score_threshold': score_threshold,
                'nms_top_k': nms_top_k,
                'nms_threshold': nms_threshold,
                'keep_top_k': keep_top_k,
                'nms_eta': nms_eta,
                'normalized': normalized
            },
            outputs=outputs)
        output.stop_gradient = True
        index.stop_gradient = True
        if not return_index:
            index = None
        if not return_rois_num:
            nms_rois_num = None

        return output, nms_rois_num, index


@paddle.jit.not_to_static
def matrix_nms(bboxes,
               scores,
               score_threshold,
               post_threshold,
               nms_top_k,
               keep_top_k,
               use_gaussian=False,
               gaussian_sigma=2.,
               background_label=0,
               normalized=True,
               return_index=False,
               return_rois_num=True,
               name=None):
    """
    **Matrix NMS**
    This operator does matrix non maximum suppression (NMS).
    First selects a subset of candidate bounding boxes that have higher scores
    than score_threshold (if provided), then the top k candidate is selected if
    nms_top_k is larger than -1. Score of the remaining candidate are then
    decayed according to the Matrix NMS scheme.
    Aftern NMS step, at most keep_top_k number of total bboxes are to be kept
    per image if keep_top_k is larger than -1.
    Args:
        bboxes (Tensor): A 3-D Tensor with shape [N, M, 4] represents the
                           predicted locations of M bounding bboxes,
                           N is the batch size. Each bounding box has four
                           coordinate values and the layout is
                           [xmin, ymin, xmax, ymax], when box size equals to 4.
                           The data type is float32 or float64.
        scores (Tensor): A 3-D Tensor with shape [N, C, M]
                           represents the predicted confidence predictions.
                           N is the batch size, C is the class number, M is
                           number of bounding boxes. For each category there
                           are total M scores which corresponding M bounding
                           boxes. Please note, M is equal to the 2nd dimension
                           of BBoxes. The data type is float32 or float64.
        score_threshold (float): Threshold to filter out bounding boxes with
                                 low confidence score.
        post_threshold (float): Threshold to filter out bounding boxes with
                                low confidence score AFTER decaying.
        nms_top_k (int): Maximum number of detections to be kept according to
                         the confidences after the filtering detections based
                         on score_threshold.
        keep_top_k (int): Number of total bboxes to be kept per image after NMS
                          step. -1 means keeping all bboxes after NMS step.
        use_gaussian (bool): Use Gaussian as the decay function. Default: False
        gaussian_sigma (float): Sigma for Gaussian decay function. Default: 2.0
        background_label (int): The index of background label, the background
                                label will be ignored. If set to -1, then all
                                categories will be considered. Default: 0
        normalized (bool): Whether detections are normalized. Default: True
        return_index(bool): Whether return selected index. Default: False
        return_rois_num(bool): whether return rois_num. Default: True
        name(str): Name of the matrix nms op. Default: None.
    Returns:
        A tuple with three Tensor: (Out, Index, RoisNum) if return_index is True,
        otherwise, a tuple with two Tensor (Out, RoisNum) is returned.
        Out (Tensor): A 2-D Tensor with shape [No, 6] containing the
             detection results.
             Each row has 6 values: [label, confidence, xmin, ymin, xmax, ymax]
             (After version 1.3, when no boxes detected, the lod is changed
             from {0} to {1})
        Index (Tensor): A 2-D Tensor with shape [No, 1] containing the
            selected indices, which are absolute values cross batches.
        rois_num (Tensor): A 1-D Tensor with shape [N] containing 
            the number of detected boxes in each image.
    Examples:
        .. code-block:: python
            import paddle
            from ppdet.modeling import ops
            boxes = paddle.static.data(name='bboxes', shape=[None,81, 4],
                                      dtype='float32', lod_level=1)
            scores = paddle.static.data(name='scores', shape=[None,81],
                                      dtype='float32', lod_level=1)
            out = ops.matrix_nms(bboxes=boxes, scores=scores, background_label=0,
                                 score_threshold=0.5, post_threshold=0.1,
                                 nms_top_k=400, keep_top_k=200, normalized=False)
    """
    check_variable_and_dtype(bboxes, 'BBoxes', ['float32', 'float64'],
                             'matrix_nms')
    check_variable_and_dtype(scores, 'Scores', ['float32', 'float64'],
                             'matrix_nms')
    check_type(score_threshold, 'score_threshold', float, 'matrix_nms')
    check_type(post_threshold, 'post_threshold', float, 'matrix_nms')
    check_type(nms_top_k, 'nums_top_k', int, 'matrix_nms')
    check_type(keep_top_k, 'keep_top_k', int, 'matrix_nms')
    check_type(normalized, 'normalized', bool, 'matrix_nms')
    check_type(use_gaussian, 'use_gaussian', bool, 'matrix_nms')
    check_type(gaussian_sigma, 'gaussian_sigma', float, 'matrix_nms')
    check_type(background_label, 'background_label', int, 'matrix_nms')

    if in_dygraph_mode():
        attrs = ('background_label', background_label, 'score_threshold',
                 score_threshold, 'post_threshold', post_threshold, 'nms_top_k',
                 nms_top_k, 'gaussian_sigma', gaussian_sigma, 'use_gaussian',
                 use_gaussian, 'keep_top_k', keep_top_k, 'normalized',
                 normalized)
        out, index, rois_num = core.ops.matrix_nms(bboxes, scores, *attrs)
        if not return_index:
            index = None
        if not return_rois_num:
            rois_num = None
        return out, rois_num, index
    else:
        helper = LayerHelper('matrix_nms', **locals())
        output = helper.create_variable_for_type_inference(dtype=bboxes.dtype)
        index = helper.create_variable_for_type_inference(dtype='int')
        outputs = {'Out': output, 'Index': index}
        if return_rois_num:
            rois_num = helper.create_variable_for_type_inference(dtype='int')
            outputs['RoisNum'] = rois_num

        helper.append_op(
            type="matrix_nms",
            inputs={'BBoxes': bboxes,
                    'Scores': scores},
            attrs={
                'background_label': background_label,
                'score_threshold': score_threshold,
                'post_threshold': post_threshold,
                'nms_top_k': nms_top_k,
                'gaussian_sigma': gaussian_sigma,
                'use_gaussian': use_gaussian,
                'keep_top_k': keep_top_k,
                'normalized': normalized
            },
            outputs=outputs)
        output.stop_gradient = True

        if not return_index:
            index = None
        if not return_rois_num:
            rois_num = None
        return output, rois_num, index


def bipartite_match(dist_matrix,
                    match_type=None,
                    dist_threshold=None,
                    name=None):
    """

    This operator implements a greedy bipartite matching algorithm, which is
    used to obtain the matching with the maximum distance based on the input
    distance matrix. For input 2D matrix, the bipartite matching algorithm can
    find the matched column for each row (matched means the largest distance),
    also can find the matched row for each column. And this operator only
    calculate matched indices from column to row. For each instance,
    the number of matched indices is the column number of the input distance
    matrix. **The OP only supports CPU**.

    There are two outputs, matched indices and distance.
    A simple description, this algorithm matched the best (maximum distance)
    row entity to the column entity and the matched indices are not duplicated
    in each row of ColToRowMatchIndices. If the column entity is not matched
    any row entity, set -1 in ColToRowMatchIndices.

    NOTE: the input DistMat can be LoDTensor (with LoD) or Tensor.
    If LoDTensor with LoD, the height of ColToRowMatchIndices is batch size.
    If Tensor, the height of ColToRowMatchIndices is 1.

    NOTE: This API is a very low level API. It is used by :code:`ssd_loss`
    layer. Please consider to use :code:`ssd_loss` instead.

    Args:
        dist_matrix(Tensor): This input is a 2-D LoDTensor with shape
            [K, M]. The data type is float32 or float64. It is pair-wise 
            distance matrix between the entities represented by each row and 
            each column. For example, assumed one entity is A with shape [K], 
            another entity is B with shape [M]. The dist_matrix[i][j] is the 
            distance between A[i] and B[j]. The bigger the distance is, the 
            better matching the pairs are. NOTE: This tensor can contain LoD 
            information to represent a batch of inputs. One instance of this 
            batch can contain different numbers of entities.
        match_type(str, optional): The type of matching method, should be
           'bipartite' or 'per_prediction'. None ('bipartite') by default.
        dist_threshold(float32, optional): If `match_type` is 'per_prediction',
            this threshold is to determine the extra matching bboxes based
            on the maximum distance, 0.5 by default.
        name(str, optional): For detailed information, please refer 
            to :ref:`api_guide_Name`. Usually name is no need to set and 
            None by default.
 
    Returns:
        Tuple:

        matched_indices(Tensor): A 2-D Tensor with shape [N, M]. The data
        type is int32. N is the batch size. If match_indices[i][j] is -1, it
        means B[j] does not match any entity in i-th instance.
        Otherwise, it means B[j] is matched to row
        match_indices[i][j] in i-th instance. The row number of
        i-th instance is saved in match_indices[i][j].

        matched_distance(Tensor): A 2-D Tensor with shape [N, M]. The data
        type is float32. N is batch size. If match_indices[i][j] is -1,
        match_distance[i][j] is also -1.0. Otherwise, assumed
        match_distance[i][j] = d, and the row offsets of each instance
        are called LoD. Then match_distance[i][j] =
        dist_matrix[d+LoD[i]][j].

    Examples:

        .. code-block:: python
            import paddle
            from ppdet.modeling import ops
            from ppdet.modeling.utils import iou_similarity

            paddle.enable_static()

            x = paddle.static.data(name='x', shape=[None, 4], dtype='float32')
            y = paddle.static.data(name='y', shape=[None, 4], dtype='float32')
            iou = iou_similarity(x=x, y=y)
            matched_indices, matched_dist = ops.bipartite_match(iou)
    """
    check_variable_and_dtype(dist_matrix, 'dist_matrix',
                             ['float32', 'float64'], 'bipartite_match')

    if in_dygraph_mode():
        match_indices, match_distance = core.ops.bipartite_match(
            dist_matrix, "match_type", match_type, "dist_threshold",
            dist_threshold)
        return match_indices, match_distance

    helper = LayerHelper('bipartite_match', **locals())
    match_indices = helper.create_variable_for_type_inference(dtype='int32')
    match_distance = helper.create_variable_for_type_inference(
        dtype=dist_matrix.dtype)
    helper.append_op(
        type='bipartite_match',
        inputs={'DistMat': dist_matrix},
        attrs={
            'match_type': match_type,
            'dist_threshold': dist_threshold,
        },
        outputs={
            'ColToRowMatchIndices': match_indices,
            'ColToRowMatchDist': match_distance
        })
    return match_indices, match_distance


@paddle.jit.not_to_static
def box_coder(prior_box,
              prior_box_var,
              target_box,
              code_type="encode_center_size",
              box_normalized=True,
              axis=0,
              name=None):
    """
    **Box Coder Layer**
    Encode/Decode the target bounding box with the priorbox information.
    
    The Encoding schema described below:
    .. math::
        ox = (tx - px) / pw / pxv
        oy = (ty - py) / ph / pyv
        ow = \log(\abs(tw / pw)) / pwv 
        oh = \log(\abs(th / ph)) / phv 
    The Decoding schema described below:
    
    .. math::
  
        ox = (pw * pxv * tx * + px) - tw / 2
        oy = (ph * pyv * ty * + py) - th / 2
        ow = \exp(pwv * tw) * pw + tw / 2
        oh = \exp(phv * th) * ph + th / 2   
    where `tx`, `ty`, `tw`, `th` denote the target box's center coordinates, 
    width and height respectively. Similarly, `px`, `py`, `pw`, `ph` denote 
    the priorbox's (anchor) center coordinates, width and height. `pxv`, 
    `pyv`, `pwv`, `phv` denote the variance of the priorbox and `ox`, `oy`, 
    `ow`, `oh` denote the encoded/decoded coordinates, width and height. 
    During Box Decoding, two modes for broadcast are supported. Say target 
    box has shape [N, M, 4], and the shape of prior box can be [N, 4] or 
    [M, 4]. Then prior box will broadcast to target box along the 
    assigned axis. 

    Args:
        prior_box(Tensor): Box list prior_box is a 2-D Tensor with shape 
            [M, 4] holds M boxes and data type is float32 or float64. Each box
            is represented as [xmin, ymin, xmax, ymax], [xmin, ymin] is the 
            left top coordinate of the anchor box, if the input is image feature
            map, they are close to the origin of the coordinate system. 
            [xmax, ymax] is the right bottom coordinate of the anchor box.       
        prior_box_var(List|Tensor|None): prior_box_var supports three types 
            of input. One is Tensor with shape [M, 4] which holds M group and 
            data type is float32 or float64. The second is list consist of 
            4 elements shared by all boxes and data type is float32 or float64. 
            Other is None and not involved in calculation. 
        target_box(Tensor): This input can be a 2-D LoDTensor with shape 
            [N, 4] when code_type is 'encode_center_size'. This input also can 
            be a 3-D Tensor with shape [N, M, 4] when code_type is 
            'decode_center_size'. Each box is represented as 
            [xmin, ymin, xmax, ymax]. The data type is float32 or float64. 
        code_type(str): The code type used with the target box. It can be
            `encode_center_size` or `decode_center_size`. `encode_center_size` 
            by default.
        box_normalized(bool): Whether treat the priorbox as a normalized box.
            Set true by default.
        axis(int): Which axis in PriorBox to broadcast for box decode, 
            for example, if axis is 0 and TargetBox has shape [N, M, 4] and 
            PriorBox has shape [M, 4], then PriorBox will broadcast to [N, M, 4]
            for decoding. It is only valid when code type is 
            `decode_center_size`. Set 0 by default. 
        name(str, optional): For detailed information, please refer 
            to :ref:`api_guide_Name`. Usually name is no need to set and 
            None by default. 

    Returns:
        Tensor:
        output_box(Tensor): When code_type is 'encode_center_size', the 
        output tensor of box_coder_op with shape [N, M, 4] representing the 
        result of N target boxes encoded with M Prior boxes and variances. 
        When code_type is 'decode_center_size', N represents the batch size 
        and M represents the number of decoded boxes.

    Examples:
 
        .. code-block:: python
 
            import paddle
            from ppdet.modeling import ops
            paddle.enable_static()
            # For encode
            prior_box_encode = paddle.static.data(name='prior_box_encode',
                                  shape=[512, 4],
                                  dtype='float32')
            target_box_encode = paddle.static.data(name='target_box_encode',
                                   shape=[81, 4],
                                   dtype='float32')
            output_encode = ops.box_coder(prior_box=prior_box_encode,
                                    prior_box_var=[0.1,0.1,0.2,0.2],
                                    target_box=target_box_encode,
                                    code_type="encode_center_size")
            # For decode
            prior_box_decode = paddle.static.data(name='prior_box_decode',
                                  shape=[512, 4],
                                  dtype='float32')
            target_box_decode = paddle.static.data(name='target_box_decode',
                                   shape=[512, 81, 4],
                                   dtype='float32')
            output_decode = ops.box_coder(prior_box=prior_box_decode,
                                    prior_box_var=[0.1,0.1,0.2,0.2],
                                    target_box=target_box_decode,
                                    code_type="decode_center_size",
                                    box_normalized=False,
                                    axis=1)
    """
    check_variable_and_dtype(prior_box, 'prior_box', ['float32', 'float64'],
                             'box_coder')
    check_variable_and_dtype(target_box, 'target_box', ['float32', 'float64'],
                             'box_coder')

    if in_dygraph_mode():
        if isinstance(prior_box_var, Variable):
            output_box = core.ops.box_coder(
                prior_box, prior_box_var, target_box, "code_type", code_type,
                "box_normalized", box_normalized, "axis", axis)

        elif isinstance(prior_box_var, list):
            output_box = core.ops.box_coder(
                prior_box, None, target_box, "code_type", code_type,
                "box_normalized", box_normalized, "axis", axis, "variance",
                prior_box_var)
        else:
            raise TypeError(
                "Input variance of box_coder must be Variable or list")
        return output_box
    else:
        helper = LayerHelper("box_coder", **locals())

        output_box = helper.create_variable_for_type_inference(
            dtype=prior_box.dtype)

        inputs = {"PriorBox": prior_box, "TargetBox": target_box}
        attrs = {
            "code_type": code_type,
            "box_normalized": box_normalized,
            "axis": axis
        }
        if isinstance(prior_box_var, Variable):
            inputs['PriorBoxVar'] = prior_box_var
        elif isinstance(prior_box_var, list):
            attrs['variance'] = prior_box_var
        else:
            raise TypeError(
                "Input variance of box_coder must be Variable or list")
        helper.append_op(
            type="box_coder",
            inputs=inputs,
            attrs=attrs,
            outputs={"OutputBox": output_box})
        return output_box


@paddle.jit.not_to_static
def generate_proposals(scores,
                       bbox_deltas,
                       im_shape,
                       anchors,
                       variances,
                       pre_nms_top_n=6000,
                       post_nms_top_n=1000,
                       nms_thresh=0.5,
                       min_size=0.1,
                       eta=1.0,
                       pixel_offset=False,
                       return_rois_num=False,
                       name=None):
    """
    **Generate proposal Faster-RCNN**
    This operation proposes RoIs according to each box with their
    probability to be a foreground object and 
    the box can be calculated by anchors. Bbox_deltais and scores
    to be an object are the output of RPN. Final proposals
    could be used to train detection net.
    For generating proposals, this operation performs following steps:
    1. Transposes and resizes scores and bbox_deltas in size of
       (H*W*A, 1) and (H*W*A, 4)
    2. Calculate box locations as proposals candidates. 
    3. Clip boxes to image
    4. Remove predicted boxes with small area. 
    5. Apply NMS to get final proposals as output.
    Args:
        scores(Tensor): A 4-D Tensor with shape [N, A, H, W] represents
            the probability for each box to be an object.
            N is batch size, A is number of anchors, H and W are height and
            width of the feature map. The data type must be float32.
        bbox_deltas(Tensor): A 4-D Tensor with shape [N, 4*A, H, W]
            represents the difference between predicted box location and
            anchor location. The data type must be float32.
        im_shape(Tensor): A 2-D Tensor with shape [N, 2] represents H, W, the
            origin image size or input size. The data type can be float32 or 
            float64.
        anchors(Tensor):   A 4-D Tensor represents the anchors with a layout
            of [H, W, A, 4]. H and W are height and width of the feature map,
            num_anchors is the box count of each position. Each anchor is
            in (xmin, ymin, xmax, ymax) format an unnormalized. The data type must be float32.
        variances(Tensor): A 4-D Tensor. The expanded variances of anchors with a layout of
            [H, W, num_priors, 4]. Each variance is in
            (xcenter, ycenter, w, h) format. The data type must be float32.
        pre_nms_top_n(float): Number of total bboxes to be kept per
            image before NMS. The data type must be float32. `6000` by default.
        post_nms_top_n(float): Number of total bboxes to be kept per
            image after NMS. The data type must be float32. `1000` by default.
        nms_thresh(float): Threshold in NMS. The data type must be float32. `0.5` by default.
        min_size(float): Remove predicted boxes with either height or
            width < min_size. The data type must be float32. `0.1` by default.
        eta(float): Apply in adaptive NMS, if adaptive `threshold > 0.5`,
            `adaptive_threshold = adaptive_threshold * eta` in each iteration.
        return_rois_num(bool): When setting True, it will return a 1D Tensor with shape [N, ] that includes Rois's 
            num of each image in one batch. The N is the image's num. For example, the tensor has values [4,5] that represents
            the first image has 4 Rois, the second image has 5 Rois. It only used in rcnn model. 
            'False' by default. 
        name(str, optional): For detailed information, please refer 
            to :ref:`api_guide_Name`. Usually name is no need to set and 
            None by default. 

    Returns:
        tuple:
        A tuple with format ``(rpn_rois, rpn_roi_probs)``.
        - **rpn_rois**: The generated RoIs. 2-D Tensor with shape ``[N, 4]`` while ``N`` is the number of RoIs. The data type is the same as ``scores``.
        - **rpn_roi_probs**: The scores of generated RoIs. 2-D Tensor with shape ``[N, 1]`` while ``N`` is the number of RoIs. The data type is the same as ``scores``.

    Examples:
        .. code-block:: python
        
            import paddle
            from ppdet.modeling import ops
            paddle.enable_static()
            scores = paddle.static.data(name='scores', shape=[None, 4, 5, 5], dtype='float32')
            bbox_deltas = paddle.static.data(name='bbox_deltas', shape=[None, 16, 5, 5], dtype='float32')
            im_shape = paddle.static.data(name='im_shape', shape=[None, 2], dtype='float32')
            anchors = paddle.static.data(name='anchors', shape=[None, 5, 4, 4], dtype='float32')
            variances = paddle.static.data(name='variances', shape=[None, 5, 10, 4], dtype='float32')
            rois, roi_probs = ops.generate_proposals(scores, bbox_deltas,
                         im_shape, anchors, variances)
    """
    if in_dygraph_mode():
        assert return_rois_num, "return_rois_num should be True in dygraph mode."
        attrs = ('pre_nms_topN', pre_nms_top_n, 'post_nms_topN', post_nms_top_n,
                 'nms_thresh', nms_thresh, 'min_size', min_size, 'eta', eta,
                 'pixel_offset', pixel_offset)
        rpn_rois, rpn_roi_probs, rpn_rois_num = core.ops.generate_proposals_v2(
            scores, bbox_deltas, im_shape, anchors, variances, *attrs)
        return rpn_rois, rpn_roi_probs, rpn_rois_num

    else:
        helper = LayerHelper('generate_proposals_v2', **locals())

        check_variable_and_dtype(scores, 'scores', ['float32'],
                                 'generate_proposals_v2')
        check_variable_and_dtype(bbox_deltas, 'bbox_deltas', ['float32'],
                                 'generate_proposals_v2')
        check_variable_and_dtype(im_shape, 'im_shape', ['float32', 'float64'],
                                 'generate_proposals_v2')
        check_variable_and_dtype(anchors, 'anchors', ['float32'],
                                 'generate_proposals_v2')
        check_variable_and_dtype(variances, 'variances', ['float32'],
                                 'generate_proposals_v2')

        rpn_rois = helper.create_variable_for_type_inference(
            dtype=bbox_deltas.dtype)
        rpn_roi_probs = helper.create_variable_for_type_inference(
            dtype=scores.dtype)
        outputs = {
            'RpnRois': rpn_rois,
            'RpnRoiProbs': rpn_roi_probs,
        }
        if return_rois_num:
            rpn_rois_num = helper.create_variable_for_type_inference(
                dtype='int32')
            rpn_rois_num.stop_gradient = True
            outputs['RpnRoisNum'] = rpn_rois_num

        helper.append_op(
            type="generate_proposals_v2",
            inputs={
                'Scores': scores,
                'BboxDeltas': bbox_deltas,
                'ImShape': im_shape,
                'Anchors': anchors,
                'Variances': variances
            },
            attrs={
                'pre_nms_topN': pre_nms_top_n,
                'post_nms_topN': post_nms_top_n,
                'nms_thresh': nms_thresh,
                'min_size': min_size,
                'eta': eta,
                'pixel_offset': pixel_offset
            },
            outputs=outputs)
        rpn_rois.stop_gradient = True
        rpn_roi_probs.stop_gradient = True

        return rpn_rois, rpn_roi_probs, rpn_rois_num


def sigmoid_cross_entropy_with_logits(input,
                                      label,
                                      ignore_index=-100,
                                      normalize=False):
    output = F.binary_cross_entropy_with_logits(input, label, reduction='none')
    mask_tensor = paddle.cast(label != ignore_index, 'float32')
    output = paddle.multiply(output, mask_tensor)
    output = paddle.reshape(output, shape=[output.shape[0], -1])
    if normalize:
        sum_valid_mask = paddle.sum(mask_tensor)
        output = output / sum_valid_mask
    return output


def smooth_l1(input, label, inside_weight=None, outside_weight=None,
              sigma=None):
    input_new = paddle.multiply(input, inside_weight)
    label_new = paddle.multiply(label, inside_weight)
    delta = 1 / (sigma * sigma)
    out = F.smooth_l1_loss(input_new, label_new, reduction='none', delta=delta)
    out = paddle.multiply(out, outside_weight)
    out = out / delta
    out = paddle.reshape(out, shape=[out.shape[0], -1])
    out = paddle.sum(out, axis=1)
    return out
