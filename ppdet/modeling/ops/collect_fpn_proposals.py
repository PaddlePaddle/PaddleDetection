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
                multi_rois.append(fluid.data(
                    name='roi_'+str(i), shape=[None, 4], dtype='float32', lod_level=1))
            for i in range(4):
                multi_scores.append(fluid.data(
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


class CollectFpnProposals(layers.Layer):
    """
    See collect_fpn_proposals
    """

    def __init__(self, min_level, max_level, post_nms_top_n, name=None):
        super(CollectFpnProposals, self).__init__()
        self.min_level = min_level
        self.max_level = max_level
        self.post_nms_top_n = post_nms_top_n
        self.name = name

    def forward(self, multi_rois, multi_scores, rois_num_per_level):
        assert rois_num_per_level is not None, 'rois_num_per_level should not be None in CollectFpnProposals'
        return collect_fpn_proposals(
            multi_rois,
            multi_scores,
            self.min_level,
            self.max_level,
            self.post_nms_top_n,
            rois_num_per_level=rois_num_per_level,
            name=self.name)
