from ppdet.core.workspace import register
from ppdet.modeling.bbox_utils import distance2bbox, bbox2distance

__all__ = ['DistanceBBoxCoder']


@register
class DistanceBBoxCoder:
    """Transform distance prediction to bounding box and vice.
    """

    def __init__(self):
        pass

    def encode(self, points, bbox, max_dis=None, eps=0.1):
        return bbox2distance(points, bbox, max_dis, eps)

    def decode(self, points, distance, max_shape=None):
        return distance2bbox(points, distance, max_shape)
