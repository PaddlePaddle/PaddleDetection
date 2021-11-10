import numbers
from typing import Mapping, Sequence

import numpy as np
import paddle
from paddle.fluid import layers

def dali_default_collate_fn(batch):
    """
    Default batch collating function for :code:`paddle.io.DataLoader`,
    get input data as a list of sample datas, each element in list
    if the data of a sample, and sample data should composed of list,
    dictionary, string, number, numpy array and paddle.Tensor, this
    function will parse input data recursively and stack number,
    numpy array and paddle.Tensor datas as batch datas. e.g. for
    following input data:

    [{'image': np.array(shape=[3, 224, 224]), 'label': 1},
     {'image': np.array(shape=[3, 224, 224]), 'label': 3},
     {'image': np.array(shape=[3, 224, 224]), 'label': 4},
     {'image': np.array(shape=[3, 224, 224]), 'label': 5},]


    This default collate function zipped each number and numpy array
    field together and stack each field as the batch field as follows:

    {'image': np.array(shape=[4, 3, 224, 224]), 'label': np.array([1, 3, 4, 5])}


    Args:
        batch(list of sample data): batch should be a list of sample data.

    Returns:
        Batched data: batched each number, numpy array and paddle.Tensor
                      in input data.
    """
    sample = batch[0]
    if isinstance(sample, np.ndarray):
        # no need for all samples shape in one batch to be same since it will be
        # overwritten using data from DALI pipeline
        shapes = {arr.shape for arr in batch}
        if len(shapes) != 1:
            return batch

        batch = np.stack(batch, axis=0)
        return batch
    elif isinstance(sample, paddle.Tensor):
        return layers.stack(batch, axis=0)
    elif isinstance(sample, numbers.Number):
        batch = np.array(batch)
        return batch
    elif isinstance(sample, (str, bytes)):
        return batch
    elif isinstance(sample, Mapping):
        return {
            key: dali_default_collate_fn([d[key] for d in batch])
            for key in sample
        }
    elif isinstance(sample, Sequence):
        sample_fields_num = len(sample)
        if not all(len(sample) == sample_fields_num for sample in iter(batch)):
            raise RuntimeError(
                "fileds number not same among samples in a batch")
        return [dali_default_collate_fn(fields) for fields in zip(*batch)]

    raise TypeError("batch data con only contains: tensor, numpy.ndarray, "
                    "dict, list, number, but got {}".format(type(sample)))
    return outputs
