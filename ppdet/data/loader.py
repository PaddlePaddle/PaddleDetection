import copy
import traceback
import logging
import threading
import sys
if sys.version_info >= (3, 0):
    import queue as Queue
else:
    import Queue
import numpy as np
from paddle.io import DataLoader
from ppdet.core.workspace import register, serializable, create
from .sampler import DistributedBatchSampler
from .transform import operators
from .transform import batch_operators

logger = logging.getLogger(__name__)


class Compose(object):
    def __init__(self, transforms, fields=None, from_=operators,
                 num_classes=81):
        self.transforms = transforms
        self.transforms_cls = []
        for t in self.transforms:
            for k, v in t.items():
                op_cls = getattr(from_, k)
                self.transforms_cls.append(op_cls(**v))
                if hasattr(op_cls, 'num_classes'):
                    op_cls.num_classes = num_classes

        self.fields = fields

    def __call__(self, data):
        if self.fields is not None:
            data_new = []
            for item in data:
                data_new.append(dict(zip(self.fields, item)))
            data = data_new

        for f in self.transforms_cls:
            try:
                data = f(data)
            except Exception as e:
                stack_info = traceback.format_exc()
                logger.warn("fail to map op [{}] with error: {} and stack:\n{}".
                            format(f, e, str(stack_info)))
                raise e

        if self.fields is not None:
            data_new = []
            for item in data:
                batch = []
                for k in self.fields:
                    batch.append(item[k])
                data_new.append(batch)
            batch_size = len(data_new)
            data_new = list(zip(*data_new))
            if batch_size > 1:
                data = [
                    np.array(item).astype(item[0].dtype) for item in data_new
                ]
            else:
                data = data_new

        return data


class BaseDataLoader(object):
    __share__ = ['num_classes']
    __inject__ = ['dataset']

    def __init__(self,
                 inputs_def=None,
                 dataset=None,
                 sample_transforms=None,
                 batch_transforms=None,
                 batch_size=1,
                 shuffle=False,
                 drop_last=False,
                 drop_empty=True,
                 num_classes=81):
        # dataset 
        self._dataset = dataset
        self._dataset.parse_dataset()
        # out fields 
        self._fields = copy.deepcopy(inputs_def[
            'fields']) if inputs_def else None
        # sample transform
        self._sample_transforms = Compose(
            sample_transforms, num_classes=num_classes)
        # get data 
        self._dataset.set_out(self._sample_transforms, self._fields)

        # batch transfrom 
        if batch_transforms:
            self._batch_transforms = Compose(batch_transforms, self._fields,
                                             batch_operators, num_classes)

        # batch sampler  
        self._batch_sampler = DistributedBatchSampler(
            self._dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last)

        self.batch_size = batch_size

    def __call__(self,
                 worker_num,
                 device,
                 return_list=False,
                 use_prefetch=False):

        loader = DataLoader(
            dataset=self._dataset,
            batch_sampler=self._batch_sampler,
            collate_fn=self._batch_transforms,
            num_workers=worker_num,
            use_buffer_reader=use_prefetch,
            places=device,
            return_list=return_list)

        return loader, len(self._batch_sampler)


@register
class TrainReader(BaseDataLoader):
    def __init__(self,
                 inputs_def=None,
                 dataset=None,
                 sample_transforms=None,
                 batch_transforms=None,
                 batch_size=1,
                 shuffle=False,
                 drop_last=False,
                 drop_empty=True,
                 num_classes=81):
        super(TrainReader, self).__init__(
            inputs_def, dataset, sample_transforms, batch_transforms,
            batch_size, shuffle, drop_last, drop_empty, num_classes)


@register
class EvalReader(BaseDataLoader):
    def __init__(self,
                 inputs_def=None,
                 dataset=None,
                 sample_transforms=None,
                 batch_transforms=None,
                 batch_size=1,
                 shuffle=False,
                 drop_last=False,
                 drop_empty=True,
                 num_classes=81):
        super(EvalReader, self).__init__(inputs_def, dataset, sample_transforms,
                                         batch_transforms, batch_size, shuffle,
                                         drop_last, drop_empty, num_classes)


@register
class TestReader(BaseDataLoader):
    def __init__(self,
                 inputs_def=None,
                 dataset=None,
                 sample_transforms=None,
                 batch_transforms=None,
                 batch_size=1,
                 shuffle=False,
                 drop_last=False,
                 drop_empty=True,
                 num_classes=81):
        super(TestReader, self).__init__(inputs_def, dataset, sample_transforms,
                                         batch_transforms, batch_size, shuffle,
                                         drop_last, drop_empty, num_classes)
