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
from . import transform
from .transform import operator, batch_operator

logger = logging.getLogger(__name__)


class Compose(object):
    def __init__(self, transforms, fields=None, from_=transform,
                 num_classes=81):
        self.transforms = transforms
        self.transforms_cls = []
        output_fields = None
        for t in self.transforms:
            for k, v in t.items():
                op_cls = getattr(from_, k)
                self.transforms_cls.append(op_cls(**v))
                if hasattr(op_cls, 'num_classes'):
                    op_cls.num_classes = num_classes

                # TODO: should be refined in the future
                if op_cls in [
                        transform.Gt2YoloTargetOp, transform.Gt2YoloTarget
                ]:
                    output_fields = ['image', 'gt_bbox']
                    output_fields.extend([
                        'target{}'.format(i)
                        for i in range(len(v['anchor_masks']))
                    ])

        self.fields = fields
        self.output_fields = output_fields if output_fields else fields

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

        if self.output_fields is not None:
            data_new = []
            for item in data:
                batch = []
                for k in self.output_fields:
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

    def __init__(self,
                 inputs_def=None,
                 sample_transforms=None,
                 batch_transforms=None,
                 batch_size=1,
                 shuffle=False,
                 drop_last=False,
                 drop_empty=True,
                 num_classes=81,
                 with_background=True,
                 **kwargs):
        # out fields 
        self._fields = inputs_def['fields'] if inputs_def else None
        # sample transform
        self._sample_transforms = Compose(
            sample_transforms, num_classes=num_classes)

        # batch transfrom 
        self._batch_transforms = None
        if batch_transforms:
            self._batch_transforms = Compose(batch_transforms,
                                             copy.deepcopy(self._fields),
                                             transform, num_classes)

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.with_background = with_background
        self.kwargs = kwargs

    def __call__(self,
                 dataset,
                 worker_num,
                 device=None,
                 batch_sampler=None,
                 return_list=False,
                 use_prefetch=True):
        self._dataset = dataset
        self._dataset.parse_dataset(self.with_background)
        # get data
        self._dataset.set_out(self._sample_transforms,
                              copy.deepcopy(self._fields))
        # set kwargs
        self._dataset.set_kwargs(**self.kwargs)
        # batch sampler
        if batch_sampler is None:
            self._batch_sampler = DistributedBatchSampler(
                self._dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                drop_last=self.drop_last)
        else:
            self._batch_sampler = batch_sampler

        loader = DataLoader(
            dataset=self._dataset,
            batch_sampler=self._batch_sampler,
            collate_fn=self._batch_transforms,
            num_workers=worker_num,
            places=device,
            return_list=return_list,
            use_buffer_reader=use_prefetch,
            use_shared_memory=False)

        return loader, len(self._batch_sampler)


@register
class TrainReader(BaseDataLoader):
    def __init__(self,
                 inputs_def=None,
                 sample_transforms=None,
                 batch_transforms=None,
                 batch_size=1,
                 shuffle=True,
                 drop_last=True,
                 drop_empty=True,
                 num_classes=81,
                 with_background=True,
                 **kwargs):
        super(TrainReader, self).__init__(inputs_def, sample_transforms,
                                          batch_transforms, batch_size, shuffle,
                                          drop_last, drop_empty, num_classes,
                                          with_background, **kwargs)


@register
class EvalReader(BaseDataLoader):
    def __init__(self,
                 inputs_def=None,
                 sample_transforms=None,
                 batch_transforms=None,
                 batch_size=1,
                 shuffle=False,
                 drop_last=True,
                 drop_empty=True,
                 num_classes=81,
                 with_background=True,
                 **kwargs):
        super(EvalReader, self).__init__(inputs_def, sample_transforms,
                                         batch_transforms, batch_size, shuffle,
                                         drop_last, drop_empty, num_classes,
                                         with_background, **kwargs)


@register
class TestReader(BaseDataLoader):
    def __init__(self,
                 inputs_def=None,
                 sample_transforms=None,
                 batch_transforms=None,
                 batch_size=1,
                 shuffle=False,
                 drop_last=False,
                 drop_empty=True,
                 num_classes=81,
                 with_background=True,
                 **kwargs):
        super(TestReader, self).__init__(inputs_def, sample_transforms,
                                         batch_transforms, batch_size, shuffle,
                                         drop_last, drop_empty, num_classes,
                                         with_background, **kwargs)
