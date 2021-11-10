import math
import random
import numpy as np
from paddle import fluid

from nvidia.dali import Pipeline
import nvidia.dali.fn as fn
from nvidia.dali.ops.transforms import Rotation
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
from nvidia.dali.plugin.paddle import DALIGenericIterator, recursive_length, to_paddle_type, feed_ndarray, \
    lod_tensor_clip
from nvidia.dali.types import DALIDataType
from nvidia.dali.backend import TensorGPU, TensorListGPU

from ppdet.data.transform.batch_operators import BatchRandomResize
from ppdet.data.transform.operators import RandomDistort, RandomCrop, RandomFlip, RandomExpand, NormalizeImage, Permute


def generate_resize_feed(sizes, batch_size):
    def resize_feed():
        size = (np.full(2, random.choice(sizes)).astype(dtype=np.float32))
        out = [size for _ in range(batch_size)]
        return out

    return resize_feed


def generate_flip_feed(t, batch_size):
    def flip_feed():
        r = []
        for _ in range(batch_size):
            r.append(np.array(t.flip_queue.get(timeout=1), dtype=np.int32))
        return r

    return flip_feed


def generate_expand_feed(t, batch_size):
    def expand_feed():
        x_batch = []
        y_batch = []
        ratio_batch = []
        for _ in range(batch_size):
            x, y, ratio = t.expand_queue.get(timeout=1)
            x_batch.append(np.array(x, dtype=np.float32))
            y_batch.append(np.array(y, dtype=np.float32))
            ratio_batch.append(np.array(ratio, dtype=np.float32))
        return x_batch, y_batch, ratio_batch

    return expand_feed


def generate_crop_feed(t, batch_size):
    # crop_box_queue = [(0.1,0.1,60,80)]

    def crop_feed():
        # size = (np.array(crop_box_queue[0]).astype(dtype=np.float32))
        x = []
        y = []
        w = []
        h = []
        for _ in range(batch_size):
            crop_box = t.crop_box_queue.get(timeout=1)
            # print(crop_box)
            x.append(np.array(crop_box[0], dtype=np.float32))
            y.append(np.array(crop_box[1], dtype=np.float32))
            w.append(np.array(crop_box[2], dtype=np.float32))
            h.append(np.array(crop_box[3], dtype=np.float32))

        return x,y,w,h

    return crop_feed


# nearly same as DALIGenericIterator except disabling caching of pd_tensors (mainly
# caching tensor shape) in __next__ because tensor shape may vary across batches
class DALICOCOIterator(DALIGenericIterator):
    def __init__(self, *args, **kwargs):
        super(DALICOCOIterator, self).__init__(*args, **kwargs)

    def __next__(self):
        # self._batch_transforms[0].generate_target_size()

        if self._first_batch is not None:
            batch = self._first_batch
            self._first_batch = None
            return batch

        # Gather outputs
        outputs = self._get_outputs()

        for i in range(self._num_gpus):
            dev_id = self._pipes[i].device_id
            # Initialize dict for all output categories
            category_outputs = dict()
            # Segregate outputs into categories
            for j, out in enumerate(outputs[i]):
                category_outputs[self.output_map[j]] = out

            pd_gpu_place = fluid.CUDAPlace(dev_id)
            pd_cpu_place = fluid.CPUPlace()

            category_pd_type = dict()
            category_place = dict()
            category_tensors = dict()
            category_shapes = dict()
            category_lengths = dict()
            for cat, out in category_outputs.items():
                lod = self.normalized_map[cat]
                assert out.is_dense_tensor() or lod > 0, \
                    "non-dense tensor lists must have LoD > 0"

                if lod > 0:
                    # +1 for batch dim
                    seq_len = recursive_length(out, lod + 1)[1:]
                    shape = out.at(0).shape
                    if callable(shape):
                        shape = shape()
                    shape = [sum(seq_len[-1])] + list(shape[lod:])
                    category_shapes[cat] = shape
                    category_lengths[cat] = seq_len
                else:
                    out = out.as_tensor()
                    category_shapes[cat] = out.shape()
                    category_lengths[cat] = []

                category_tensors[cat] = out
                category_pd_type[cat] = to_paddle_type(out)
                if isinstance(out, (TensorGPU, TensorListGPU)):
                    category_place[cat] = pd_gpu_place
                else:
                    category_place[cat] = pd_cpu_place

            # disable caching for pd_tensors
            # if self._data_batches[i] is None:
            pd_tensors = {}
            for cat, lod in self.normalized_map.items():
                lod_tensor = fluid.core.LoDTensor()
                lod_tensor._set_dims(category_shapes[cat])
                pd_tensors[cat] = lod_tensor
            self._data_batches[i] = pd_tensors
            # else:
            #     pd_tensors = self._data_batches[i]

            # Copy data from DALI Tensors to LoDTensors
            for cat, tensor in category_tensors.items():
                if hasattr(tensor, 'shape'):  # could be tensor list
                    assert self._dynamic_shape or \
                        tensor.shape() == pd_tensors[cat].shape(), \
                        ("Shapes do not match: DALI tensor has size {0}, "
                         "but LoDTensor has size {1}".format(
                             tensor.shape(), pd_tensors[cat].shape()))

                lod_tensor = pd_tensors[cat]
                lod_tensor._set_dims(category_shapes[cat])
                seq_len = category_lengths[cat]
                lod_tensor.set_recursive_sequence_lengths(seq_len)
                ptr = lod_tensor._mutable_data(category_place[cat],
                                               category_pd_type[cat])
                feed_ndarray(tensor, ptr)

        self._schedule_runs()

        self._advance_and_check_drop_last()

        if self._reader_name:
            if_drop, left = self._remove_padded()
            if np.any(if_drop):
                output = []
                for batch, to_copy in zip(self._data_batches, left):
                    batch = batch.copy()
                    for cat in self.output_map:
                        batch[cat] = lod_tensor_clip(batch[cat], to_copy)
                    output.append(batch)
                return output

        else:
            if self._last_batch_policy == LastBatchPolicy.PARTIAL and (self._counter > self._size) and self._size > 0:
                # First calculate how much data is required to
                # return exactly self._size entries.
                diff = self._num_gpus * self.batch_size - (self._counter
                                                           - self._size)
                # Figure out how many GPUs to grab from.
                num_gpus_to_grab = int(math.ceil(diff / self.batch_size))
                # Figure out how many results to grab from the last GPU
                # (as a fractional GPU batch may be required to bring us
                # right up to self._size).
                mod_diff = diff % self.batch_size
                data_from_last_gpu = mod_diff if mod_diff else self.batch_size

                # Grab the relevant data.
                # 1) Grab everything from the relevant GPUs.
                # 2) Grab the right data from the last GPU.
                # 3) Append data together correctly and return.
                output = self._data_batches[0:num_gpus_to_grab]
                output[-1] = output[-1].copy()
                for cat in self.output_map:
                    lod_tensor = output[-1][cat]
                    output[-1][cat] = lod_tensor_clip(
                        lod_tensor, data_from_last_gpu)
                return output

        return self._data_batches


class COCOPipeline(Pipeline):
    def __init__(self, dataset, batch_size=1, transforms=None, batch_transforms=None):
        super().__init__(batch_size=batch_size,
                         num_threads=4,
                         device_id=0,
                         prefetch_queue_depth=2,
                         seed=0)
        self.transforms = transforms
        self.batch_transforms = batch_transforms
        self.dataset = dataset
        self.dataset_iter = iter(self.dataset)
        self.rot = Rotation()

    def define_graph(self):
        # faster rcnn layout is CHW
        # r = fn.external_source(source=self.dataset, batch=False, layout='CHW')
        # ppyolo layout is HWC
        # batch_seed = fn.external_source(source=[0,0])
        # TODO: sometimes image external source is executed after resize and crop
        #  external source: the cause is in Pipeline._build_graph, it will iterate nodes from
        #  outputs to inputs, then use reverse order to build node.
        #  (Note: after setting name parameter for external_source, the issue seems not happen again.
        #  Update: It happened still with low probability..)
        r = fn.external_source(source=self.dataset, batch=False, layout='HWC', name="image_source", device="gpu")
        for t in self.transforms.transforms_cls:
            if isinstance(t, RandomDistort):
                low, high, p = t.brightness
                # PaddleDetection doesn't normalize image before adjusting brightness
                low /= 255
                high /= 255
                b = fn.random.uniform(range=(low, high))
                r = fn.brightness(r, brightness_shift=b, device='gpu')

                low, high, p = t.contrast
                c = fn.random.uniform(range=(low, high))
                r = fn.contrast(r, contrast=c, contrast_center=0, device='gpu')

                low, high, p = t.saturation
                s = fn.random.uniform(range=(low, high))
                r = fn.saturation(r, saturation=s, device='gpu')

                low, high, p = t.hue
                h = fn.random.uniform(range=(low, high))
                r = fn.hue(r, hue=h, device='gpu')

            if isinstance(t, RandomExpand):
                x, y, ratio = fn.external_source(source=generate_expand_feed(t, self.batch_size), num_outputs=3)
                # TODO: first pad then crop?
                # r = fn.crop(r, crop=(300,300), crop_pos_x=-0.2, crop_pos_y=-0.2)
                # using pad operator with negative x and y will got error: [/opt/dali/dali/operators/image/crop/crop_attr.h:161] Assert on "anchor_norm[dim] >= 0.0f && anchor_norm[dim] <= 1.0f" failed: Anchor for dimension 0 (-0.200000) is out of range [0.0, 1.0]
                # so paste operator is suited better here
                r = fn.paste(r, fill_value=t.fill_value, paste_x=x, paste_y=y, ratio=ratio, device='gpu')

            if isinstance(t, RandomCrop):
                x,y,w,h = fn.external_source(source=generate_crop_feed(t, self.batch_size), num_outputs=4)
                r = fn.crop(r, crop_w=w, crop_h=h, crop_pos_x=x, crop_pos_y=y, device='gpu')

            if isinstance(t, RandomFlip):
                fl = fn.external_source(source=generate_flip_feed(t, self.batch_size))
                r = fn.flip(r, horizontal=fl, device='gpu')

        for t in self.batch_transforms.transforms_cls:
            if isinstance(t, BatchRandomResize):
                size = fn.external_source(source=generate_resize_feed(t.target_size, self.batch_size), name="resize_source")
                r = fn.resize(r, size=size, dtype=DALIDataType.FLOAT, device='gpu')

            if isinstance(t, NormalizeImage):
                r = r/255
                mean = fn.constant(fdata=t.mean)
                std = fn.constant(fdata=t.mean)
                # r = fn.normalize(r, mean=mean, stddev=std, axis_names='HW')
                r = fn.normalize(r, mean=0.5, stddev=1)

            if isinstance(t, Permute):
                r = fn.transpose(r, perm=[2, 0, 1])

        return r, size

    # nearly same as parent implementation, except we sort external source group using
    # batch property (coco image external source batch property is False, other external
    # sources are True. we need DALI load coco image external source first) because the
    # original implementation will lose order when converting set to list
    def _setup_input_callbacks(self):
        from nvidia.dali.external_source import _is_external_source_with_callback
        groups = set()
        for op in self._ops:
            if _is_external_source_with_callback(op):
                group = op._group
                # print(op.id, op.name)
                groups.add(group)
        groups = list(groups)
        groups = sorted(groups, key=lambda v:0 if v.batch==False else 1)
        # for group in groups:
        #     print(group.batch)
        self._input_callbacks = groups
        if self._py_num_workers == 0:
            self._parallel_input_callbacks = []
            self._seq_input_callbacks = self._input_callbacks
        else:
            self._parallel_input_callbacks = [group for group in groups if group.parallel]
            self._seq_input_callbacks = [group for group in groups if not group.parallel]

    # def iter_setup(self):
    #     try:
    #         p = self.source_iter.next()
    #     except:
    #         print("Exception occured")
    #         self.source_iter = iter(self.source)
    #         p = self.source_iter.next()
    #
    #     self.feed_input(self.inp, p)
    #
    #     # super().iter_setup()
