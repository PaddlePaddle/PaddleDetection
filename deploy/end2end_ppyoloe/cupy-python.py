import sys
import requests
import cv2
import random
import time
import numpy as np
import cupy as cp
import tensorrt as trt
from PIL import Image
from collections import OrderedDict, namedtuple
from pathlib import Path


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)


names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush']
colors = {name: [random.randint(0, 255) for _ in range(3)] for i, name in enumerate(names)}

url = 'https://oneflow-static.oss-cn-beijing.aliyuncs.com/tripleMu/image1.jpg'
file = requests.get(url)
img = cv2.imdecode(np.frombuffer(file.content, np.uint8), 1)

w = Path(sys.argv[1])

assert w.exists() and w.suffix in ('.engine', '.plan'), 'Wrong engine path'

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

mean = cp.asarray(mean)
std = cp.asarray(std)

# Infer TensorRT Engine
Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
logger = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(logger, namespace="")
with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
    model = runtime.deserialize_cuda_engine(f.read())
bindings = OrderedDict()
fp16 = False  # default updated below
for index in range(model.num_bindings):
    name = model.get_binding_name(index)
    dtype = trt.nptype(model.get_binding_dtype(index))
    shape = tuple(model.get_binding_shape(index))
    data = cp.empty(shape, dtype=cp.dtype(dtype))
    bindings[name] = Binding(name, dtype, shape, data, int(data.data.ptr))
    if model.binding_is_input(index) and dtype == np.float16:
        fp16 = True
binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
context = model.create_execution_context()

image = img.copy()
image, ratio, dwdh = letterbox(image, auto=False)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_copy = image.copy()

image = image.transpose((2, 0, 1))
image = np.expand_dims(image, 0)
image = np.ascontiguousarray(image)

im = cp.asarray(image)
im = im.astype(cp.float32)
im /= 255
im -= mean
im /= std

# warmup for 10 times
for _ in range(10):
    tmp = cp.random.randn(1, 3, 640, 640).astype(cp.float32)
    binding_addrs['image'] = int(tmp.data.ptr)
    context.execute_v2(list(binding_addrs.values()))

start = time.perf_counter()
binding_addrs['image'] = int(im.data.ptr)
context.execute_v2(list(binding_addrs.values()))
print(f'Cost {(time.perf_counter() - start) * 1000}ms')

nums = bindings['num_dets'].data
boxes = bindings['det_boxes'].data
scores = bindings['det_scores'].data
classes = bindings['det_classes'].data

num = int(nums[0][0])
box_img = boxes[0, :num].round().astype(cp.int32)
score_img = scores[0, :num]
clss_img = classes[0, :num]
for i, (box, score, clss) in enumerate(zip(box_img, score_img, clss_img)):
    name = names[int(clss)]
    color = colors[name]
    cv2.rectangle(image_copy, box[:2].tolist(), box[2:].tolist(), color, 2)
    cv2.putText(image_copy, name, (int(box[0]), int(box[1]) - 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, [225, 255, 255], thickness=2)

cv2.imshow('Result', cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
