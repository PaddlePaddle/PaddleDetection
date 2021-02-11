# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
import paddlehub as hub
import cv2
from PIL import Image
import numpy as np
import base64

img_file = 'demo_images/test.jpg'
background = 'element_source/background/1.png'
beard_file = 'element_source/beard/1.png'
glasses_file = 'element_source/glasses/4.png'
hat_file = 'element_source/hat/1.png'

model = hub.Module(name='solov2_blazeface', use_gpu=True)
output = model.predict(
    image=img_file,
    background=background,
    hat_file=hat_file,
    beard_file=beard_file,
    glasses_file=glasses_file,
    visualization=True)
cv2.imwrite("chrismas_final.png", output)
