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

import requests
import json
import cv2
import base64
import time
import numpy as np


def cv2_to_base64(image):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')


def base64_to_cv2(b64str):
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data


# Send HTTP request
org_im = cv2.cvtColor(cv2.imread('demo_images/test.jpg'), cv2.COLOR_BGR2RGB)
h, w, c = org_im.shape
hat_ids = 1
data = {
    'images': [cv2_to_base64(org_im)],
    'background': 3,
    "beard": 2,
    "glasses": 3,
    "hat": 3
}
headers = {"Content-type": "application/json"}
url = "http://127.0.0.1:8880/predict/solov2_blazeface"
start = time.time()
r = requests.post(url=url, headers=headers, data=json.dumps(data))
end = time.time()
print('cost:', end - start)
result = base64_to_cv2(r.json()["results"]['image'])
cv2.imwrite("chrismas_final.png", result)
