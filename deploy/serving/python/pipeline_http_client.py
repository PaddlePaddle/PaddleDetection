# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import glob
import requests
import json
import base64
import os
import argparse

parser = argparse.ArgumentParser(description="args for paddleserving")
parser.add_argument("--image_dir", type=str)
parser.add_argument("--image_file", type=str)
parser.add_argument("--http_port", type=int, default=18093)
parser.add_argument("--service_name", type=str, default="ppdet")
args = parser.parse_args()


def get_test_images(infer_dir, infer_img):
    """
    Get image path list in TEST mode
    """
    assert infer_img is not None or infer_dir is not None, \
        "--image_file or --image_dir should be set"
    assert infer_img is None or os.path.isfile(infer_img), \
            "{} is not a file".format(infer_img)
    assert infer_dir is None or os.path.isdir(infer_dir), \
            "{} is not a directory".format(infer_dir)

    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        return [infer_img]

    images = set()
    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), \
        "infer_dir {} is not a directory".format(infer_dir)
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
    images = list(images)

    assert len(images) > 0, "no image found in {}".format(infer_dir)
    print("Found {} inference images in total.".format(len(images)))

    return images


if __name__ == "__main__":
    url = f"http://127.0.0.1:{args.http_port}/{args.service_name}/prediction"
    logid = 10000
    img_list = get_test_images(args.image_dir, args.image_file)

    for img_file in img_list:
        with open(img_file, 'rb') as file:
            image_data = file.read()

        # base64 encode
        image = base64.b64encode(image_data).decode('utf8')

        data = {"key": ["image_0"], "value": [image], "logid": logid}
        # send requests
        r = requests.post(url=url, data=json.dumps(data))
        print(r.json())
