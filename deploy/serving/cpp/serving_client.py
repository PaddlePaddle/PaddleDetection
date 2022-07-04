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

import os
import glob
import base64
import argparse
from paddle_serving_client import Client
from paddle_serving_client.proto import general_model_config_pb2 as m_config
import google.protobuf.text_format

parser = argparse.ArgumentParser(description="args for paddleserving")
parser.add_argument(
    "--serving_client", type=str, help="the directory of serving_client")
parser.add_argument("--image_dir", type=str)
parser.add_argument("--image_file", type=str)
parser.add_argument("--http_port", type=int, default=9997)
parser.add_argument(
    "--threshold", type=float, default=0.5, help="Threshold of score.")
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


def postprocess(fetch_dict, fetch_vars, draw_threshold=0.5):
    result = []
    if "conv2d_441.tmp_1" in fetch_dict:
        heatmap = fetch_dict["conv2d_441.tmp_1"]
        print(heatmap)
        result.append(heatmap)
    else:
        bboxes = fetch_dict[fetch_vars[0]]
        for bbox in bboxes:
            if bbox[0] > -1 and bbox[1] > draw_threshold:
                print(f"{int(bbox[0])} {bbox[1]} "
                      f"{bbox[2]} {bbox[3]} {bbox[4]} {bbox[5]}")
                result.append(f"{int(bbox[0])} {bbox[1]} "
                              f"{bbox[2]} {bbox[3]} {bbox[4]} {bbox[5]}")
    return result


def get_model_vars(client_config_dir):
    # read original serving_client_conf.prototxt
    client_config_file = os.path.join(client_config_dir,
                                      "serving_client_conf.prototxt")
    with open(client_config_file, 'r') as f:
        model_var = google.protobuf.text_format.Merge(
            str(f.read()), m_config.GeneralModelConfig())
    # modify feed_var to run core/general-server/op/
    [model_var.feed_var.pop() for _ in range(len(model_var.feed_var))]
    feed_var = m_config.FeedVar()
    feed_var.name = "input"
    feed_var.alias_name = "input"
    feed_var.is_lod_tensor = False
    feed_var.feed_type = 20
    feed_var.shape.extend([1])
    model_var.feed_var.extend([feed_var])
    with open(
            os.path.join(client_config_dir, "serving_client_conf_cpp.prototxt"),
            "w") as f:
        f.write(str(model_var))
    # get feed_vars/fetch_vars
    feed_vars = [var.name for var in model_var.feed_var]
    fetch_vars = [var.name for var in model_var.fetch_var]
    return feed_vars, fetch_vars


if __name__ == '__main__':
    url = f"127.0.0.1:{args.http_port}"
    logid = 10000
    img_list = get_test_images(args.image_dir, args.image_file)
    feed_vars, fetch_vars = get_model_vars(args.serving_client)

    client = Client()
    client.load_client_config(
        os.path.join(args.serving_client, "serving_client_conf_cpp.prototxt"))
    client.connect([url])

    for img_file in img_list:
        with open(img_file, 'rb') as file:
            image_data = file.read()
        image = base64.b64encode(image_data).decode('utf8')
        fetch_dict = client.predict(
            feed={feed_vars[0]: image}, fetch=fetch_vars)
        result = postprocess(fetch_dict, fetch_vars, args.threshold)
