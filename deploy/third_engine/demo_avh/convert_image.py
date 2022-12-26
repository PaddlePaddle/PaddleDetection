# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import pathlib
import re
import sys
import cv2
import math
from PIL import Image
import numpy as np


def resize_norm_img(img, image_shape, padding=True):
    imgC, imgH, imgW = image_shape
    img = cv2.resize(img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, [2, 0, 1]) / 255
    img = np.expand_dims(img, 0)
    img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std
    return img.astype(np.float32)


def create_header_file(name, tensor_name, tensor_data, output_path):
    """
    This function generates a header file containing the data from the numpy array provided.
    """
    file_path = pathlib.Path(f"{output_path}/" + name).resolve()
    # Create header file with npy_data as a C array
    raw_path = file_path.with_suffix(".h").resolve()
    with open(raw_path, "a") as header_file:
        header_file.write(
            "\n" + f"const size_t {tensor_name}_len = {tensor_data.size};\n" +
            f'__attribute__((section(".data.tvm"), aligned(16))) float {tensor_name}[] = '
        )

        header_file.write("{")
        for i in np.ndindex(tensor_data.shape):
            header_file.write(f"{tensor_data[i]}, ")
        header_file.write("};\n\n")


def create_headers(image_name):
    """
    This function generates C header files for the input and output arrays required to run inferences
    """
    img_path = os.path.join("./", f"{image_name}")

    # Resize image to 32x320
    img = cv2.imread(img_path)
    img = resize_norm_img(img, [3, 32, 320])
    img_data = img.astype("float32")

    # # Add the batch dimension, as we are expecting 4-dimensional input: NCHW.
    img_data = np.expand_dims(img_data, axis=0)

    if os.path.exists("./include/inputs.h"):
        os.remove("./include/inputs.h")
    if os.path.exists("./include/outputs.h"):
        os.remove("./include/outputs.h")
    # Create input header file
    create_header_file("inputs", "input", img_data, "./include")
    # Create output header file
    output_data = np.zeros([8500], np.float32)
    create_header_file(
        "outputs",
        "output0",
        output_data,
        "./include", )
    output_data = np.zeros([170000], np.float32)
    create_header_file(
        "outputs",
        "output1",
        output_data,
        "./include", )


if __name__ == "__main__":
    create_headers(sys.argv[1])
