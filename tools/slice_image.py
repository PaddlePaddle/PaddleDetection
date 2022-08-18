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

import argparse
from tqdm import tqdm


def slice_data(image_dir, dataset_json_path, output_dir, slice_size,
               overlap_ratio):
    try:
        from sahi.scripts.slice_coco import slice
    except Exception as e:
        raise RuntimeError(
            'Unable to use sahi to slice images, please install sahi, for example: `pip install sahi`, see https://github.com/obss/sahi'
        )
    tqdm.write(
        f" slicing for slice_size={slice_size}, overlap_ratio={overlap_ratio}")
    slice(
        image_dir=image_dir,
        dataset_json_path=dataset_json_path,
        output_dir=output_dir,
        slice_size=slice_size,
        overlap_ratio=overlap_ratio, )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir', type=str, default=None, help="The image folder path.")
    parser.add_argument(
        '--json_path', type=str, default=None, help="Dataset json path.")
    parser.add_argument(
        '--output_dir', type=str, default=None, help="Output dir.")
    parser.add_argument(
        '--slice_size', type=int, default=500, help="slice_size")
    parser.add_argument(
        '--overlap_ratio', type=float, default=0.25, help="overlap_ratio")
    args = parser.parse_args()

    slice_data(args.image_dir, args.json_path, args.output_dir, args.slice_size,
               args.overlap_ratio)


if __name__ == "__main__":
    main()
