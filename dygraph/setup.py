# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import os.path as osp
import glob
import shutil
from setuptools import find_packages, setup


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


def parse_requirements(fname):
    with open(fname, encoding="utf-8-sig") as f:
        requirements = f.readlines()
    return requirements


def package_model_zoo():
    from ppdet.model_zoo import MODEL_ZOO_FILENAME
    cur_dir = osp.dirname(osp.realpath(__file__))
    cfg_dir = osp.join(cur_dir, "configs")
    cfgs = glob.glob(osp.join(cfg_dir, '*/*.yml'))

    valid_cfgs = []
    for cfg in cfgs:
        # exclude dataset base config
        if osp.split(osp.split(cfg)[0])[1] not in ['datasets']:
            valid_cfgs.append(cfg)
    model_names = [osp.splitext(osp.split(cfg)[1])[0] for cfg in valid_cfgs]

    model_zoo_file = osp.join(cur_dir, 'ppdet', 'model_zoo', MODEL_ZOO_FILENAME)
    with open(model_zoo_file, 'w') as wf:
        for model_name in model_names:
            wf.write("{}\n".format(model_name))

    return [model_zoo_file]


packages = [
    'ppdet',
    'ppdet.core',
    'ppdet.data',
    'ppdet.engine',
    'ppdet.metrics',
    'ppdet.modeling',
    'ppdet.model_zoo',
    'ppdet.py_op',
    'ppdet.utils',
]

if __name__ == "__main__":
    setup(
        name='ppdet',
        packages=find_packages(exclude=("configs", "tools", "deploy")),
        package_data={'ppdet.model_zoo': package_model_zoo()},
        author='PaddlePaddle',
        version='2.0-rc',
        install_requires=parse_requirements('./requirements.txt'),
        description='Object detection and instance segmentation toolkit based on PaddlePaddle',
        long_description=readme(),
        long_description_content_type='text/markdown',
        url='https://github.com/PaddlePaddle/PaddleDetection',
        download_url='https://github.com/PaddlePaddle/PaddleDetection.git',
        keywords=['ppdet paddle ppyolo'],
        classifiers=[
            'Intended Audience :: Developers',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Natural Language :: Chinese (Simplified)',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7', 'Topic :: Utilities'
        ],
        license='Apache License 2.0',
        ext_modules=[], )
