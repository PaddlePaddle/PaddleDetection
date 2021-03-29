# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

SIZE_UNIT = ['K', 'M', 'G', 'T']
SHM_QUERY_CMD = 'df -h'
SHM_KEY = 'shm'
SHM_DEFAULT_MOUNT = '/dev/shm'

# [ shared memory size check ]
# In detection models, image/target data occupies a lot of memory, and
# will occupy lots of shared memory in multi-process DataLoader, we use
# following code to get shared memory size and perform a size check to
# disable shared memory use if shared memory size is not enough.
# Shared memory getting process as follows:
# 1. use `df -h` get all mount info
# 2. pick up spaces whose mount info contains 'shm'
# 3. if 'shm' space number is only 1, return its size
# 4. if there are multiple 'shm' space, try to find the default mount
#    directory '/dev/shm' is Linux-like system, otherwise return the
#    biggest space size.


def _parse_size_in_M(size_str):
    num, unit = size_str[:-1], size_str[-1]
    assert unit in SIZE_UNIT, \
            "unknown shm size unit {}".format(unit)
    return float(num) * \
            (1024 ** (SIZE_UNIT.index(unit) - 1))


def _get_shared_memory_size_in_M():
    try:
        df_infos = os.popen(SHM_QUERY_CMD).readlines()
    except:
        return None
    else:
        shm_infos = []
        for df_info in df_infos:
            info = df_info.strip()
            if info.find(SHM_KEY) >= 0:
                shm_infos.append(info.split())

        if len(shm_infos) == 0:
            return None
        elif len(shm_infos) == 1:
            return _parse_size_in_M(shm_infos[0][3])
        else:
            default_mount_infos = [
                si for si in shm_infos if si[-1] == SHM_DEFAULT_MOUNT
            ]
            if default_mount_infos:
                return _parse_size_in_M(default_mount_infos[0][3])
            else:
                return max([_parse_size_in_M(si[3]) for si in shm_infos])
