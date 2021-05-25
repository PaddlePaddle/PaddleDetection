# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from . import (core, data, engine, modeling, model_zoo, optimizer, metrics,
               utils, slim)


try:
    from .version import full_version as __version__
    from .version import commit as __git_commit__
except ImportError:
    import sys
    sys.stderr.write("Warning: import ppdet from source directory " \
            "without installing, run 'python setup.py install' to " \
            "install ppdet firstly\n")
