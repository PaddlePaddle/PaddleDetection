#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


def _get_class_default_kwargs(cls, *args, **kwargs):
    """
    Get default arguments of a class in dict format, if args and
    kwargs is specified, it will replace default arguments
    """
    varnames = cls.__init__.__code__.co_varnames
    argcount = cls.__init__.__code__.co_argcount
    keys = varnames[:argcount]
    assert keys[0] == 'self'
    keys = keys[1:]

    values = list(cls.__init__.__defaults__)
    assert len(values) == len(keys)

    if len(args) > 0:
        for i, arg in enumerate(args):
            values[i] = arg

    default_kwargs = dict(zip(keys, values))

    if len(kwargs) > 0:
        for k, v in kwargs.items():
            default_kwargs[k] = v

    return default_kwargs
