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

import importlib
import inspect

import yaml
from .schema import SharedConfig

__all__ = ['serializable', 'Callable']


def represent_dictionary_order(self, dict_data):
    return self.represent_mapping('tag:yaml.org,2002:map', dict_data.items())


def setup_orderdict():
    from collections import OrderedDict
    yaml.add_representer(OrderedDict, represent_dictionary_order)


def _make_python_constructor(cls):
    def python_constructor(loader, node):
        if isinstance(node, yaml.SequenceNode):
            args = loader.construct_sequence(node, deep=True)
            return cls(*args)
        else:
            kwargs = loader.construct_mapping(node, deep=True)
            try:
                return cls(**kwargs)
            except Exception as ex:
                print("Error when construct {} instance from yaml config".
                      format(cls.__name__))
                raise ex

    return python_constructor


def _make_python_representer(cls):
    # python 2 compatibility
    if hasattr(inspect, 'getfullargspec'):
        argspec = inspect.getfullargspec(cls)
    else:
        argspec = inspect.getargspec(cls.__init__)
    argnames = [arg for arg in argspec.args if arg != 'self']

    def python_representer(dumper, obj):
        if argnames:
            data = {name: getattr(obj, name) for name in argnames}
        else:
            data = obj.__dict__
        if '_id' in data:
            del data['_id']
        return dumper.represent_mapping(u'!{}'.format(cls.__name__), data)

    return python_representer


def serializable(cls):
    """
    Add loader and dumper for given class, which must be
    "trivially serializable"

    Args:
        cls: class to be serialized

    Returns: cls
    """
    yaml.add_constructor(u'!{}'.format(cls.__name__),
                         _make_python_constructor(cls))
    yaml.add_representer(cls, _make_python_representer(cls))
    return cls


yaml.add_representer(SharedConfig,
                     lambda d, o: d.represent_data(o.default_value))


@serializable
class Callable(object):
    """
    Helper to be used in Yaml for creating arbitrary class objects

    Args:
        full_type (str): the full module path to target function
    """

    def __init__(self, full_type, args=[], kwargs={}):
        super(Callable, self).__init__()
        self.full_type = full_type
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        if '.' in self.full_type:
            idx = self.full_type.rfind('.')
            module = importlib.import_module(self.full_type[:idx])
            func_name = self.full_type[idx + 1:]
        else:
            try:
                module = importlib.import_module('builtins')
            except Exception:
                module = importlib.import_module('__builtin__')
            func_name = self.full_type

        func = getattr(module, func_name)
        return func(*self.args, **self.kwargs)
