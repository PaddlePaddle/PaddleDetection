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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import importlib
import os
import sys

import yaml
import copy
import collections

from .config.schema import SchemaDict, SharedConfig, extract_schema
from .config.yaml_helpers import serializable

__all__ = [
    'global_config',
    'load_config',
    'merge_config',
    'get_registered_modules',
    'create',
    'register',
    'serializable',
    'dump_value',
]


def dump_value(value):
    # XXX this is hackish, but collections.abc is not available in python 2
    if hasattr(value, '__dict__') or isinstance(value, (dict, tuple, list)):
        value = yaml.dump(value, default_flow_style=True)
        value = value.replace('\n', '')
        value = value.replace('...', '')
        return "'{}'".format(value)
    else:
        # primitive types
        return str(value)


class AttrDict(dict):
    """Single level attribute dict, NOT recursive"""

    def __init__(self, **kwargs):
        super(AttrDict, self).__init__()
        super(AttrDict, self).update(kwargs)

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError("object has no attribute '{}'".format(key))


global_config = AttrDict()

READER_KEY = '_READER_'


def load_config(file_path):
    """
    Load config from file.

    Args:
        file_path (str): Path of the config file to be loaded.

    Returns: global config
    """
    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"

    cfg = AttrDict()
    with open(file_path) as f:
        cfg = merge_config(yaml.load(f, Loader=yaml.Loader), cfg)

    if READER_KEY in cfg:
        reader_cfg = cfg[READER_KEY]
        if reader_cfg.startswith("~"):
            reader_cfg = os.path.expanduser(reader_cfg)
        if not reader_cfg.startswith('/'):
            reader_cfg = os.path.join(os.path.dirname(file_path), reader_cfg)

        with open(reader_cfg) as f:
            merge_config(yaml.load(f, Loader=yaml.Loader))
        del cfg[READER_KEY]

    merge_config(cfg)
    return global_config


def dict_merge(dct, merge_dct):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.

    Args:
        dct: dict onto which the merge is executed
        merge_dct: dct merged into dct

    Returns: dct
    """
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict) and
                isinstance(merge_dct[k], collections.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]
    return dct


def merge_config(config, another_cfg=None):
    """
    Merge config into global config or another_cfg.

    Args:
        config (dict): Config to be merged.

    Returns: global config
    """
    global global_config
    dct = another_cfg if another_cfg is not None else global_config
    return dict_merge(dct, config)


def get_registered_modules():
    return {k: v for k, v in global_config.items() if isinstance(v, SchemaDict)}


def make_partial(cls):
    op_module = importlib.import_module(cls.__op__.__module__)
    op = getattr(op_module, cls.__op__.__name__)
    cls.__category__ = getattr(cls, '__category__', None) or 'op'

    def partial_apply(self, *args, **kwargs):
        kwargs_ = self.__dict__.copy()
        kwargs_.update(kwargs)
        return op(*args, **kwargs_)

    if getattr(cls, '__append_doc__', True):  # XXX should default to True?
        if sys.version_info[0] > 2:
            cls.__doc__ = "Wrapper for `{}` OP".format(op.__name__)
            cls.__init__.__doc__ = op.__doc__
            cls.__call__ = partial_apply
            cls.__call__.__doc__ = op.__doc__
        else:
            # XXX work around for python 2
            partial_apply.__doc__ = op.__doc__
            cls.__call__ = partial_apply
    return cls


def register(cls):
    """
    Register a given module class.

    Args:
        cls (type): Module class to be registered.

    Returns: cls
    """
    if cls.__name__ in global_config:
        raise ValueError("Module class already registered: {}".format(
            cls.__name__))
    if hasattr(cls, '__op__'):
        cls = make_partial(cls)
    global_config[cls.__name__] = extract_schema(cls)
    return cls


def create(cls_or_name, **kwargs):
    """
    Create an instance of given module class.

    Args:
        cls_or_name (type or str): Class of which to create instance.

    Returns: instance of type `cls_or_name`
    """
    assert type(cls_or_name) in [type, str
                                 ], "should be a class or name of a class"
    name = type(cls_or_name) == str and cls_or_name or cls_or_name.__name__
    assert name in global_config and \
        isinstance(global_config[name], SchemaDict), \
        "the module {} is not registered".format(name)
    config = global_config[name]
    config.update(kwargs)
    config.validate()
    cls = getattr(config.pymodule, name)

    kwargs = {}
    kwargs.update(global_config[name])

    # parse `shared` annoation of registered modules
    if getattr(config, 'shared', None):
        for k in config.shared:
            target_key = config[k]
            shared_conf = config.schema[k].default
            assert isinstance(shared_conf, SharedConfig)
            if target_key is not None and not isinstance(target_key,
                                                         SharedConfig):
                continue  # value is given for the module
            elif shared_conf.key in global_config:
                # `key` is present in config
                kwargs[k] = global_config[shared_conf.key]
            else:
                kwargs[k] = shared_conf.default_value

    # parse `inject` annoation of registered modules
    if getattr(config, 'inject', None):
        for k in config.inject:
            target_key = config[k]
            # optional dependency
            if target_key is None:
                continue
            # also accept dictionaries and serialized objects
            if isinstance(target_key, dict) or hasattr(target_key, '__dict__'):
                continue
            elif isinstance(target_key, str):
                if target_key not in global_config:
                    raise ValueError("Missing injection config:", target_key)
                target = global_config[target_key]
                if isinstance(target, SchemaDict):
                    kwargs[k] = create(target_key)
                elif hasattr(target, '__dict__'):  # serialized object
                    kwargs[k] = target
            else:
                raise ValueError("Unsupported injection type:", target_key)
    # prevent modification of global config values of reference types
    # (e.g., list, dict) from within the created module instances
    kwargs = copy.deepcopy(kwargs)
    return cls(**kwargs)
