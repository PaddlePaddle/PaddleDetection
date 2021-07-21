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
import collections

try:
    collectionsAbc = collections.abc
except AttributeError:
    collectionsAbc = collections

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

BASE_KEY = '_BASE_'


# parse and load _BASE_ recursively
def _load_config_with_base(file_path):
    with open(file_path) as f:
        file_cfg = yaml.load(f, Loader=yaml.Loader)

    # NOTE: cfgs outside have higher priority than cfgs in _BASE_
    if BASE_KEY in file_cfg:
        all_base_cfg = AttrDict()
        base_ymls = list(file_cfg[BASE_KEY])
        for base_yml in base_ymls:
            if base_yml.startswith("~"):
                base_yml = os.path.expanduser(base_yml)
            if not base_yml.startswith('/'):
                base_yml = os.path.join(os.path.dirname(file_path), base_yml)

            with open(base_yml) as f:
                base_cfg = _load_config_with_base(base_yml)
                all_base_cfg = merge_config(base_cfg, all_base_cfg)

        del file_cfg[BASE_KEY]
        return merge_config(file_cfg, all_base_cfg)

    return file_cfg


def load_config(file_path):
    """
    Load config from file.

    Args:
        file_path (str): Path of the config file to be loaded.

    Returns: global config
    """
    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"

    # load config from file and merge into global config
    cfg = _load_config_with_base(file_path)
    cfg['filename'] = os.path.splitext(os.path.split(file_path)[-1])[0]
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
                isinstance(merge_dct[k], collectionsAbc.Mapping)):
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
    dct = another_cfg or global_config
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
    cls = getattr(config.pymodule, name)
    cls_kwargs = {}
    cls_kwargs.update(global_config[name])

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
                cls_kwargs[k] = global_config[shared_conf.key]
            else:
                cls_kwargs[k] = shared_conf.default_value

    # parse `inject` annoation of registered modules
    if getattr(cls, 'from_config', None):
        cls_kwargs.update(cls.from_config(config, **kwargs))

    if getattr(config, 'inject', None):
        for k in config.inject:
            target_key = config[k]
            # optional dependency
            if target_key is None:
                continue

            if isinstance(target_key, dict) or hasattr(target_key, '__dict__'):
                if 'name' not in target_key.keys():
                    continue
                inject_name = str(target_key['name'])
                if inject_name not in global_config:
                    raise ValueError(
                        "Missing injection name {} and check it's name in cfg file".
                        format(k))
                target = global_config[inject_name]
                for i, v in target_key.items():
                    if i == 'name':
                        continue
                    target[i] = v
                if isinstance(target, SchemaDict):
                    cls_kwargs[k] = create(inject_name)
            elif isinstance(target_key, str):
                if target_key not in global_config:
                    raise ValueError("Missing injection config:", target_key)
                target = global_config[target_key]
                if isinstance(target, SchemaDict):
                    cls_kwargs[k] = create(target_key)
                elif hasattr(target, '__dict__'):  # serialized object
                    cls_kwargs[k] = target
            else:
                raise ValueError("Unsupported injection type:", target_key)
    # prevent modification of global config values of reference types
    # (e.g., list, dict) from within the created module instances
    #kwargs = copy.deepcopy(kwargs)
    return cls(**cls_kwargs)
