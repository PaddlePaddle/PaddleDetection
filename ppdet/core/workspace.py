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
from copy import deepcopy

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
        self.root = None

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError("object has no attribute '{}'".format(key))

    def __deepcopy__(self, memo):
        cls = self.__class__
        newone = cls.__new__(cls)
        # set current config to memo, all sub-object SchemaDict
        # set root attribute to its global config, use for create
        # function to find current processing config by SchemaDict
        memo['root'] = newone
        for k, v in self.items():
            newone[k] = deepcopy(v, memo)
        return newone


class AttrStr(str):
    def __new__(cls, value, *args, **kwargs):
        return str.__new__(cls, value)

    def __init__(self, value, root=None):
        self.root = root


global_config = AttrDict()

# NOTE: in order for loading multiple configs and create multiple
#       models, load_config return a unique config AttrDict for
#       each config filename, this 'filename: config AttrDict'
#       stores in CONFIGS
CONFIGS = {}

BASE_KEY = '_BASE_'


def _get_config_by_filename(filename):
    if filename not in CONFIGS:
        new_global_config = deepcopy(global_config)
        new_global_config.root = new_global_config
        CONFIGS[filename] = new_global_config
    return CONFIGS[filename]


def _parse_attr_str(config):
    def _parse_recursive(config, root):
        if isinstance(config, collectionsAbc.Mapping):
            for k, v in config.items():
                if isinstance(v, str):
                    config[k] = AttrStr(v)
                    config[k].root = root
                elif isinstance(v, (collectionsAbc.Mapping,
                                  collectionsAbc.Sequence)):
                    _parse_recursive(v, root)
        elif isinstance(config, (list, tuple)):
            for i, v in enumerate(config):
                if isinstance(v, str):
                    config[i] = AttrStr(v)
                    config[i].root = root
                elif isinstance(v, (collectionsAbc.Mapping,
                                  collectionsAbc.Sequence)):
                    _parse_recursive(v, root)

    _parse_recursive(config, config)


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

    unique_cfg = _get_config_by_filename(cfg['filename'])
    merge_config(cfg, unique_cfg)
    _parse_attr_str(unique_cfg)

    return unique_cfg


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


def merge_config(config, dst_config=None):
    """
    Merge config into global config or another_cfg.

    Args:
        config (dict): Config to be merged.
        dst_config (dict): Config to merge to.

    Returns: global config
    """
    if dst_config is not None:
        return dict_merge(dst_config, config)

    num_cfg = len(CONFIGS)
    if num_cfg > 1:
        raise RuntimeError(
            "You are processing {} configs at the "
            "same time(call load_config {} times to load "
            "different config files), you should call merge_config"
            "function with specified config in this situation, "
            "e.g. merge_config(cfg1, cfg)".format(num_cfg, num_cfg))
    elif num_cfg == 0:
        raise RuntimeError("no config loaded, please call load_config "
                           "firstly")
    else:
        dst_config = list(CONFIGS.values())[0]

    return dict_merge(dst_config, config)


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
    # add a link SchemaDict -> config AttrDict, for find config AttrDict
    # in create, while create only pass SchemaDict as input
    global_config[cls.__name__].set_root(global_config)
    return cls


def create(cls_or_name, config=None, **kwargs):
    """
    Create an instance of given module class.

    Args:
        cls_or_name (type or str): Class of which to create instance.
        config (AttrDict): the config object which you are currently
                           processing, this must be set if you are
                           processing multiple configs.

    Returns: instance of type `cls_or_name`
    """
    assert type(cls_or_name) in [type, str, AttrStr
                                 ], "should be a class or name of a class"
    name = type(cls_or_name) != type and cls_or_name or cls_or_name.__name__

    cur_global_config = config or getattr(cls_or_name, 'root', None)
    if cur_global_config is None:
        num_cfg = len(CONFIGS)
        if num_cfg == 1:
            cur_global_config = list(CONFIGS.values())[0]
        else:
            raise RuntimeError(
                "You are processing {} configs at the "
                "same time(call load_config {} times to load "
                "different config files), you should call create "
                "function with specified config in this situation, "
                "e.g. create(cfg.architecture, config=cfg)".format(num_cfg,
                                                                   num_cfg))

    assert name in cur_global_config and \
        isinstance(cur_global_config[name], SchemaDict), \
        "the module {} is not registered".format(name)
    cfg = cur_global_config[name]
    cls = getattr(cfg.pymodule, name)
    cls_kwargs = {}
    cls_kwargs.update(cur_global_config[name])

    # parse `shared` annoation of registered modules
    if getattr(cfg, 'shared', None):
        for k in cfg.shared:
            target_key = cfg[k]
            shared_conf = cfg.schema[k].default
            assert isinstance(shared_conf, SharedConfig)
            if target_key is not None and not isinstance(target_key,
                                                         SharedConfig):
                continue  # value is given for the module
            elif shared_conf.key in cur_global_config:
                # `key` is present in config
                cls_kwargs[k] = cur_global_config[shared_conf.key]
            else:
                cls_kwargs[k] = shared_conf.default_value

    # parse `inject` annoation of registered modules
    if getattr(cls, 'from_config', None):
        cls_kwargs.update(cls.from_config(cfg, **kwargs))

    if getattr(cfg, 'inject', None):
        for k in cfg.inject:
            target_key = cfg[k]
            # optional dependency
            if target_key is None:
                continue

            if isinstance(target_key, str):
                if target_key not in cur_global_config:
                    raise ValueError("Missing injection config:", target_key)
                target = cur_global_config[target_key]
                if isinstance(target, SchemaDict):
                    cls_kwargs[k] = create(target_key, cur_global_config)
                elif hasattr(target, '__dict__'):  # serialized object
                    cls_kwargs[k] = target
            elif isinstance(target_key, dict) or hasattr(target_key, '__dict__'):
                if 'name' not in target_key.keys():
                    continue
                inject_name = str(target_key['name'])
                if inject_name not in cur_global_config:
                    raise ValueError(
                        "Missing injection name {} and check it's name in cfg file".
                        format(k))
                target = cur_global_config[inject_name]
                for i, v in target_key.items():
                    if i == 'name':
                        continue
                    target[i] = v
                if isinstance(target, SchemaDict):
                    cls_kwargs[k] = create(inject_name, cur_global_config)
            else:
                raise ValueError("Unsupported injection type:", target_key)

    # prevent modification of global config values of reference types
    # (e.g., list, dict) from within the created module instances
    #kwargs = copy.deepcopy(kwargs)
    return cls(**cls_kwargs)
