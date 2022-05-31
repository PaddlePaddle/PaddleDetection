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

import inspect
import importlib
import re

try:
    from docstring_parser import parse as doc_parse
except Exception:

    def doc_parse(*args):
        pass


try:
    from typeguard import check_type
except Exception:

    def check_type(*args):
        pass


__all__ = ['SchemaValue', 'SchemaDict', 'SharedConfig', 'extract_schema']


class SchemaValue(object):
    def __init__(self, name, doc='', type=None):
        super(SchemaValue, self).__init__()
        self.name = name
        self.doc = doc
        self.type = type

    def set_default(self, value):
        self.default = value

    def has_default(self):
        return hasattr(self, 'default')


class SchemaDict(dict):
    def __init__(self, **kwargs):
        super(SchemaDict, self).__init__()
        self.schema = {}
        self.strict = False
        self.doc = ""
        self.update(kwargs)

    def __setitem__(self, key, value):
        # XXX also update regular dict to SchemaDict??
        if isinstance(value, dict) and key in self and isinstance(self[key],
                                                                  SchemaDict):
            self[key].update(value)
        else:
            super(SchemaDict, self).__setitem__(key, value)

    def __missing__(self, key):
        if self.has_default(key):
            return self.schema[key].default
        elif key in self.schema:
            return self.schema[key]
        else:
            raise KeyError(key)

    def copy(self):
        newone = SchemaDict()
        newone.__dict__.update(self.__dict__)
        newone.update(self)
        return newone

    def set_schema(self, key, value):
        assert isinstance(value, SchemaValue)
        self.schema[key] = value

    def set_strict(self, strict):
        self.strict = strict

    def has_default(self, key):
        return key in self.schema and self.schema[key].has_default()

    def is_default(self, key):
        if not self.has_default(key):
            return False
        if hasattr(self[key], '__dict__'):
            return True
        else:
            return key not in self or self[key] == self.schema[key].default

    def find_default_keys(self):
        return [
            k for k in list(self.keys()) + list(self.schema.keys())
            if self.is_default(k)
        ]

    def mandatory(self):
        return any([k for k in self.schema.keys() if not self.has_default(k)])

    def find_missing_keys(self):
        missing = [
            k for k in self.schema.keys()
            if k not in self and not self.has_default(k)
        ]
        placeholders = [k for k in self if self[k] in ('<missing>', '<value>')]
        return missing + placeholders

    def find_extra_keys(self):
        return list(set(self.keys()) - set(self.schema.keys()))

    def find_mismatch_keys(self):
        mismatch_keys = []
        for arg in self.schema.values():
            if arg.type is not None:
                try:
                    check_type("{}.{}".format(self.name, arg.name),
                               self[arg.name], arg.type)
                except Exception:
                    mismatch_keys.append(arg.name)
        return mismatch_keys

    def validate(self):
        missing_keys = self.find_missing_keys()
        if missing_keys:
            raise ValueError("Missing param for class<{}>: {}".format(
                self.name, ", ".join(missing_keys)))
        extra_keys = self.find_extra_keys()
        if extra_keys and self.strict:
            raise ValueError("Extraneous param for class<{}>: {}".format(
                self.name, ", ".join(extra_keys)))
        mismatch_keys = self.find_mismatch_keys()
        if mismatch_keys:
            raise TypeError("Wrong param type for class<{}>: {}".format(
                self.name, ", ".join(mismatch_keys)))


class SharedConfig(object):
    """
    Representation class for `__shared__` annotations, which work as follows:

    - if `key` is set for the module in config file, its value will take
      precedence
    - if `key` is not set for the module but present in the config file, its
      value will be used
    - otherwise, use the provided `default_value` as fallback

    Args:
        key: config[key] will be injected
        default_value: fallback value
    """

    def __init__(self, key, default_value=None):
        super(SharedConfig, self).__init__()
        self.key = key
        self.default_value = default_value


def extract_schema(cls):
    """
    Extract schema from a given class

    Args:
        cls (type): Class from which to extract.

    Returns:
        schema (SchemaDict): Extracted schema.
    """
    ctor = cls.__init__
    # python 2 compatibility
    if hasattr(inspect, 'getfullargspec'):
        argspec = inspect.getfullargspec(ctor)
        annotations = argspec.annotations
        has_kwargs = argspec.varkw is not None
    else:
        argspec = inspect.getfullargspec(ctor)
        # python 2 type hinting workaround, see pep-3107
        # however, since `typeguard` does not support python 2, type checking
        # is still python 3 only for now
        annotations = getattr(ctor, '__annotations__', {})
        has_kwargs = argspec.varkw is not None

    names = [arg for arg in argspec.args if arg != 'self']
    defaults = argspec.defaults
    num_defaults = argspec.defaults is not None and len(argspec.defaults) or 0
    num_required = len(names) - num_defaults

    docs = cls.__doc__
    if docs is None and getattr(cls, '__category__', None) == 'op':
        docs = cls.__call__.__doc__
    try:
        docstring = doc_parse(docs)
    except Exception:
        docstring = None

    if docstring is None:
        comments = {}
    else:
        comments = {}
        for p in docstring.params:
            match_obj = re.match('^([a-zA-Z_]+[a-zA-Z_0-9]*).*', p.arg_name)
            if match_obj is not None:
                comments[match_obj.group(1)] = p.description

    schema = SchemaDict()
    schema.name = cls.__name__
    schema.doc = ""
    if docs is not None:
        start_pos = docs[0] == '\n' and 1 or 0
        schema.doc = docs[start_pos:].split("\n")[0].strip()
    # XXX handle paddle's weird doc convention
    if '**' == schema.doc[:2] and '**' == schema.doc[-2:]:
        schema.doc = schema.doc[2:-2].strip()
    schema.category = hasattr(cls, '__category__') and getattr(
        cls, '__category__') or 'module'
    schema.strict = not has_kwargs
    schema.pymodule = importlib.import_module(cls.__module__)
    schema.inject = getattr(cls, '__inject__', [])
    schema.shared = getattr(cls, '__shared__', [])
    for idx, name in enumerate(names):
        comment = name in comments and comments[name] or name
        if name in schema.inject:
            type_ = None
        else:
            type_ = name in annotations and annotations[name] or None
        value_schema = SchemaValue(name, comment, type_)
        if name in schema.shared:
            assert idx >= num_required, "shared config must have default value"
            default = defaults[idx - num_required]
            value_schema.set_default(SharedConfig(name, default))
        elif idx >= num_required:
            default = defaults[idx - num_required]
            value_schema.set_default(default)
        schema.set_schema(name, value_schema)

    return schema
