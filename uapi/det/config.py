# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import io
import collections.abc

import yaml
# We have to ensure that all custom YAML constructors are registered
# This is done by importing ppdet
import ppdet
from ppdet.core.workspace import load_config, AttrDict, SchemaDict

from ..base import BaseConfig


class DetConfig(BaseConfig):
    def reset_from_dict(self, dict_like_obj):
        self._dict.clear()
        self._update_from_dict(dict_like_obj, self._dict)
    
    def load(self, config_path):
        # First we use ppdet API to parse the config files with inheritance
        cfg = load_config(config_path)
        # Since `load_config()` returns an `AttrDict` that contains `AttrDict` and `SchemaDict` objects, 
        # which can not be recognized by `_PPDetSerializableLoader` and `_PPDetSerializableDumper`, 
        # we recursively convert it to a plain dict.
        cfg = self._convert_to_plain_dict(cfg)
        cfg_in_single_str = yaml.dump(cfg)
        with io.StringIO(cfg_in_single_str) as f:
            dict_ = yaml.load(f, Loader=_PPDetSerializableLoader)
        if not isinstance(dict_, dict):
            raise TypeError
        self.reset_from_dict(dict_)

    def dump(self, config_path):
        with open(config_path, 'w') as f:
            yaml.dump(self.dict, f, Dumper=_PPDetSerializableDumper)
    
    def update(self, dict_like_obj):
        self._update_from_dict(dict_like_obj, self._dict)

    def update_dataset(self, dataset_path, dataset_type=None):
        if dataset_type is None:
            dataset_type = 'COCODataSet'
        if dataset_type == 'COCODataSet':
            ds_cfg = self._make_dataset_config(dataset_path)
        else:
            raise ValueError(f"{dataset_type} is not supported.")
        self.update(ds_cfg)
    
    def _make_dataset_config(self, dataset_root_path):
        return {'TrainDataset':
            _PPDetSerializableHandler('COCODataSet', {
                'image_dir': 'train',
                'anno_path': 'annotations/train.json',
                'dataset_dir': dataset_root_path,
                'data_fields': ['image', 'gt_bbox', 'gt_class', 'is_crowd']
            }), 
            'EvalDataset': _PPDetSerializableHandler('COCODataSet', {
                'image_dir': 'val',
                'anno_path': 'annotations/val.json',
                'dataset_dir': dataset_root_path
            }),
            'TestDataset': _PPDetSerializableHandler('ImageFolder', {
                'anno_path': 'annotations/val.json',
                'dataset_dir': dataset_root_path
            }),
        }
    
    def update_optimizer(self, optimizer_type):
        # Not yet implemented
        raise NotImplementedError

    def update_backbone(self, backbone_type):
        # Not yet implemented
        raise NotImplementedError

    def update_lr_scheduler(self, lr_scheduler_type):
        # Not yet implemented
        raise NotImplementedError

    def update_batch_size(self, batch_size, mode='train'):
        # Not yet implemented
        raise NotImplementedError

    def update_weight_decay(self, weight_decay):
        # Not yet implemented
        raise NotImplementedError

    def _update_device(self, device):
        assert device is not None, 'device should not be None'
        if ':' in device:
            device_type = device.split(':')[0]
        else:
            device_type = device
        self.update({'use_' + device_type: True})

    def _update_from_dict(self, src_dic, dst_dic):
        # Refer to 
        # https://github.com/Bobholamovic/PaddleDetection/blob/0eeb077e91bc8ec596b1748d6a61031b6e542ace/ppdet/core/workspace.py#L121
        # Additionally, this function deals with the case when `src_dic` or `dst_dic` contains `_PPDetSerializableHandler` objects.
        for k, v in src_dic.items():
            if isinstance(v, collections.abc.Mapping):
                if k in dst_dic:
                    if isinstance(dst_dic[k], _PPDetSerializableHandler):
                        self._update_sohandler(v, dst_dic[k])
                    else:
                        self._update_from_dict(v, dst_dic[k])
                else:
                    dst_dic[k] = v
            elif isinstance(v, _PPDetSerializableHandler):
                if k in dst_dic:
                    self._update_sohandler(v, dst_dic[k])
                else:
                    dst_dic[k] = v
                self._update_from_dict(v.dic, dst_dic)
            else:
                dst_dic[k] = v

    def _update_sohandler(self, src_obj, dst_handler):
        if isinstance(src_obj, collections.abc.Mapping):
            type_key = _PPDetSerializableHandler.TYPE_KEY
            if type_key in src_obj:
                dst_handler.tag = src_obj[type_key]
                dict_ = copy.deepcopy(src_obj)
                dict_.pop(type_key)
            else:
                dict_ = src_obj
            self._update_from_dict(dict_, dst_handler.dic)
        else:
            assert isinstance(dst_handler, _PPDetSerializableHandler)
            dst_handler.copy_from(src_obj)

    def _convert_to_plain_dict(self, ppdet_dict):
        dict_ = dict()
        for k, v in ppdet_dict.items():
            if isinstance(v, (AttrDict, SchemaDict)):
                v = self._convert_to_plain_dict(v)
            dict_[k] = v
        return dict_


class _PPDetSerializableHandler(object):
    TYPE_KEY = '_type_'
    
    def __init__(self, tag, dic):
        super().__init__()
        self.tag = tag
        self.dic = dic

    def __repr__(self):
        # TODO: Prettier format
        return repr({self.TYPE_KEY: self.tag, **self.dic})

    def copy_from(self, sohandler):
        self.tag = sohandler.tag
        self.dic.clear()
        self.dic.update(sohandler.dic)


class _PPDetSerializableConstructor(yaml.constructor.SafeConstructor):
    def construct_sohandler(self, tag_suffix, node):
        if not isinstance(node, yaml.nodes.MappingNode):
            raise TypeError("Currently, we can only handle a MappingNode.")
        mapping = self.construct_mapping(node)
        return _PPDetSerializableHandler(tag_suffix, mapping)


class _PPDetSerializableLoader(_PPDetSerializableConstructor, yaml.loader.SafeLoader):
    def __init__(self, stream):
        _PPDetSerializableConstructor.__init__(self)
        yaml.loader.SafeLoader.__init__(self, stream)


class _PPDetSerializableRepresenter(yaml.representer.SafeRepresenter):
    def represent_sohandler(self, data):
        # XXX: Manually represent a serializable object according to the rules defined in
        # https://github.com/Bobholamovic/PaddleDetection/blob/0eeb077e91bc8ec596b1748d6a61031b6e542ace/ppdet/core/config/yaml_helpers.py#L58
        # We prepend a '!' to reconstruct the complete tag
        tag = u'!'+data.tag
        return self.represent_mapping(tag, data.dic)


class _PPDetSerializableDumper(_PPDetSerializableRepresenter, yaml.dumper.SafeDumper):
    def __init__(self, stream,
            default_style=None, default_flow_style=False,
            canonical=None, indent=None, width=None,
            allow_unicode=None, line_break=None,
            encoding=None, explicit_start=None, explicit_end=None,
            version=None, tags=None, sort_keys=True):

        _PPDetSerializableRepresenter.__init__(self, default_style=default_style,
                default_flow_style=default_flow_style, sort_keys=sort_keys)

        yaml.dumper.SafeDumper.__init__(self,  stream,
                                        default_style=default_style,
                                        default_flow_style=default_flow_style,
                                        canonical=canonical,
                                        indent=indent,
                                        width=width,
                                        allow_unicode=allow_unicode,
                                        line_break=line_break,
                                        encoding=encoding,
                                        explicit_start=explicit_start,
                                        explicit_end=explicit_end,
                                        version=version,
                                        tags=tags,
                                        sort_keys=sort_keys)

# We note that all custom tags defined in ppdet starts with a '!'.
# We assume that all unknown tags in the config file corresponds to a serializable class defined in ppdet.
_PPDetSerializableLoader.add_multi_constructor(u'!', _PPDetSerializableConstructor.construct_sohandler)
_PPDetSerializableDumper.add_representer(_PPDetSerializableHandler, _PPDetSerializableRepresenter.represent_sohandler)