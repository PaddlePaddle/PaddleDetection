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

from __future__ import print_function

import re
import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter

import yaml

from ppdet.core.workspace import get_registered_modules, load_config
from ppdet.utils.cli import ColorTTY

color_tty = ColorTTY()

MISC_CONFIG = {
    "architecture": "<value>",
    "max_iters": "<value>",
    "train_feed": "<value>",
    "eval_feed": "<value>",
    "test_feed": "<value>",
    "pretrain_weights": "<value>",
    "save_dir": "<value>",
    "weights": "<value>",
    "metric": "<value>",
    "log_smooth_window": 20,
    "snapshot_iter": 10000,
    "use_gpu": True,
}


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


def dump_config(module, minimal=False):
    args = module.schema.values()
    if minimal:
        args = [arg for arg in args if not arg.has_default()]
    return yaml.dump(
        {
            module.name: {
                arg.name: arg.default if arg.has_default() else "<value>"
                for arg in args
            }
        },
        default_flow_style=False,
        default_style='')


def list_modules(**kwargs):
    target_category = kwargs['category']
    module_schema = get_registered_modules()
    module_by_category = {}

    for schema in module_schema.values():
        category = schema.category
        if target_category is not None and schema.category != target_category:
            continue
        if category not in module_by_category:
            module_by_category[category] = [schema]
        else:
            module_by_category[category].append(schema)

    for cat, modules in module_by_category.items():
        print("Available modules in the category '{}':".format(cat))
        print("")
        max_len = max([len(mod.name) for mod in modules])
        for mod in modules:
            print(color_tty.green(mod.name.ljust(max_len)),
                  mod.doc.split('\n')[0])
        print("")


def help_module(**kwargs):
    schema = get_registered_modules()[kwargs['module']]

    doc = schema.doc is None and "Not documented" or "{}".format(schema.doc)
    func_args = {arg.name: arg.doc for arg in schema.schema.values()}
    max_len = max([len(k) for k in func_args.keys()])
    opts = "\n".join([
        "{} {}".format(color_tty.green(k.ljust(max_len)), v)
        for k, v in func_args.items()
    ])
    template = dump_config(schema)
    print("{}\n\n{}\n\n{}\n\n{}\n\n{}\n\n{}\n{}\n".format(
        color_tty.bold(color_tty.blue("MODULE DESCRIPTION:")),
        doc,
        color_tty.bold(color_tty.blue("MODULE OPTIONS:")),
        opts,
        color_tty.bold(color_tty.blue("CONFIGURATION TEMPLATE:")),
        template,
        color_tty.bold(color_tty.blue("COMMAND LINE OPTIONS:")), ))
    for arg in schema.schema.values():
        print("--opt {}.{}={}".format(schema.name, arg.name,
                                      dump_value(arg.default)
                                      if arg.has_default() else "<value>"))


def generate_config(**kwargs):
    minimal = kwargs['minimal']
    modules = kwargs['modules']
    module_schema = get_registered_modules()
    visited = []
    schema = []

    def walk(m):
        if m in visited:
            return
        s = module_schema[m]
        schema.append(s)
        visited.append(m)

    for mod in modules:
        walk(mod)

    # XXX try to be smart about when to add header,
    # if any "architecture" module, is included, head will be added as well
    if any([getattr(m, 'category', None) == 'architecture' for m in schema]):
        # XXX for ordered printing
        header = ""
        for k, v in MISC_CONFIG.items():
            header += yaml.dump(
                {
                    k: v
                }, default_flow_style=False, default_style='')
        print(header)

    for s in schema:
        print(dump_config(s, minimal))


# FIXME this is pretty hackish, maybe implement a custom YAML printer?
def analyze_config(**kwargs):
    config = load_config(kwargs['file'])
    modules = get_registered_modules()
    green = '___{}___'.format(color_tty.colors.index('green') + 31)

    styled = {}
    for key in config.keys():
        if not config[key]:  # empty schema
            continue

        if key not in modules and not hasattr(config[key], '__dict__'):
            styled[key] = config[key]
            continue
        elif key in modules:
            module = modules[key]
        else:
            type_name = type(config[key]).__name__
            if type_name in modules:
                module = modules[type_name].copy()
                module.update({
                    k: v
                    for k, v in config[key].__dict__.items()
                    if k in module.schema
                })
                key += " ({})".format(type_name)
        default = module.find_default_keys()
        missing = module.find_missing_keys()
        mismatch = module.find_mismatch_keys()
        extra = module.find_extra_keys()
        dep_missing = []
        for dep in module.inject:
            if isinstance(module[dep], str) and module[dep] != '<value>':
                if module[dep] not in modules:  # not a valid module
                    dep_missing.append(dep)
                else:
                    dep_mod = modules[module[dep]]
                    # empty dict but mandatory
                    if not dep_mod and dep_mod.mandatory():
                        dep_missing.append(dep)
        override = list(
            set(module.keys()) - set(default) - set(extra) - set(dep_missing))
        replacement = {}
        for name in set(override + default + extra + mismatch + missing):
            new_name = name
            if name in missing:
                value = "<missing>"
            else:
                value = module[name]

            if name in extra:
                value = dump_value(value) + " <extraneous>"
            elif name in mismatch:
                value = dump_value(value) + " <type mismatch>"
            elif name in dep_missing:
                value = dump_value(value) + " <module config missing>"
            elif name in override and value != '<missing>':
                mark = green
                new_name = mark + name
            replacement[new_name] = value
        styled[key] = replacement
    buffer = yaml.dump(styled, default_flow_style=False, default_style='')
    buffer = (re.sub(r"<missing>", r"[31m<missing>[0m", buffer))
    buffer = (re.sub(r"<extraneous>", r"[33m<extraneous>[0m", buffer))
    buffer = (re.sub(r"<type mismatch>", r"[31m<type mismatch>[0m", buffer))
    buffer = (re.sub(r"<module config missing>",
                     r"[31m<module config missing>[0m", buffer))
    buffer = re.sub(r"___(\d+)___(.*?):", r"[\1m\2[0m:", buffer)
    print(buffer)


if __name__ == '__main__':
    argv = sys.argv[1:]

    parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(help='Supported Commands')
    list_parser = subparsers.add_parser("list", help="list available modules")
    help_parser = subparsers.add_parser(
        "help", help="show detail options for module")
    generate_parser = subparsers.add_parser(
        "generate", help="generate configuration template")
    analyze_parser = subparsers.add_parser(
        "analyze", help="analyze configuration file")

    list_parser.set_defaults(func=list_modules)
    help_parser.set_defaults(func=help_module)
    generate_parser.set_defaults(func=generate_config)
    analyze_parser.set_defaults(func=analyze_config)

    list_group = list_parser.add_mutually_exclusive_group()
    list_group.add_argument(
        "-c",
        "--category",
        type=str,
        default=None,
        help="list modules for <category>")

    help_parser.add_argument(
        "module",
        help="module to show info for",
        choices=list(get_registered_modules().keys()))

    generate_parser.add_argument(
        "modules",
        nargs='+',
        help="include these module in generated configuration template",
        choices=list(get_registered_modules().keys()))
    generate_group = generate_parser.add_mutually_exclusive_group()
    generate_group.add_argument(
        "--minimal", action='store_true', help="only include required options")
    generate_group.add_argument(
        "--full",
        action='store_false',
        dest='minimal',
        help="include all options")

    analyze_parser.add_argument("file", help="configuration file to analyze")

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(argv)
    if hasattr(args, 'func'):
        args.func(**vars(args))
