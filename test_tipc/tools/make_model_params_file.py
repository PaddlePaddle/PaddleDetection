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
import sys
import json
import copy


def json_load(file_path):
    with open(file_path) as f:
        out_list = json.load(f)
    return out_list


def json_save(out_list, file_path):
    with open(file_path, 'w') as f:
        json.dump(out_list, f, indent=4)


def txt_load(file_path):
    with open(file_path) as f:
        out_list = f.readlines()
    return out_list


def txt_save(out_list, file_path):
    with open(file_path, 'w') as f:
        f.writelines(out_list)


def replace_keyword_(out_list, key, value):
    for i, line in enumerate(out_list):
        out_list[i] = line.replace(key, value)


def replace_one_line_(out_list, key, new_str):
    for i, line in enumerate(out_list):
        temp = line.split(':')[0]
        if key == temp:
            out_list[i] = temp + ':' + new_str + '\n'
            break


def replace_one_line_keyword_(out_list, key, old_str, new_str):
    for i, line in enumerate(out_list):
        temp = line.split(':')[0]
        if key == temp:
            out_list[i] = line.replace(old_str, new_str)
            break


def save_model_params(template_config,
                      configs_dir,
                      model,
                      postfix="train_infer_python.txt",
                      slim=None,
                      overwrite=False):
    out_config = copy.deepcopy(template_config)
    model = copy.deepcopy(model)
    # replace _template_model
    _template_model = model[2].get("config_dir", model[0])
    replace_keyword_(out_config, "_template_model", _template_model)
    # replace pretrain_weights or weights
    if "weights_dir" in model[2].keys():
        _template_weights = model[2]["weights_dir"] + model[1]
        replace_one_line_keyword_(out_config, "pretrain_weights",
                                  "_template_name", _template_weights)
        replace_one_line_keyword_(out_config, "weights", "_template_name",
                                  _template_weights)
    # replace _template_name
    replace_keyword_(out_config, "_template_name", model[1])
    # replace _template_epoch
    replace_keyword_(out_config, f"_template_epoch", str(model[2]['epoch']))
    # replace _template_batch_size
    replace_keyword_(out_config, f"_template_batch_size",
                     str(model[2]['batch_size']))
    # replace _template_slim
    for slim_mode, slim_yml in model[2]["slim"].items():
        replace_keyword_(out_config, f"_template_{slim_mode}", slim_yml)
    if slim is not None:
        # replace model_name
        model[1] = model[1] + '_' + slim.split('_')[0].upper()
        replace_one_line_(out_config, 'model_name', model[1])
        # replace trainer, infer_mode, infer_quant, run_mode
        if slim == 'kl_quant':
            replace_one_line_(out_config, 'trainer', 'null')
        else:
            replace_one_line_(out_config, 'trainer', f'{slim}_train')
        replace_one_line_(out_config, 'infer_mode', slim)
        if slim == 'pact' or slim == 'kl_quant':
            replace_one_line_(out_config, 'infer_quant', 'True')
            replace_one_line_(out_config, '--run_mode', 'fluid|trt_int8')

    if os.path.exists(os.path.join(configs_dir,
                                   f"{model[1]}_{postfix}")) and not overwrite:
        return out_config

    txt_save(out_config, os.path.join(configs_dir, f"{model[1]}_{postfix}"))
    print(f'save {model[1]}_{postfix} done. ')
    return out_config


def generate_tipc_configs(
        configs_dir,
        model_configs,
        template_dir,
        template_file_name=(
            "train_infer_python.txt",
            "train_linux_gpu_normal_amp_infer_python_linux_gpu_cpu.txt",
            "model_linux_gpu_normal_normal_infer_cpp_linux_gpu_cpu.txt")):
    count_model = 0
    base_template = txt_load(os.path.join(template_dir, template_file_name[0]))
    base_amp_template = txt_load(
        os.path.join(template_dir, template_file_name[1]))
    cpp_template = txt_load(os.path.join(template_dir, template_file_name[2]))
    for model_group, model_list in model_configs.items():
        count_model += len(model_list)
        print(f"build {model_group}...")
        model_config_dir = os.path.join(configs_dir, model_group)
        if not os.path.exists(model_config_dir):
            os.makedirs(model_config_dir)
        for model_name, model_info in model_list.items():
            print(f"build {model_name} configs...")
            model_ = [model_group, model_name, model_info]
            # save base python params
            save_model_params(base_template, model_config_dir, model_)
            # save base amp python params
            save_model_params(
                base_amp_template,
                model_config_dir,
                model_,
                postfix=template_file_name[1])
            # save cpp params
            save_model_params(
                cpp_template,
                model_config_dir,
                model_,
                postfix=template_file_name[2])

            # save slim python params
            count_model += len(model_info["slim"])
            for slim_mode, slim_yml in model_info["slim"].items():
                # save base python params
                save_model_params(
                    base_template,
                    model_config_dir,
                    model_,
                    postfix=template_file_name[0],
                    slim=slim_mode)
                # save slim cpp params
                save_model_params(
                    cpp_template,
                    model_config_dir,
                    model_,
                    postfix=template_file_name[2],
                    slim=slim_mode)

    return count_model


if __name__ == '__main__':
    utils_path = os.path.split(os.path.realpath(sys.argv[0]))[0]
    tipc_path = os.path.split(utils_path)[0]
    model_configs = json_load(os.path.join(utils_path, "model_configs.json"))
    configs_dir = os.path.join(tipc_path, "configs")
    if not os.path.exists(configs_dir):
        os.makedirs(configs_dir)

    count_model = generate_tipc_configs(configs_dir, model_configs,
                                        os.path.join(utils_path, "template"))
    print(f'num_total_models: {count_model}')
    print('Done!')
