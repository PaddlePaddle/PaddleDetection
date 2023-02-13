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

import os

from ..base import BaseModel
from ..base.utils.path import get_cache_dir
from ..base.utils.arg import CLIArgument
from ..base.utils.misc import abspath

from .config import DetConfig


class DetModel(BaseModel):
    def train(self,
              dataset=None,
              batch_size=None,
              learning_rate=None,
              epochs_iters=None,
              device='gpu',
              resume_path=None,
              dy2st=False,
              amp=None,
              save_dir=None):
        # NOTE: We must use an absolute path here, 
        # so we can run the scripts either inside or outside the repo dir.
        if dataset is not None:
            dataset = abspath(dataset)
        if resume_path is not None:
            resume_path = abspath(resume_path)
        if save_dir is not None:
            save_dir = abspath(save_dir)

        # Update config and collect cli args
        config_file_path = self.model_info['config_path']
        config = DetConfig.build_from_file(config_file_path)
        cli_args = []

        config._update_dataset(dataset)
        if batch_size is not None:
            config.update({'TrainReader.batch_size': batch_size})
        if learning_rate is not None:
            config.update({'LearningRate.base_lr': learning_rate})
        if epochs_iters is not None:
            config.update({'epoch': epochs_iters})
        if device is not None:
            config._update_device(device)
        if resume_path is not None:
            resume_dir = os.path.dirname(resume_path)
            cli_args.append(CLIArgument('--resume', resume_dir))
        if dy2st:
            cli_args.append(CLIArgument('--to_static', '', ''))
        if amp is not None:
            # TODO: PaddleDetection
            cli_args.append(CLIArgument('--amp', '', ''))
        if save_dir is not None:
            config.update({'save_dir': save_dir})

        config.dump(config_file_path)
        self.runner.train(config_file_path, cli_args, device)

    def predict(self,
                input_path,
                weight_path,
                device='gpu',
                save_dir=None):
        input_path = abspath(input_path)
        weight_path = abspath(weight_path)
        if save_dir is not None:
            save_dir = abspath(save_dir)

        config_file_path = self.model_info['config_path']
        config = DetConfig.build_from_file(config_file_path)
        cli_args = []

        config.update({'weights': weight_path})
        config._update_device(device)
        if input_path is not None:
            cli_args.append(CLIArgument('--infer_img', input_path))
        if save_dir is not None:
            cli_args.append(CLIArgument('--output_dir', save_dir))

        config.dump(config_file_path)
        self.runner.predict(config_file_path, cli_args, device)

    def export(self, weight_path, save_dir, input_shape=None):
        weight_path = abspath(weight_path)
        save_dir = abspath(save_dir)

        config_file_path = self.model_info['config_path']
        config = DetConfig.build_from_file(config_file_path)
        cli_args = []

        config.update({'weights': weight_path})
        cli_args.append(CLIArgument('--output_dir', save_dir))
        if input_shape is not None:
            cli_args.append(CLIArgument('-o TestReader.inputs_def.image_shape', input_shape, '='))

        config.dump(config_file_path)
        self.runner.export(config_file_path, cli_args, None)

    def infer(self, model_dir, input_path, save_dir, device='gpu'):
        model_dir = abspath(model_dir)
        input_path = abspath(input_path)
        save_dir = abspath(save_dir)

        config_file_path = self.model_info['config_path']
        config = DetConfig.build_from_file(config_file_path)
        cli_args = []

        cli_args.append(CLIArgument('--model_dir', model_dir, '='))
        cli_args.append(CLIArgument('--image_file', input_path, '='))
        cli_args.append(CLIArgument('--output_dir', save_dir, '='))
        cli_args.append(CLIArgument('--device', device, '='))

        config.dump(config_file_path)
        self.runner.infer(config_file_path, cli_args, device)

    def compression(self,
                    dataset,
                    batch_size=None,
                    learning_rate=None,
                    epochs_iters=None,
                    device=None,
                    weight_path=None,
                    save_dir=None):
        dataset = abspath(dataset)
        if weight_path is not None:
            weight_path = abspath(weight_path)
        if save_dir is not None:
            save_dir = abspath(save_dir)

        # Update YAML config file
        config_file_path = self.model_info['auto_compression_config_path']
        config = DetConfig.build_from_file(config_file_path)
        config._update_dataset_config(dataset)
        config_file_path = self.config_file_path
        config.dump(config_file_path)

        # Parse CLI arguments
        cli_args = []
        if batch_size is not None:
            cli_args.append(CLIArgument('--batch_size', batch_size))
        if learning_rate is not None:
            cli_args.append(CLIArgument('--learning_rate', learning_rate))
        if epochs_iters is not None:
            cli_args.append(CLIArgument('--iters', epochs_iters))
        if weight_path is not None:
            cli_args.append(CLIArgument('--model_path', weight_path))
        if save_dir is not None:
            cli_args.append(CLIArgument('--save_dir', save_dir))

        self.runner.compression(config_file_path, cli_args, device)
