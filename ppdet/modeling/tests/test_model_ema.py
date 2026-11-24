# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import math
import unittest

import numpy as np
import paddle

from ppdet.optimizer.ema import ModelEMA


class TestModelEMA(unittest.TestCase):
    def test_matches_weighted_average_after_resume(self):
        for mode in ('normal', 'threshold', 'exponential'):
            with self.subTest(mode=mode):
                model = paddle.nn.Linear(2, 1)
                model.register_buffer('scalar_buffer', paddle.to_tensor(0, dtype='int64'))
                ema = ModelEMA(model, decay=0.9, gamma=2, ema_decay_type=mode,
                               ema_black_list=['bias'])
                checkpoint = {'weight': paddle.full([2, 1], 3.0, dtype='float32')}
                ema.resume(checkpoint, step=10)

                expected = np.full((2, 1), 3.0, dtype=np.float32)
                expected_scalar = 0.0
                for step, value in enumerate((8, -2, 5), start=10):
                    if mode == 'threshold':
                        decay = min(0.9, (1 + step) / (10 + step))
                    elif mode == 'exponential':
                        decay = 0.9 * (1 - math.exp(-(step + 1) / 2))
                    else:
                        decay = 0.9
                    model.weight.set_value(paddle.full([2, 1], value, dtype='float32'))
                    model.scalar_buffer.set_value(paddle.to_tensor(value, dtype='int64'))
                    ema.update()
                    expected = decay * expected + (1 - decay) * value
                    expected_scalar = decay * expected_scalar + (1 - decay) * value
                    np.testing.assert_allclose(ema.state_dict['weight'].numpy(), expected,
                                               rtol=1e-6, atol=1e-6)
                    np.testing.assert_allclose(ema.state_dict['scalar_buffer'].numpy(),
                                               expected_scalar, rtol=1e-6, atol=1e-6)
                np.testing.assert_array_equal(checkpoint['weight'].numpy(), [[3.0], [3.0]])
                np.testing.assert_array_equal(ema.apply()['bias'].numpy(), model.bias.numpy())

    def test_exponential_apply_snapshot_survives_update(self):
        model = paddle.nn.Linear(2, 1)
        ema = ModelEMA(model, ema_decay_type='exponential', decay=0.9, gamma=1)
        model.weight.set_value(paddle.full([2, 1], 2.0))
        ema.update()
        snapshot = ema.apply()['weight']
        expected = snapshot.numpy().copy()
        model.weight.set_value(paddle.full([2, 1], 12.0))
        ema.update()
        np.testing.assert_array_equal(snapshot.numpy(), expected)
        self.assertGreater(float(ema.apply()['weight'].mean()), float(expected.mean()))
