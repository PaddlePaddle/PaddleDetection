# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn

from paddle.utils.download import get_weights_path_from_url
from ppdet.core.workspace import register, serializable
from ..shape_spec import ShapeSpec

__all__ = ['CLRResNet']

model_urls = {
    'resnet18':
    'https://x2paddle.bj.bcebos.com/vision/models/resnet18-pt.pdparams',
    'resnet34':
    'https://x2paddle.bj.bcebos.com/vision/models/resnet34-pt.pdparams',
    'resnet50':
    'https://x2paddle.bj.bcebos.com/vision/models/resnet50-pt.pdparams',
    'resnet101':
    'https://x2paddle.bj.bcebos.com/vision/models/resnet101-pt.pdparams',
    'resnet152':
    'https://x2paddle.bj.bcebos.com/vision/models/resnet152-pt.pdparams',
    'resnext50_32x4d':
    'https://x2paddle.bj.bcebos.com/vision/models/resnext50_32x4d-pt.pdparams',
    'resnext101_32x8d':
    'https://x2paddle.bj.bcebos.com/vision/models/resnext101_32x8d-pt.pdparams',
    'wide_resnet50_2':
    'https://x2paddle.bj.bcebos.com/vision/models/wide_resnet50_2-pt.pdparams',
    'wide_resnet101_2':
    'https://x2paddle.bj.bcebos.com/vision/models/wide_resnet101_2-pt.pdparams',
}


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D

        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")

        self.conv1 = nn.Conv2D(
            inplanes, planes, 3, padding=1, stride=stride, bias_attr=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(planes, planes, 3, padding=1, bias_attr=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckBlock(nn.Layer):

    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(BottleneckBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = nn.Conv2D(inplanes, width, 1, bias_attr=False)
        self.bn1 = norm_layer(width)

        self.conv2 = nn.Conv2D(
            width,
            width,
            3,
            padding=dilation,
            stride=stride,
            groups=groups,
            dilation=dilation,
            bias_attr=False)
        self.bn2 = norm_layer(width)

        self.conv3 = nn.Conv2D(
            width, planes * self.expansion, 1, bias_attr=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Layer):
    """ResNet model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        Block (BasicBlock|BottleneckBlock): Block module of model.
        depth (int, optional): Layers of ResNet, Default: 50.
        width (int, optional): Base width per convolution group for each convolution block, Default: 64.
        num_classes (int, optional): Output dim of last fc layer. If num_classes <= 0, last fc layer 
                            will not be defined. Default: 1000.
        with_pool (bool, optional): Use pool before the last fc layer or not. Default: True.
        groups (int, optional): Number of groups for each convolution block, Default: 1.
    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of ResNet model.
    Examples:
        .. code-block:: python
            import paddle
            from paddle.vision.models import ResNet
            from paddle.vision.models.resnet import BottleneckBlock, BasicBlock
            # build ResNet with 18 layers
            resnet18 = ResNet(BasicBlock, 18)
            # build ResNet with 50 layers
            resnet50 = ResNet(BottleneckBlock, 50)
            # build Wide ResNet model
            wide_resnet50_2 = ResNet(BottleneckBlock, 50, width=64*2)
            # build ResNeXt model
            resnext50_32x4d = ResNet(BottleneckBlock, 50, width=4, groups=32)
            x = paddle.rand([1, 3, 224, 224])
            out = resnet18(x)
            print(out.shape)
            # [1, 1000]
    """

    def __init__(self, block, depth=50, width=64, with_pool=True, groups=1):
        super(ResNet, self).__init__()
        layer_cfg = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3]
        }

        layers = layer_cfg[depth]
        self.groups = groups
        self.base_width = width
        self.with_pool = with_pool
        self._norm_layer = nn.BatchNorm2D

        self.inplanes = 64
        self.dilation = 1

        self.conv1 = nn.Conv2D(
            3,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias_attr=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if with_pool:
            self.avgpool = nn.AdaptiveAvgPool2D((1, 1))

        ch_out_list = [64, 128, 256, 512]
        block = BottleneckBlock if depth >= 50 else BasicBlock

        self._out_channels = [block.expansion * v for v in ch_out_list]
        self._out_strides = [4, 8, 16, 32]
        self.return_idx = [0, 1, 2, 3]

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(
                    self.inplanes,
                    planes * block.expansion,
                    1,
                    stride=stride,
                    bias_attr=False),
                norm_layer(planes * block.expansion), )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    norm_layer=norm_layer))

        return nn.Sequential(*layers)

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self._out_channels[i], stride=self._out_strides[i])
            for i in self.return_idx
        ]

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        out_layers = []
        x = self.layer1(x)
        out_layers.append(x)
        x = self.layer2(x)
        out_layers.append(x)
        x = self.layer3(x)
        out_layers.append(x)
        x = self.layer4(x)
        out_layers.append(x)

        if self.with_pool:
            x = self.avgpool(x)

        return out_layers


@register
@serializable
class CLRResNet(nn.Layer):
    def __init__(self,
                 resnet='resnet18',
                 pretrained=True,
                 out_conv=False,
                 fea_stride=8,
                 out_channel=128,
                 in_channels=[64, 128, 256, 512],
                 cfg=None):
        super(CLRResNet, self).__init__()
        self.cfg = cfg
        self.in_channels = in_channels

        self.model = eval(resnet)(pretrained=pretrained)
        self.out = None
        if out_conv:
            out_channel = 512
            for chan in reversed(self.in_channels):
                if chan < 0: continue
                out_channel = chan
                break
            self.out = nn.Conv2D(
                out_channel * self.model.expansion,
                cfg.featuremap_out_channel,
                kernel_size=1,
                bias_attr=False)

    @property
    def out_shape(self):
        return self.model.out_shape

    def forward(self, x):
        x = self.model(x)
        if self.out:
            x[-1] = self.out(x[-1])
        return x


def _resnet(arch, Block, depth, pretrained, **kwargs):
    model = ResNet(Block, depth, **kwargs)
    if pretrained:
        assert arch in model_urls, "{} model do not have a pretrained model now, you should set pretrained=False".format(
            arch)
        weight_path = get_weights_path_from_url(model_urls[arch])

        param = paddle.load(weight_path)
        model.set_dict(param)

    return model


def resnet18(pretrained=False, **kwargs):
    """ResNet 18-layer model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`ResNet <api_paddle_vision_ResNet>`.
    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of ResNet 18-layer model.
    Examples:
        .. code-block:: python
            import paddle
            from paddle.vision.models import resnet18
            # build model
            model = resnet18()
            # build model and load imagenet pretrained weight
            # model = resnet18(pretrained=True)
            x = paddle.rand([1, 3, 224, 224])
            out = model(x)
            print(out.shape)
            # [1, 1000]
    """
    return _resnet('resnet18', BasicBlock, 18, pretrained, **kwargs)


def resnet34(pretrained=False, **kwargs):
    """ResNet 34-layer model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`ResNet <api_paddle_vision_ResNet>`.
    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of ResNet 34-layer model.
    Examples:
        .. code-block:: python
            import paddle
            from paddle.vision.models import resnet34
            # build model
            model = resnet34()
            # build model and load imagenet pretrained weight
            # model = resnet34(pretrained=True)
            x = paddle.rand([1, 3, 224, 224])
            out = model(x)
            print(out.shape)
            # [1, 1000]
    """
    return _resnet('resnet34', BasicBlock, 34, pretrained, **kwargs)


def resnet50(pretrained=False, **kwargs):
    """ResNet 50-layer model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`ResNet <api_paddle_vision_ResNet>`.
    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of ResNet 50-layer model.
    Examples:
        .. code-block:: python
            import paddle
            from paddle.vision.models import resnet50
            # build model
            model = resnet50()
            # build model and load imagenet pretrained weight
            # model = resnet50(pretrained=True)
            x = paddle.rand([1, 3, 224, 224])
            out = model(x)
            print(out.shape)
            # [1, 1000]
    """
    return _resnet('resnet50', BottleneckBlock, 50, pretrained, **kwargs)


def resnet101(pretrained=False, **kwargs):
    """ResNet 101-layer model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`ResNet <api_paddle_vision_ResNet>`.
    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of ResNet 101-layer.
    Examples:
        .. code-block:: python
            import paddle
            from paddle.vision.models import resnet101
            # build model
            model = resnet101()
            # build model and load imagenet pretrained weight
            # model = resnet101(pretrained=True)
            x = paddle.rand([1, 3, 224, 224])
            out = model(x)
            print(out.shape)
            # [1, 1000]
    """
    return _resnet('resnet101', BottleneckBlock, 101, pretrained, **kwargs)


def resnet152(pretrained=False, **kwargs):
    """ResNet 152-layer model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`ResNet <api_paddle_vision_ResNet>`.
    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of ResNet 152-layer model.
    Examples:
        .. code-block:: python
            import paddle
            from paddle.vision.models import resnet152
            # build model
            model = resnet152()
            # build model and load imagenet pretrained weight
            # model = resnet152(pretrained=True)
            x = paddle.rand([1, 3, 224, 224])
            out = model(x)
            print(out.shape)
            # [1, 1000]
    """
    return _resnet('resnet152', BottleneckBlock, 152, pretrained, **kwargs)


def resnext50_32x4d(pretrained=False, **kwargs):
    """ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformations for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    
    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`ResNet <api_paddle_vision_ResNet>`.
    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of ResNeXt-50 32x4d model.
    Examples:
        .. code-block:: python
            import paddle
            from paddle.vision.models import resnext50_32x4d
            # build model
            model = resnext50_32x4d()
            # build model and load imagenet pretrained weight
            # model = resnext50_32x4d(pretrained=True)
            x = paddle.rand([1, 3, 224, 224])
            out = model(x)
            print(out.shape)
            # [1, 1000]
    """
    kwargs['groups'] = 32
    kwargs['width'] = 4
    return _resnet('resnext50_32x4d', BottleneckBlock, 50, pretrained, **kwargs)


def resnext50_64x4d(pretrained=False, **kwargs):
    """ResNeXt-50 64x4d model from
    `"Aggregated Residual Transformations for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    
    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`ResNet <api_paddle_vision_ResNet>`.
    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of ResNeXt-50 64x4d model.
    Examples:
        .. code-block:: python
            import paddle
            from paddle.vision.models import resnext50_64x4d
            # build model
            model = resnext50_64x4d()
            # build model and load imagenet pretrained weight
            # model = resnext50_64x4d(pretrained=True)
            x = paddle.rand([1, 3, 224, 224])
            out = model(x)
            print(out.shape)
            # [1, 1000]
    """
    kwargs['groups'] = 64
    kwargs['width'] = 4
    return _resnet('resnext50_64x4d', BottleneckBlock, 50, pretrained, **kwargs)


def resnext101_32x4d(pretrained=False, **kwargs):
    """ResNeXt-101 32x4d model from
    `"Aggregated Residual Transformations for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    
    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`ResNet <api_paddle_vision_ResNet>`.
    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of ResNeXt-101 32x4d model.
    Examples:
        .. code-block:: python
            import paddle
            from paddle.vision.models import resnext101_32x4d
            # build model
            model = resnext101_32x4d()
            # build model and load imagenet pretrained weight
            # model = resnext101_32x4d(pretrained=True)
            x = paddle.rand([1, 3, 224, 224])
            out = model(x)
            print(out.shape)
            # [1, 1000]
    """
    kwargs['groups'] = 32
    kwargs['width'] = 4
    return _resnet('resnext101_32x4d', BottleneckBlock, 101, pretrained,
                   **kwargs)


def resnext101_64x4d(pretrained=False, **kwargs):
    """ResNeXt-101 64x4d model from
    `"Aggregated Residual Transformations for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    
    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`ResNet <api_paddle_vision_ResNet>`.
    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of ResNeXt-101 64x4d model.
    Examples:
        .. code-block:: python
            import paddle
            from paddle.vision.models import resnext101_64x4d
            # build model
            model = resnext101_64x4d()
            # build model and load imagenet pretrained weight
            # model = resnext101_64x4d(pretrained=True)
            x = paddle.rand([1, 3, 224, 224])
            out = model(x)
            print(out.shape)
            # [1, 1000]
    """
    kwargs['groups'] = 64
    kwargs['width'] = 4
    return _resnet('resnext101_64x4d', BottleneckBlock, 101, pretrained,
                   **kwargs)


def resnext152_32x4d(pretrained=False, **kwargs):
    """ResNeXt-152 32x4d model from
    `"Aggregated Residual Transformations for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    
    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`ResNet <api_paddle_vision_ResNet>`.
    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of ResNeXt-152 32x4d model.
    Examples:
        .. code-block:: python
            import paddle
            from paddle.vision.models import resnext152_32x4d
            # build model
            model = resnext152_32x4d()
            # build model and load imagenet pretrained weight
            # model = resnext152_32x4d(pretrained=True)
            x = paddle.rand([1, 3, 224, 224])
            out = model(x)
            print(out.shape)
            # [1, 1000]
    """
    kwargs['groups'] = 32
    kwargs['width'] = 4
    return _resnet('resnext152_32x4d', BottleneckBlock, 152, pretrained,
                   **kwargs)


def resnext152_64x4d(pretrained=False, **kwargs):
    """ResNeXt-152 64x4d model from
    `"Aggregated Residual Transformations for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    
    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`ResNet <api_paddle_vision_ResNet>`.
    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of ResNeXt-152 64x4d model.
    Examples:
        .. code-block:: python
            import paddle
            from paddle.vision.models import resnext152_64x4d
            # build model
            model = resnext152_64x4d()
            # build model and load imagenet pretrained weight
            # model = resnext152_64x4d(pretrained=True)
            x = paddle.rand([1, 3, 224, 224])
            out = model(x)
            print(out.shape)
            # [1, 1000]
    """
    kwargs['groups'] = 64
    kwargs['width'] = 4
    return _resnet('resnext152_64x4d', BottleneckBlock, 152, pretrained,
                   **kwargs)


def wide_resnet50_2(pretrained=False, **kwargs):
    """Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`ResNet <api_paddle_vision_ResNet>`.
    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of Wide ResNet-50-2 model.
    Examples:
        .. code-block:: python
            import paddle
            from paddle.vision.models import wide_resnet50_2
            # build model
            model = wide_resnet50_2()
            # build model and load imagenet pretrained weight
            # model = wide_resnet50_2(pretrained=True)
            x = paddle.rand([1, 3, 224, 224])
            out = model(x)
            print(out.shape)
            # [1, 1000]
    """
    kwargs['width'] = 64 * 2
    return _resnet('wide_resnet50_2', BottleneckBlock, 50, pretrained, **kwargs)


def wide_resnet101_2(pretrained=False, **kwargs):
    """Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    Args:
        pretrained (bool, optional): Whether to load pre-trained weights. If True, returns a model pre-trained
                            on ImageNet. Default: False.
        **kwargs (optional): Additional keyword arguments. For details, please refer to :ref:`ResNet <api_paddle_vision_ResNet>`.
    Returns:
        :ref:`api_paddle_nn_Layer`. An instance of Wide ResNet-101-2 model.
    Examples:
        .. code-block:: python
            import paddle
            from paddle.vision.models import wide_resnet101_2
            # build model
            model = wide_resnet101_2()
            # build model and load imagenet pretrained weight
            # model = wide_resnet101_2(pretrained=True)
            x = paddle.rand([1, 3, 224, 224])
            out = model(x)
            print(out.shape)
            # [1, 1000]
    """
    kwargs['width'] = 64 * 2
    return _resnet('wide_resnet101_2', BottleneckBlock, 101, pretrained,
                   **kwargs)
