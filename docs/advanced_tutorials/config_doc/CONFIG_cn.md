# 配置模块设计与介绍

## 简介

为了使配置过程更加自动化并减少配置错误，PaddleDetection的配置管理采取了较为严谨的设计。


## 设计思想

目前主流框架全局配置基本是一个Python dict，这种设计对配置的检查并不严格，拼写错误或者遗漏的配置项往往会造成训练过程中的严重错误，进而造成时间及资源的浪费。为了避免这些陷阱，从自动化和静态分析的原则出发，PaddleDetection采用了一种用户友好、 易于维护和扩展的配置设计。


## 基本设计

利用Python的反射机制，PaddleDection的配置系统从Python类的构造函数抽取多种信息 - 如参数名、初始值、参数注释、数据类型（如果给出type hint）- 来作为配置规则。 这种设计便于设计的模块化，提升可测试性及扩展性。


### API

配置系统的大多数功能由 `ppdet.core.workspace` 模块提供

-   `register`: 装饰器，将类注册为可配置模块；能够识别类定义中的一些特殊标注。
    -   `__category__`: 为便于组织，模块可以分为不同类别。
    -   `__inject__`: 如果模块由多个子模块组成，可以这些子模块实例作为构造函数的参数注入。对应的默认值及配置项可以是类名字符串，yaml序列化的对象，指向序列化对象的配置键值或者Python dict（构造函数需要对其作出处理，参见下面的例子）。
    -   `__op__`: 配合 `__append_doc__` （抽取目标OP的 注释）使用，可以方便快速的封装PaddlePaddle底层OP。
-   `serializable`: 装饰器，利用 [pyyaml](https://pyyaml.org/wiki/PyYAMLDocumentation) 的序列化机制，可以直接将一个类实例序列化及反序列化。
-   `create`: 根据全局配置构造一个模块实例。
-   `load_config` and `merge_config`: 加载yaml文件，合并命令行提供的配置项。


### 示例

以 `RPNHead` 模块为例，该模块包含多个PaddlePaddle OP，先将这些OP封装成类，并将其实例在构造 `RPNHead` 时注入。

```python
# excerpt from `ppdet/modeling/ops.py`
from ppdet.core.workspace import register, serializable

# ... more operators

@register
@serializable
class GenerateProposals(object):
    # NOTE this class simply wraps a PaddlePaddle operator
    __op__ = fluid.layers.generate_proposals
    # NOTE docstring for args are extracted from PaddlePaddle OP
    __append_doc__ = True

    def __init__(self,
                 pre_nms_top_n=6000,
                 post_nms_top_n=1000,
                 nms_thresh=.5,
                 min_size=.1,
                 eta=1.):
        super(GenerateProposals, self).__init__()
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = min_size
        self.eta = eta

# ... more operators

# excerpt from `ppdet/modeling/anchor_heads/rpn_head.py`
from ppdet.core.workspace import register
from ppdet.modeling.ops import AnchorGenerator, RPNTargetAssign, GenerateProposals

@register
class RPNHead(object):
    """
    RPN Head

    Args:
        anchor_generator (object): `AnchorGenerator` instance
        rpn_target_assign (object): `RPNTargetAssign` instance
        train_proposal (object): `GenerateProposals` instance for training
        test_proposal (object): `GenerateProposals` instance for testing
    """
    __inject__ = [
        'anchor_generator', 'rpn_target_assign', 'train_proposal',
        'test_proposal'
    ]

    def __init__(self,
                 anchor_generator=AnchorGenerator().__dict__,
                 rpn_target_assign=RPNTargetAssign().__dict__,
                 train_proposal=GenerateProposals(12000, 2000).__dict__,
                 test_proposal=GenerateProposals().__dict__):
        super(RPNHead, self).__init__()
        self.anchor_generator = anchor_generator
        self.rpn_target_assign = rpn_target_assign
        self.train_proposal = train_proposal
        self.test_proposal = test_proposal
        if isinstance(anchor_generator, dict):
            self.anchor_generator = AnchorGenerator(**anchor_generator)
        if isinstance(rpn_target_assign, dict):
            self.rpn_target_assign = RPNTargetAssign(**rpn_target_assign)
        if isinstance(train_proposal, dict):
            self.train_proposal = GenerateProposals(**train_proposal)
        if isinstance(test_proposal, dict):
            self.test_proposal = GenerateProposals(**test_proposal)
```

对应的yaml配置如下，请注意这里给出的是 **完整** 配置，其中所有默认值配置项都可以省略。上面的例子中的模块所有的构造函数参数都提供了默认值，因此配置文件中可以完全略过其配置。

```yaml
RPNHead:
  test_proposal:
    eta: 1.0
    min_size: 0.1
    nms_thresh: 0.5
    post_nms_top_n: 1000
    pre_nms_top_n: 6000
  train_proposal:
    eta: 1.0
    min_size: 0.1
    nms_thresh: 0.5
    post_nms_top_n: 2000
    pre_nms_top_n: 12000
  anchor_generator:
    # ...
  rpn_target_assign:
    # ...
```

`RPNHead` 模块实际使用代码示例。

```python
from ppdet.core.workspace import load_config, merge_config, create

load_config('some_config_file.yml')
merge_config(more_config_options_from_command_line)

rpn_head = create('RPNHead')
# ... code that use the created module!
```

配置文件用可以直接序列化模块实例，用 `!` 标示，如

```yaml
LearningRate:
  base_lr: 0.01
  schedulers:
  - !PiecewiseDecay
    gamma: 0.1
    milestones: [60000, 80000]
  - !LinearWarmup
    start_factor: 0.3333333333333333
    steps: 500
```


## 安装依赖

配置系统用到两个Python包，均为可选安装。

-   [typeguard](https://github.com/agronholm/typeguard) 在Python 3中用来进行数据类型验证。
-   [docstring\_parser](https://github.com/rr-/docstring_parser) 用来解析注释。

如需安装，运行下面命令即可。

```shell
pip install typeguard http://github.com/willthefrog/docstring_parser/tarball/master
```


## 相关工具

为了方便用户配置，PaddleDection提供了一个工具 (`tools/configure.py`)， 共支持四个子命令：

1.  `list`: 列出当前已注册的模块，如需列出具体类别的模块，可以使用 `--category` 指定。
2.  `help`: 显示指定模块的帮助信息，如描述，配置项，配置文件模板及命令行示例。
3.  `analyze`: 检查配置文件中的缺少或者多余的配置项以及依赖缺失，如果给出type hint， 还可以检查配置项中错误的数据类型。非默认配置也会高亮显示。
4.  `generate`: 根据给出的模块列表生成配置文件，默认生成完整配置，如果指定 `--minimal` ，生成最小配置，即省略所有默认配置项。例如，执行下列命令可以生成Faster R-CNN (`ResNet` backbone + `FPN`) 架构的配置文件:

    ```shell
    python tools/configure.py generate FasterRCNN ResNet RPNHead RoIAlign BBoxAssigner BBoxHead LearningRate OptimizerBuilder
    ```

    如需最小配置，运行：

    ```shell
    python tools/configure.py generate --minimal FasterRCNN BBoxHead
    ```
