English | [简体中文](CONFIG_cn.md)

# Config Pipline

## Introduction

PaddleDetection takes a rather principled approach to configuration management. We aim to automate the configuration workflow and to reduce configuration errors.


## Rationale

Presently, configuration in mainstream frameworks are usually dictionary based: the global config is simply a giant, loosely defined Python dictionary.

This approach is error prone, e.g., misspelled or displaced keys may lead to serious errors in training process, causing time loss and wasted resources.

To avoid the common pitfalls, with automation and static analysis in mind, we propose a configuration design that is user friendly, easy to maintain and extensible.


## Design

The design utilizes some of Python's reflection mechanism to extract configuration schematics from Python class definitions.

To be specific, it extracts information from class constructor arguments, including names, docstrings, default values, data types (if type hints are available).

This approach advocates modular and testable design, leading to a unified and extensible code base.


### API

Most of the functionality is exposed in `ppdet.core.workspace` module.

-   `register`: This decorator register a class as configurable module; it understands several special annotations in the class definition.
    -   `__category__`: For better organization, modules are classified into categories.
    -   `__inject__`: A list of constructor arguments, which are intended to take module instances as input, module instances will be created at runtime an injected. The corresponding configuration value can be a class name string, a serialized object, a config key pointing to a serialized object, or a dict (in which case the constructor needs to handle it, see example below).
    -   `__op__`: Shortcut for wrapping PaddlePaddle operators into a callable objects, together with `__append_doc__` (extracting docstring from target PaddlePaddle operator automatically), this can be a real time saver.
-   `serializable`: This decorator make a class directly serializable in yaml config file, by taking advantage of [pyyaml](https://pyyaml.org/wiki/PyYAMLDocumentation)'s serialization mechanism.
-   `create`: Constructs a module instance according to global configuration.
-   `load_config` and `merge_config`: Loading yaml file and merge config settings from command line.


### Example

Take the `RPNHead` module for example, it is composed of several PaddlePaddle operators. We first wrap those operators into classes, then pass in instances of these classes when instantiating the `RPNHead` module.

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

The corresponding(generated) YAML snippet is as follows, note this is the configuration in **FULL**, all the default values can be omitted. In case of the above example, all arguments have default value, meaning nothing is required in the config file.

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

Example snippet that make use of the `RPNHead` module.

```python
from ppdet.core.workspace import load_config, merge_config, create

load_config('some_config_file.yml')
merge_config(more_config_options_from_command_line)

rpn_head = create('RPNHead')
# ... code that use the created module!
```

Configuration file can also have serialized objects in it, denoted with `!`, for example

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


## Requirements

Two Python packages are used, both are optional.

-   [typeguard](https://github.com/agronholm/typeguard) is used for type checking in Python 3.
-   [docstring\_parser](https://github.com/rr-/docstring_parser) is needed for docstring parsing.

To install them, simply run:

```shell
pip install typeguard http://github.com/willthefrog/docstring_parser/tarball/master
```


## Tooling

A small utility (`tools/configure.py`) is included to simplify the configuration process, it provides 4 commands to walk users through the configuration process:

1.  `list`: List currently registered modules by category, one can also specify which category to list with the `--category` flag.
2.  `help`: Get help information for a module, including description, options, configuration template and example command line flags.
3.  `analyze`: Check configuration file for missing/extraneous options, options with mismatch type (if type hint is given) and missing dependencies, it also highlights user provided values (overridden default values).
4.  `generate`: Generate a configuration template for a given list of modules. By default it generates a complete configuration file, which can be quite verbose; if a `--minimal` flag is given, it generates a template that only contain non optional settings. For example, to generate a configuration for Faster R-CNN architecture with `ResNet` backbone and `FPN`, run:

    ```shell
    python tools/configure.py generate FasterRCNN ResNet RPNHead RoIAlign BBoxAssigner BBoxHead LearningRate OptimizerBuilder
    ```

    For a minimal version, run:

    ```shell
    python tools/configure.py generate --minimal FasterRCNN BBoxHead
    ```
