import numpy as np
import paddle
import logging
from paddleslim.common import get_logger

__all__ = [
    "make_unstructured_pruner", "UnstructuredPruner", "UnstructuredPrunerGMP"
]

_logger = get_logger(__name__, level=logging.INFO)

NORMS_ALL = [
    'BatchNorm', 'GroupNorm', 'LayerNorm', 'SpectralNorm', 'BatchNorm1D',
    'BatchNorm2D', 'BatchNorm3D', 'InstanceNorm1D', 'InstanceNorm2D',
    'InstanceNorm3D', 'SyncBatchNorm', 'LocalResponseNorm'
]


class UnstructuredPruner():
    """
    The unstructure pruner.
    Args:
      - model(Paddle.nn.Layer): The model to be pruned.
      - mode(str): Pruning mode, must be selected from 'ratio' and 'threshold'.
      - threshold(float): The parameters whose absolute values are smaller than the THRESHOLD will be zeros. Default: 0.01
      - ratio(float): The parameters whose absolute values are in the smaller part decided by the ratio will be zeros. Default: 0.55
      - skip_params_type(str): The argument to control which type of ops will be ignored. Currently we only support None or exclude_conv1x1 as input. It acts as a straightforward call to conv1x1 pruning.  Default: None
      - skip_params_func(function): The function used to select the parameters which should be skipped when performing pruning. Default: normalization-related params. 
      - configs(dict): The dictionary contains all the configs for pruner defined in its subclass. Here in this base class, it takes no effect. Default: None
    """

    def __init__(self,
                 model,
                 mode,
                 threshold=0.01,
                 ratio=0.55,
                 skip_params_type=None,
                 skip_params_func=None,
                 configs=None):
        assert mode in ('ratio', 'threshold'
                        ), "mode must be selected from 'ratio' and 'threshold'"
        assert skip_params_type is None or skip_params_type == 'exclude_conv1x1', "skip_params_type only supports None or exclude_conv1x1 for now."
        self.model = model
        self.mode = mode
        self.threshold = threshold
        self.ratio = ratio

        # Prority: passed-in skip_params_func > skip_params_type (exclude_conv1x1) > built-in _get_skip_params
        if skip_params_func is not None:
            skip_params_func = skip_params_func
        elif skip_params_type == 'exclude_conv1x1':
            skip_params_func = self._get_skip_params_conv1x1
        elif skip_params_func is None:
            skip_params_func = self._get_skip_params

        self.skip_params = skip_params_func(model)
        self._apply_masks()

    def mask_parameters(self, param, mask):
        """
        Update masks and parameters. It is executed to each layer before each iteration.
        User can overwrite this function in subclass to implememt different pruning stragies.
        Args:
          - parameters(list<Tensor>): The parameters to be pruned.
          - masks(list<Tensor>): The masks used to keep zero values in parameters.
        """
        param_tmp = param * mask
        param_tmp.stop_gradient = True
        paddle.assign(param_tmp, output=param)

    def _apply_masks(self):
        self.masks = {}
        for name, sub_layer in self.model.named_sublayers():
            for param in sub_layer.parameters(include_sublayers=False):
                tmp_array = np.ones(param.shape, dtype=np.float32)
                mask_name = "_".join([param.name.replace(".", "_"), "mask"])
                if mask_name not in sub_layer._buffers:
                    sub_layer.register_buffer(mask_name,
                                              paddle.to_tensor(tmp_array))
                self.masks[param.name] = sub_layer._buffers[mask_name]
        for name, sub_layer in self.model.named_sublayers():
            sub_layer.register_forward_pre_hook(self._forward_pre_hook)

    def update_threshold(self):
        '''
        Update the threshold after each optimization step.
        User should overwrite this method togther with self.mask_parameters()
        '''
        params_flatten = []
        for name, sub_layer in self.model.named_sublayers():
            if not self._should_prune_layer(sub_layer):
                continue
            for param in sub_layer.parameters(include_sublayers=False):
                t_param = param.value().get_tensor()
                v_param = np.array(t_param)
                params_flatten.append(v_param.flatten())
        params_flatten = np.concatenate(params_flatten, axis=0)
        total_length = params_flatten.size
        self.threshold = np.sort(np.abs(params_flatten))[max(
            0, round(self.ratio * total_length) - 1)].item()

    def _update_masks(self):
        for name, sub_layer in self.model.named_sublayers():
            if not self._should_prune_layer(sub_layer): continue
            for param in sub_layer.parameters(include_sublayers=False):
                mask = self.masks.get(param.name)
                bool_tmp = (paddle.abs(param) >= self.threshold)
                paddle.assign(bool_tmp, output=mask)

    def summarize_weights(self, model, ratio=0.1):
        """
        The function is used to get the weights corresponding to a given ratio
        when you are uncertain about the threshold in __init__() function above.
        For example, when given 0.1 as ratio, the function will print the weight value,
        the abs(weights) lower than which count for 10% of the total numbers.
        Args:
          - model(paddle.nn.Layer): The model which have all the parameters.
          - ratio(float): The ratio illustrated above.
        Return:
          - threshold(float): a threshold corresponding to the input ratio.
        """
        data = []
        for name, sub_layer in model.named_sublayers():
            if not self._should_prune_layer(sub_layer):
                continue
            for param in sub_layer.parameters(include_sublayers=False):
                data.append(np.array(param.value().get_tensor()).flatten())
        data = np.concatenate(data, axis=0)
        threshold = np.sort(np.abs(data))[max(0, int(ratio * len(data) - 1))]
        return threshold

    def step(self):
        """
        Update the threshold after each optimization step.
        """
        if self.mode == 'ratio':
            self.update_threshold()
            self._update_masks()
        elif self.mode == 'threshold':
            self._update_masks()

    def _forward_pre_hook(self, layer, input):
        if not self._should_prune_layer(layer):
            return input
        for param in layer.parameters(include_sublayers=False):
            mask = self.masks.get(param.name)
            self.mask_parameters(param, mask)
        return input

    def update_params(self):
        """
        Update the parameters given self.masks, usually called before saving models and evaluation step during training. 
        If you load a sparse model and only want to inference, no need to call the method.
        """
        for name, sub_layer in self.model.named_sublayers():
            for param in sub_layer.parameters(include_sublayers=False):
                mask = self.masks.get(param.name)
                param_tmp = param * mask
                param_tmp.stop_gradient = True
                paddle.assign(param_tmp, output=param)

    @staticmethod
    def total_sparse(model):
        """
        This static function is used to get the whole model's sparsity.
        It is static because during testing, we can calculate sparsity without initializing a pruner instance.
        
        Args:
          - model(paddle.nn.Layer): The sparse model.
        Returns:
          - ratio(float): The model's sparsity.
        """
        total = 0
        values = 0
        for name, sub_layer in model.named_sublayers():
            for param in sub_layer.parameters(include_sublayers=False):
                total += np.product(param.shape)
                values += len(paddle.nonzero(param))
        ratio = 1 - float(values) / total
        return ratio

    @staticmethod
    def total_sparse_conv1x1(model):
        """
        This static function is used to get the partial model's sparsity in terms of conv1x1 layers.
        It is static because during testing, we can calculate sparsity without initializing a pruner instance.
        
        Args:
          - model(paddle.nn.Layer): The sparse model.
        Returns:
          - ratio(float): The model's sparsity.
        """
        total = 0
        values = 0
        for name, sub_layer in model.named_sublayers():
            if type(sub_layer).__name__.split('.')[-1] in NORMS_ALL:
                continue
            for param in sub_layer.parameters(include_sublayers=False):
                cond = len(param.shape) == 4 and param.shape[
                    2] == 1 and param.shape[3] == 1
                if not cond: continue
                total += np.product(param.shape)
                values += len(paddle.nonzero(param))
        ratio = 1 - float(values) / total
        return ratio

    def _get_skip_params(self, model):
        """
        This function is used to check whether the given model's layers are valid to be pruned. 
        Usually, the convolutions are to be pruned while we skip the normalization-related parameters.
        Deverlopers could replace this function by passing their own when initializing the UnstructuredPuner instance.
        Args:
          - model(Paddle.nn.Layer): the current model waiting to be checked.
        Return:
          - skip_params(set<String>): a set of parameters' names
        """
        skip_params = set()
        for _, sub_layer in model.named_sublayers():
            if type(sub_layer).__name__.split('.')[-1] in NORMS_ALL:
                skip_params.add(sub_layer.full_name())
        return skip_params

    def _get_skip_params_conv1x1(self, model):
        skip_params = set()
        for _, sub_layer in model.named_sublayers():
            if type(sub_layer).__name__.split('.')[-1] in NORMS_ALL:
                skip_params.add(sub_layer.full_name())
            for param in sub_layer.parameters(include_sublayers=False):
                cond = len(param.shape) == 4 and param.shape[
                    2] == 1 and param.shape[3] == 1
                if not cond: skip_params.add(sub_layer.full_name())
        return skip_params

    def _should_prune_layer(self, layer):
        should_prune = layer.full_name() not in self.skip_params
        return should_prune


class UnstructuredPrunerGMP(UnstructuredPruner):
    """
    The unstructure pruner using GMP training strategy (Gradual Magnitute Pruning). In this subclass of UnstructuredPruner, most methods are inheritated apart from the step(), since we add some ratio increment logics here.
    Conceptually, the algorithm divide the training into three phases: stable, pruning and tuning. And the ratio is increasing from initial_ratio gradually and nonlinearly w.r.t. the training epochs/iterations.
    Args:
      - model(Paddle.nn.Layer): The model to be pruned.
      - mode(str): Pruning mode, must be selected from 'ratio' and 'threshold'.
      - threshold(float): The parameters whose absolute values are smaller than the THRESHOLD will be zeros. Default: 0.01
      - ratio(float): The parameters whose absolute values are in the smaller part decided by the ratio will be zeros. Default: 0.55
      - skip_params_type(str): The argument to control which type of ops will be ignored. Currently we only support None or exclude_conv1x1 as input. It acts as a straightforward call to conv1x1 pruning.  Default: None
      - skip_params_func(function): The function used to select the parameters which should be skipped when performing pruning. Default: normalization-related params. 
      - configs(Dict): The dictionary contains all the configs for GMP pruner. Default: None
    """

    def __init__(self,
                 model,
                 mode,
                 threshold=0.01,
                 ratio=0.55,
                 skip_params_type=None,
                 skip_params_func=None,
                 configs=None):

        assert mode == 'ratio', "Mode must be RATIO in GMP pruner."
        assert configs is not None, "Configs must be passed in for GMP pruner."
        super(UnstructuredPrunerGMP, self).__init__(
            model, mode, threshold, ratio, skip_params_type, skip_params_func)
        self.stable_iterations = configs.get('stable_iterations')
        self.pruning_iterations = configs.get('pruning_iterations')
        self.tunning_iterations = configs.get('tunning_iterations')
        self.pruning_steps = configs.get('pruning_steps')
        self.initial_ratio = configs.get('initial_ratio')
        self.ratio = 0.0
        self.target_ratio = ratio
        self.cur_iteration = configs.get('resume_iteration')

        assert self.pruning_iterations / self.pruning_steps > 10, "To guarantee the performance of GMP pruner, pruning iterations must be larger than pruning steps by a margin."
        self._prepare_training_hyper_parameters()

    def _prepare_training_hyper_parameters(self):
        self.ratios_stack = []
        self.ratio_increment_period = int(self.pruning_iterations /
                                          self.pruning_steps)
        for i in range(self.pruning_steps):
            ratio_tmp = ((i / self.pruning_steps) - 1.0)**3 + 1
            ratio_tmp = ratio_tmp * (self.target_ratio - self.initial_ratio
                                     ) + self.initial_ratio
            self.ratios_stack.append(ratio_tmp)

        stable_steps = int(
            float(self.stable_iterations) / self.pruning_iterations *
            self.pruning_steps)
        tunning_steps = int(
            float(self.tunning_iterations) / self.pruning_iterations *
            self.pruning_steps)
        stable_ratios_stack = [0.0] * stable_steps
        tunning_ratios_stack = [self.target_ratio] * tunning_steps

        self.ratios_stack = stable_ratios_stack + self.ratios_stack + tunning_ratios_stack
        self.ratios_stack.reverse()

        # pop out used ratios to resume training
        for i in range(self.cur_iteration):
            if len(self.
                   ratios_stack) > 0 and i % self.ratio_increment_period == 0:
                self.ratio = self.ratios_stack.pop()

    def step(self):
        ori_ratio = self.ratio
        if self.cur_iteration % self.ratio_increment_period == 0:
            if len(self.ratios_stack) > 0:
                self.ratio = self.ratios_stack.pop()
            else:
                self.ratio = self.target_ratio

        # Update the threshold and masks only when a new ratio has been set.
        # This condition check would save training time dramatically since we only update the threshold by the triger of self.ratio_increment_period.
        if ori_ratio != self.ratio:
            self.update_threshold()
            self._update_masks()
        self.cur_iteration += 1


def make_unstructured_pruner(model,
                             mode,
                             threshold=0.01,
                             ratio=0.55,
                             skip_params_type=None,
                             skip_params_func=None,
                             configs=None):
    '''
    The entry function for different UnstructuredPruner classes.
    
    Args:
      They are exactly the same with class::UnstructuredPruner.
    Returns:
      - pruner(UnstructuredPruner): The pruner object.
    '''
    if configs is None or configs.get('pruning_strategy') == 'base':
        return UnstructuredPruner(
            model,
            mode,
            threshold=threshold,
            ratio=ratio,
            skip_params_type=skip_params_type,
            skip_params_func=skip_params_func)
    elif configs.get('pruning_strategy') == 'gmp':
        return UnstructuredPrunerGMP(
            model,
            mode,
            threshold=threshold,
            ratio=ratio,
            skip_params_type=skip_params_type,
            skip_params_func=skip_params_func,
            configs=configs)
    raise ValueError("{} is not supported.".format(
        configs.get('pruning_strategy')))
