from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppdet.core.workspace import load_config, merge_config, create
from ppdet.utils.checkpoint import load_weight, load_pretrain_weight
from ppdet.utils.logger import setup_logger

from paddleslim.nas.ofa import OFA, RunConfig, utils
from paddleslim.nas.ofa.convert_super import Convert, supernet

logger = setup_logger(__name__)

def OFAModel(cfg, slim_cfg):
    # create model
    model = create(cfg.architecture)
    # load pretrain weights
    if cfg.pretrain_weights is not None:
        load_pretrain_weight(model, cfg.pretrain_weights)
    # get model state dict which is used to set into ofa model
    param_state_dict = model.state_dict()
    paddle.save(model.state_dict(),'check_ofa.pdparams')

    slim_cfg = load_config(slim_cfg)

    ofa_model = build_ofa(model, slim_cfg)
    
    input_spec = [{
        "image": paddle.ones(
            shape=[1, 3, 640, 640], dtype='float32'),
        "im_shape": paddle.full(
            [1, 2], 640, dtype='float32'),
        "scale_factor": paddle.ones(
            shape=[1, 2], dtype='float32')
        }]

    ofa_model._clear_search_space(input_spec=input_spec)
    ofa_model._build_ss = True
    check_ss = ofa_model._sample_config('expand_ratio', phase=None)
    ### tokenize the search space
    ofa_model.tokenize()
    ### check token map, search cands and search space
    logger.info('Token map is {}'.format(ofa_model.token_map))
    logger.info('Search candidates is {}'.format(ofa_model.search_cands))
    logger.info('The length of search_space is {}, search_space is {}'.format(len(ofa_model._ofa_layers), ofa_model._ofa_layers))
    # set model state dict into ofa model
    utils.set_state_dict(ofa_model.model, param_state_dict)
    return ofa_model
              
def build_ofa(model, slim_cfg):
    # ofa and supernet config
    task = slim_cfg.task
    expand_ratio = slim_cfg.expand_ratio
    
    skip_neck = slim_cfg.skip_neck
    skip_head = slim_cfg.skip_head
    
    run_config = slim_cfg.RunConfig
    if 'skip_layers' in run_config:
        skip_layers = run_config['skip_layers']
    else:
        skip_layers = []
    
    # supernet config
    sp_config = supernet(expand_ratio=expand_ratio)
    # convert to supernet
    model = Convert(sp_config).convert(model)

    skip_names = []
    if skip_neck:
        skip_names.append('neck.')
    if skip_head:
        skip_names.append('head.')
        
    for name, sublayer in model.named_sublayers():
        for n in skip_names:
            if n in name:
                skip_layers.append(name) 
    
    run_config['skip_layers'] = skip_layers
    run_config = RunConfig(**run_config)

    # build ofa model
    ofa_model = OFA(model, run_config=run_config)

    ofa_model.set_epoch(0)
    ofa_model.set_task(task)

    return ofa_model
