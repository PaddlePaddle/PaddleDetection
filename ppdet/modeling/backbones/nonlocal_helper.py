from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import paddle.fluid as fluid
from paddle.fluid import ParamAttr
from paddle.fluid.initializer import Normal, Constant
from paddle.fluid.regularizer import L2Decay
                
nonlocal_params = {
    "use_zero_init_conv" : False,
    "conv_init_std" : 0.01,
    "use_maxpool"   : False,
    "use_bn"        : False,
    "use_scale"     : True,  # vital for the model prformance!!!
    "bn_momentum"   : 0.9,
    "bn_epsilon"    : 1.0000001e-5,
    "bn_init_gamma" : 0.9,
    "weight_decay_bn":1.e-4,
}

def space_nonlocal(input, dim_in, dim_out, prefix, dim_inner, max_pool_stride = 2):
    cur = input
    theta = fluid.layers.conv2d(
                input=cur,
                num_filters=dim_inner,
                filter_size=1,
                stride=1,
                padding=0,
                param_attr=
                    ParamAttr(name = prefix + '_theta_w',
                        initializer = Normal(
                            loc = 0.0, 
                            scale = nonlocal_params["conv_init_std"])
                             ),
                bias_attr=False,
                name = prefix + '_theta')
    theta_shape = theta.shape
    theta_shape_op = fluid.layers.shape( theta )
    theta_shape_op.stop_gradient = True
    
    if nonlocal_params["use_maxpool"]:
        max_pool = fluid.layers.pool2d(
                        input=cur,
                        pool_size=max_pool_stride,
                        pool_type='max',
                        pool_stride=max_pool_stride,
                        pool_padding=0,
                        name=prefix + '_pool')
    else:
        max_pool = cur
    
    phi = fluid.layers.conv2d(
              input=max_pool,
              num_filters=dim_inner,
              filter_size=1,
              stride=1,
              padding=0,
              param_attr = ParamAttr(name = prefix + '_phi_w',
                 initializer = Normal(loc = 0.0, 
                 scale = nonlocal_params["conv_init_std"])),
              bias_attr = False,
              name = prefix + '_phi')
    phi_shape = phi.shape
    
    g = fluid.layers.conv2d(
            input = max_pool,
            num_filters=dim_inner,
            filter_size=1,
            stride=1,
            padding=0,
            param_attr = 
                ParamAttr(name = prefix + '_g_w',
                initializer = 
                    Normal(loc = 0.0, 
                        scale = nonlocal_params["conv_init_std"])),
            bias_attr = False,
            name = prefix + '_g')
    g_shape = g.shape
    # we have to use explicit batch size (to support arbitrary spacetime size)
    # e.g. (8, 1024, 4, 14, 14) => (8, 1024, 784)
    theta = fluid.layers.reshape(theta, shape=(0, 0, -1) )
    theta = fluid.layers.transpose(theta, [0, 2, 1])
    phi = fluid.layers.reshape(phi, [0, 0, -1])
    theta_phi = fluid.layers.matmul(theta, phi, name = prefix + '_affinity')
    g = fluid.layers.reshape(g, [0, 0, -1])
    
    # softmax
    if nonlocal_params["use_scale"]:
        theta_phi_sc = fluid.layers.scale(theta_phi, scale = dim_inner**-.5)
    else:
        theta_phi_sc = theta_phi
    p = fluid.layers.softmax(theta_phi_sc, name = prefix + '_affinity' + '_prob')

    # note g's axis[2] corresponds to p's axis[2]
    # e.g. g(8, 1024, 784_2) * p(8, 784_1, 784_2) => (8, 1024, 784_1)
    p = fluid.layers.transpose(p, [0, 2, 1])
    t = fluid.layers.matmul(g, p, name = prefix + '_y')

    # reshape back
    # e.g. (8, 1024, 784) => (8, 1024, 4, 14, 14)
    t_shape = t.shape
    
    n = fluid.layers.slice(theta_shape_op, axes=[0], starts=[0], ends=[1])
    h = fluid.layers.slice(theta_shape_op, axes=[0], starts=[2], ends=[3])
    w = fluid.layers.slice(theta_shape_op, axes=[0], starts=[3], ends=[4])
    ch = int(theta_shape[1])
    
    t_re = fluid.layers.reshape(t, shape=[n, ch, h, w])
    blob_out = t_re
    blob_out = fluid.layers.conv2d(
                    input = blob_out,
                    num_filters = dim_out,
                    filter_size = [1, 1],
                    stride = [1, 1], 
                    padding = [0, 0],
                    param_attr = ParamAttr(
                        name = prefix + '_out' + "_w",
                        initializer = Constant(value = 0.)
                            if nonlocal_params["use_zero_init_conv"]
                            else Normal(loc=0.0, 
                                scale=nonlocal_params["conv_init_std"])),
                    bias_attr = False,
                    name = prefix + '_out')
    blob_out_shape = blob_out.shape
    
    if nonlocal_params["use_bn"]:
        bn_name = prefix + "_bn"
        blob_out = fluid.layers.batch_norm(blob_out,
                      momentum = nonlocal_params["bn_momentum"],
                      epsilon = nonlocal_params["bn_epsilon"],
                      name = bn_name,
                      param_attr = ParamAttr(name = bn_name + "_s",
                          initializer = Constant(value = nonlocal_params["bn_init_gamma"]),
                          regularizer = L2Decay(nonlocal_params["weight_decay_bn"])),
                      bias_attr = ParamAttr(name = bn_name + "_b",
                          regularizer = L2Decay(nonlocal_params["weight_decay_bn"])),
                              moving_mean_name = bn_name + "_rm",
                              moving_variance_name = bn_name + "_riv")
    
    return blob_out


def add_space_nonlocal(input, dim_in, dim_out, prefix, dim_inner ):
    '''
    add_space_nonlocal: 
        Non-local Neural Networks: see https://arxiv.org/abs/1711.07971
    '''
    conv = space_nonlocal(input, dim_in, dim_out, prefix, dim_inner)
    output = fluid.layers.elementwise_add(input, conv, name = prefix + '_sum')
    return output

