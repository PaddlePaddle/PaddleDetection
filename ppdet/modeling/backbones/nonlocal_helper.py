from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import paddle.fluid as fluid
from paddle.fluid import ParamAttr

                
nonlocal_params = {
    "use_zero_init_conv" : False,
    "conv_init_std" : 0.01,
    "no_bias"       : True,
    "use_maxpool"   : False,
    "use_softmax"   : True,
    "use_bn"        : False,
    "use_scale"     : True,  # vital for the model prformance!!!
    "use_affine"    : False,
    "bn_momentum"   : 0.9,
    "bn_epsilon"    : 1.0000001e-5,
    "bn_init_gamma" : 0.9,
    "weight_decay_bn":1.e-4,
    
}


def space_nonlocal(input, dim_in, dim_out, prefix, dim_inner, max_pool_stride = 2):
    cur = input
    theta = fluid.layers.conv2d(input = cur, num_filters = dim_inner, \
                                filter_size = [1, 1], stride = [1, 1], \
                                padding = [0, 0], \
                                param_attr=ParamAttr(name = prefix + '_theta' + "_w", \
                                    initializer = fluid.initializer.Normal(loc = 0.0, 
                                    scale = nonlocal_params["conv_init_std"])), \
                                bias_attr = ParamAttr(name = prefix + '_theta' + "_b", \
                                    initializer = fluid.initializer.Constant(value = 0.)) \
                                        if not nonlocal_params["no_bias"] else False, \
                                name = prefix + '_theta')
    theta_shape = theta.shape
    theta_shape_op = fluid.layers.shape( theta )
    theta_shape_op.stop_gradient = True
    
    if nonlocal_params["use_maxpool"]:
        max_pool = fluid.layers.pool2d(input = cur, \
                                        pool_size = [max_pool_stride, max_pool_stride], \
                                        pool_type = 'max', \
                                        pool_stride = [max_pool_stride, max_pool_stride], \
                                        pool_padding = [0, 0], \
                                        name = prefix + '_pool')
    else:
        max_pool = cur
    
    phi = fluid.layers.conv2d(input = max_pool, num_filters = dim_inner, \
                             filter_size = [1, 1], stride = [1, 1], \
                             padding = [0, 0], \
                             param_attr = ParamAttr(name = prefix + '_phi' + "_w", \
                                 initializer = fluid.initializer.Normal(loc = 0.0, 
                                 scale = nonlocal_params["conv_init_std"])), \
                             bias_attr = ParamAttr(name = prefix + '_phi' + "_b", \
                                 initializer = fluid.initializer.Constant(value = 0.)) \
                                      if (nonlocal_params["no_bias"] == 0) else False, \
                             name = prefix + '_phi')
    phi_shape = phi.shape
    
    g = fluid.layers.conv2d(input = max_pool, num_filters = dim_inner, \
                 filter_size = [1, 1], stride = [1, 1], \
                 padding = [0, 0], \
                 param_attr = ParamAttr(name = prefix + '_g' + "_w", \
                     initializer = fluid.initializer.Normal(loc = 0.0, scale = nonlocal_params["conv_init_std"])), \
                 bias_attr = ParamAttr(name = prefix + '_g' + "_b", \
                     initializer = fluid.initializer.Constant(value = 0.)) if (nonlocal_params["no_bias"] == 0) else False, \
                 name = prefix + '_g')
    g_shape = g.shape
    # we have to use explicit batch size (to support arbitrary spacetime size)
    # e.g. (8, 1024, 4, 14, 14) => (8, 1024, 784)
    theta = fluid.layers.reshape(theta, shape=(0, 0, -1) )
    theta = fluid.layers.transpose(theta, [0, 2, 1])
    phi = fluid.layers.reshape(phi, [0, 0, -1])
    theta_phi = fluid.layers.matmul(theta, phi, name = prefix + '_affinity')
    g = fluid.layers.reshape(g, [0, 0, -1])
    
    if nonlocal_params["use_softmax"]:
        if nonlocal_params["use_scale"]:
            theta_phi_sc = fluid.layers.scale(theta_phi, scale = dim_inner**-.5)
        else:
            theta_phi_sc = theta_phi
        p = fluid.layers.softmax(theta_phi_sc, name = prefix + '_affinity' + '_prob')
    else:
        # not clear about what is doing in xlw's code
        p = None  # not implemented
        raise "Not implemented when not use softmax"

    # note g's axis[2] corresponds to p's axis[2]
    # e.g. g(8, 1024, 784_2) * p(8, 784_1, 784_2) => (8, 1024, 784_1)
    p = fluid.layers.transpose(p, [0, 2, 1])
    t = fluid.layers.matmul(g, p, name = prefix + '_y')

    # reshape back
    # e.g. (8, 1024, 784) => (8, 1024, 4, 14, 14)
    n = fluid.layers.slice(theta_shape_op, axes=[0], starts=[0], ends=[1])
    h = fluid.layers.slice(theta_shape_op, axes=[0], starts=[2], ends=[3])
    w = fluid.layers.slice(theta_shape_op, axes=[0], starts=[3], ends=[4])
    ch = int(theta_shape[1])

    t_re = fluid.layers.reshape(t, shape=[n, ch, h, w])
    blob_out = t_re
    blob_out = fluid.layers.conv2d(input = blob_out, num_filters = dim_out, \
                                  filter_size = [1, 1], stride = [1, 1], padding = [0, 0], \
                                  param_attr = ParamAttr(name = prefix + '_out' + "_w", \
                                      initializer = fluid.initializer.Constant(value = 0.) \
                                        if nonlocal_params["use_zero_init_conv"] \
                                        else fluid.initializer.Normal(loc = 0.0, 
                                            scale = nonlocal_params["conv_init_std"])), \
                                  bias_attr = ParamAttr(name = prefix + '_out' + "_b", \
                                          initializer = fluid.initializer.Constant(value = 0.)) \
                                           if (nonlocal_params["no_bias"] == 0) else False, \
                                  name = prefix + '_out')
    blob_out_shape = blob_out.shape
    

    if nonlocal_params["use_bn"]:
        bn_name = prefix + "_bn"
        blob_out = fluid.layers.batch_norm(blob_out, \
                      # is_test = test_mode, \
                      momentum = nonlocal_params["bn_momentum"], \
                      epsilon = nonlocal_params["bn_epsilon"], \
                      name = bn_name, \
                      param_attr = ParamAttr(name = bn_name + "_s", \
                      initializer = fluid.initializer.Constant(value = nonlocal_params["bn_init_gamma"]), \
                      regularizer = fluid.regularizer.L2Decay(nonlocal_params["weight_decay_bn"])), \
                      bias_attr = ParamAttr(name = bn_name + "_b", \
                      regularizer = fluid.regularizer.L2Decay(nonlocal_params["weight_decay_bn"])), \
                      moving_mean_name = bn_name + "_rm", \
                      moving_variance_name = bn_name + "_riv") # add bn
    
    if nonlocal_params["use_affine"]:
        affine_scale = fluid.layers.create_parameter(\
                       shape=[blob_out_shape[1]], dtype = blob_out.dtype, \
                       attr=ParamAttr(name=prefix + '_affine' + '_s'), \
                       default_initializer = fluid.initializer.Constant(value = 1.))
        affine_bias = fluid.layers.create_parameter(\
                      shape=[blob_out_shape[1]], dtype = blob_out.dtype, \
                      attr=ParamAttr(name=prefix + '_affine' + '_b'), \
                      default_initializer = fluid.initializer.Constant(value = 0.))
        blob_out = fluid.layers.affine_channel(blob_out, scale = affine_scale, \
                      bias = affine_bias, name = prefix + '_affine')   # add affine
    
    return blob_out


def add_space_nonlocal(input, dim_in, dim_out, prefix, dim_inner ):
    '''
    add_space_nonlocal: 
        Non-local Neural Networks: see https://arxiv.org/abs/1711.07971
    '''
    conv = space_nonlocal(input, dim_in, dim_out, prefix, dim_inner)
    output = fluid.layers.elementwise_add(input, conv, name = prefix + '_sum')
    return output

