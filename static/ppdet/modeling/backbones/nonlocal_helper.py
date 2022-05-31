from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import paddle.fluid as fluid
from paddle.fluid import ParamAttr
from paddle.fluid.initializer import ConstantInitializer


def space_nonlocal(input,
                   dim_in,
                   dim_out,
                   prefix,
                   dim_inner,
                   with_bias=False,
                   with_scale=True):
    theta = fluid.layers.conv2d(
        input=input,
        num_filters=dim_inner,
        filter_size=1,
        stride=1,
        padding=0,
        param_attr=ParamAttr(name=prefix + '_theta_w'),
        bias_attr=ParamAttr(
            name=prefix + '_theta_b', initializer=ConstantInitializer(value=0.))
        if with_bias else False)
    theta_shape = theta.shape
    theta_shape_op = fluid.layers.shape(theta)
    theta_shape_op.stop_gradient = True

    # we have to use explicit batch size (to support arbitrary spacetime size)
    # e.g. (8, 1024, 4, 14, 14) => (8, 1024, 784)
    theta = fluid.layers.reshape(theta, shape=(0, 0, -1))
    theta = fluid.layers.transpose(theta, [0, 2, 1])

    phi = fluid.layers.conv2d(
        input=input,
        num_filters=dim_inner,
        filter_size=1,
        stride=1,
        padding=0,
        param_attr=ParamAttr(name=prefix + '_phi_w'),
        bias_attr=ParamAttr(
            name=prefix + '_phi_b', initializer=ConstantInitializer(value=0.))
        if with_bias else False,
        name=prefix + '_phi')
    phi = fluid.layers.reshape(phi, [0, 0, -1])

    theta_phi = fluid.layers.matmul(theta, phi)

    g = fluid.layers.conv2d(
        input=input,
        num_filters=dim_inner,
        filter_size=1,
        stride=1,
        padding=0,
        param_attr=ParamAttr(name=prefix + '_g_w'),
        bias_attr=ParamAttr(
            name=prefix + '_g_b', initializer=ConstantInitializer(value=0.))
        if with_bias else False,
        name=prefix + '_g')
    g = fluid.layers.reshape(g, [0, 0, -1])

    # scale
    if with_scale:
        theta_phi = fluid.layers.scale(theta_phi, scale=dim_inner**-.5)
    p = fluid.layers.softmax(theta_phi)

    # note g's axis[2] corresponds to p's axis[2]
    # e.g. g(8, 1024, 784_2) * p(8, 784_1, 784_2) => (8, 1024, 784_1)
    p = fluid.layers.transpose(p, [0, 2, 1])
    t = fluid.layers.matmul(g, p)

    # reshape back
    # e.g. (8, 1024, 784) => (8, 1024, 4, 14, 14)
    n = fluid.layers.slice(theta_shape_op, axes=[0], starts=[0], ends=[1])
    h = fluid.layers.slice(theta_shape_op, axes=[0], starts=[2], ends=[3])
    w = fluid.layers.slice(theta_shape_op, axes=[0], starts=[3], ends=[4])
    ch = int(theta_shape[1])

    t_re = fluid.layers.reshape(t, shape=[n, ch, h, w])
    blob_out = t_re
    blob_out = fluid.layers.conv2d(
        input=blob_out,
        num_filters=dim_out,
        filter_size=1,
        stride=1,
        padding=0,
        param_attr=ParamAttr(
            name=prefix + '_out_w', initializer=ConstantInitializer(value=0.0)),
        bias_attr=ParamAttr(
            name=prefix + '_out_b', initializer=ConstantInitializer(value=0.0))
        if with_bias else False,
        name=prefix + '_out')
    blob_out_shape = blob_out.shape
    return blob_out


def add_space_nonlocal(input,
                       dim_in,
                       dim_out,
                       prefix,
                       dim_inner,
                       with_bias=False,
                       with_scale=True):
    '''
    add_space_nonlocal: 
        Non-local Neural Networks: see https://arxiv.org/abs/1711.07971
    '''
    conv = space_nonlocal(
        input,
        dim_in,
        dim_out,
        prefix,
        dim_inner,
        with_bias=with_bias,
        with_scale=with_scale)
    output = input + conv
    return output
