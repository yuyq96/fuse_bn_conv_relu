import copy

import torch
from torch import nn
from torch.nn.utils.parametrize import type_before_parametrizations


# Used for identifying intrinsic modules used in quantization
class _FusedModule(torch.nn.Sequential):
    pass


class BnConv1d(_FusedModule):
    r"""This is a sequential container which calls the Batch Norm 1d and Conv 1d modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, bn, conv):
        assert (
            type_before_parametrizations(bn) == nn.BatchNorm1d
            and type_before_parametrizations(conv) == nn.Conv1d
        ), 'Incorrect types for input modules{}{}'.format(
            type_before_parametrizations(bn),
            type_before_parametrizations(conv),
        )
        super().__init__(bn, conv)


class BnConv2d(_FusedModule):
    r"""This is a sequential container which calls the Batch Norm 2d and Conv 2d modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, bn, conv):
        assert (
            type_before_parametrizations(bn) == nn.BatchNorm2d
            and type_before_parametrizations(conv) == nn.Conv2d
        ), 'Incorrect types for input modules{}{}'.format(
            type_before_parametrizations(bn),
            type_before_parametrizations(conv),
        )
        super().__init__(bn, conv)


class BnConv3d(_FusedModule):
    r"""This is a sequential container which calls the Batch Norm 3d and Conv 3d modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, bn, conv):
        assert (
            type_before_parametrizations(bn) == nn.BatchNorm3d
            and type_before_parametrizations(conv) == nn.Conv3d
        ), 'Incorrect types for input modules{}{}'.format(
            type_before_parametrizations(bn),
            type_before_parametrizations(conv),
        )
        super().__init__(bn, conv)


class BnConvReLU1d(_FusedModule):
    r"""This is a sequential container which calls the Batch Norm 1d, Conv 1d, and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, bn, conv, relu):
        assert (
            type_before_parametrizations(bn) == nn.BatchNorm1d
            and type_before_parametrizations(conv) == nn.Conv1d
            and type_before_parametrizations(relu) == nn.ReLU
        ), 'Incorrect types for input modules{}{}{}'.format(
            type_before_parametrizations(bn),
            type_before_parametrizations(conv),
            type_before_parametrizations(relu),
        )
        super().__init__(bn, conv, relu)


class BnConvReLU2d(_FusedModule):
    r"""This is a sequential container which calls the Batch Norm 2d, Conv 2d, and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, bn, conv, relu):
        assert (
            type_before_parametrizations(bn) == nn.BatchNorm2d
            and type_before_parametrizations(conv) == nn.Conv2d
            and type_before_parametrizations(relu) == nn.ReLU
        ), 'Incorrect types for input modules{}{}{}'.format(
            type_before_parametrizations(bn),
            type_before_parametrizations(conv),
            type_before_parametrizations(relu),
        )
        super().__init__(bn, conv, relu)


class BnConvReLU3d(_FusedModule):
    r"""This is a sequential container which calls the Batch Norm 3d, Conv 3d, and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""

    def __init__(self, bn, conv, relu):
        assert (
            type_before_parametrizations(bn) == nn.BatchNorm3d
            and type_before_parametrizations(conv) == nn.Conv3d
            and type_before_parametrizations(relu) == nn.ReLU
        ), 'Incorrect types for input modules{}{}{}'.format(
            type_before_parametrizations(bn),
            type_before_parametrizations(conv),
            type_before_parametrizations(relu),
        )
        super().__init__(bn, conv, relu)


class BnLinear1d(_FusedModule):
    r"""This is a sequential container which calls the BatchNorm1d and Linear modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, linear, bn):
        assert (
            type_before_parametrizations(bn) == nn.BatchNorm1d
            and type_before_parametrizations(linear) == nn.Linear
        ), 'Incorrect types for input modules{}{}'.format(
            type_before_parametrizations(bn),
            type_before_parametrizations(linear),
        )
        super().__init__(bn, linear)


def fuse_bn_conv_eval(bn, conv, transpose=False):
    assert(not (bn.training or conv.training)), "Fusion only for eval!"
    fused_conv = copy.deepcopy(conv)

    fused_conv.weight, fused_conv.bias = fuse_bn_conv_weights(
        bn_rm=bn.running_mean,
        bn_rv=bn.running_var,
        bn_eps=bn.eps,
        bn_w=bn.weight,
        bn_b=bn.bias,
        conv_w=fused_conv.weight,
        conv_b=fused_conv.bias,
        transpose=transpose,
    )

    return fused_conv


def fuse_bn_conv_weights(
    bn_rm, bn_rv, bn_eps, bn_w, bn_b, conv_w, conv_b, transpose=False
):
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_scale = bn_w * torch.rsqrt(bn_rv + bn_eps)
    if conv_b is None:
        if transpose:
            conv_b = conv_w.new_zeros(conv_w.shape[1])
        else:
            conv_b = conv_w.new_zeros(conv_w.shape[0])

    if transpose:
        shape = [-1, 1] + [1] * (len(conv_w.shape) - 2)
    else:
        shape = [1, -1] + [1] * (len(conv_w.shape) - 2)

    fused_w = conv_w * bn_scale.reshape(shape)
    fused_b = torch.addmv(conv_b, conv_w.sum(tuple(range(2, len(conv_w.shape)))), bn_b - bn_rm * bn_scale)

    return torch.nn.Parameter(fused_w), torch.nn.Parameter(fused_b)


def fuse_bn_linear_eval(linear, bn):
    assert(not (bn.training or linear.training)), "Fusion only for eval!"
    fused_linear = copy.deepcopy(linear)

    fused_linear.weight, fused_linear.bias = fuse_bn_linear_weights(
        bn_rm=bn.running_mean,
        bn_rv=bn.running_var,
        bn_eps=bn.eps,
        bn_w=bn.weight,
        bn_b=bn.bias,
        linear_w=fused_linear.weight,
        linear_b=fused_linear.bias,
    )

    return fused_linear


def fuse_bn_linear_weights(
    bn_rm, bn_rv, bn_eps, bn_w, bn_b, linear_w, linear_b
):
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_scale = bn_w * torch.rsqrt(bn_rv + bn_eps)
    if linear_b is None:
        linear_b = torch.zeros_like(bn_rm)

    fused_w = linear_w * bn_scale.unsqueeze(0)
    fused_b = torch.addmv(linear_b, linear_w, bn_b - bn_rm * bn_scale)

    return torch.nn.Parameter(fused_w), torch.nn.Parameter(fused_b)
