import torch
from torch import nn
from torch.nn.utils.parametrize import type_before_parametrizations


# Used for identifying intrinsic modules used in quantization
class _FusedModule(torch.nn.Sequential):
    pass


class ConvsReLU1d(_FusedModule):
    r"""This is a sequential container which calls the Conv1d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, convs, relu):
        assert (
            (
                type_before_parametrizations(convs) == nn.Conv1d
                or (
                    type_before_parametrizations(convs) == nn.Sequential
                    and all(
                        type_before_parametrizations(conv) == nn.Conv1d
                        for conv in convs
                    )
                )
            )
            and type_before_parametrizations(relu) == nn.ReLU
        ), 'Incorrect types for input modules{}{}'.format(
            type_before_parametrizations(convs),
            type_before_parametrizations(relu)
        )
        super().__init__(convs, relu)


class ConvsReLU2d(_FusedModule):
    r"""This is a sequential container which calls the Conv2d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, convs, relu):
        assert (
            (
                type_before_parametrizations(convs) == nn.Conv2d
                or (
                    type_before_parametrizations(convs) == nn.Sequential
                    and all(
                        type_before_parametrizations(conv) == nn.Conv2d
                        for conv in convs
                    )
                )
            )
            and type_before_parametrizations(relu) == nn.ReLU
        ), 'Incorrect types for input modules{}{}'.format(
            type_before_parametrizations(convs),
            type_before_parametrizations(relu)
        )
        super().__init__(convs, relu)


class ConvsReLU3d(_FusedModule):
    r"""This is a sequential container which calls the Conv3d and ReLU modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, convs, relu):
        assert (
            (
                type_before_parametrizations(convs) == nn.Conv3d
                or (
                    type_before_parametrizations(convs) == nn.Sequential
                    and all(
                        type_before_parametrizations(conv) == nn.Conv3d
                        for conv in convs
                    )
                )
            )
            and type_before_parametrizations(relu) == nn.ReLU
        ), 'Incorrect types for input modules{}{}'.format(
            type_before_parametrizations(convs),
            type_before_parametrizations(relu)
        )
        super().__init__(convs, relu)


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
