from typing import Callable, Dict, Optional, Type, Union

from torch import nn
from torch.ao.quantization.fuser_method_mappings import reverse2, reverse3
from torch.ao.quantization.utils import Pattern

from nn.intrinsic import (
    BnConv1d,
    BnConv2d,
    BnConv3d,
    BnConvReLU1d,
    BnConvReLU2d,
    BnConvReLU3d,
    BnLinear1d,
    ConvsReLU1d,
    ConvsReLU2d,
    ConvsReLU3d,
)
from utils import fusion as fusion_utils


def fuse_bn_conv(is_qat, bn, conv):
    r"""Given the bn and conv modules, fuses them and returns the fused module

    Args:
        is_qat: a flag for whether we are using quantization aware training fusion
        or post training quantization fusion
        bn: Spatial BN instance that needs to be fused with the conv
        conv: Module instance of type conv2d/conv3d

    Examples::

        >>> b1 = nn.BatchNorm2d(10)
        >>> m1 = nn.Conv2d(10, 20, 3)
        >>> m2 = fuse_bn_conv(b1, m1)
    """
    assert bn.training == conv.training, (
        "BN and Conv both must be in the same mode (train or eval)."
    )

    fused_module_class_map = {
        nn.Conv1d: BnConv1d,
        nn.Conv2d: BnConv2d,
        nn.Conv3d: BnConv3d,
    }

    if is_qat:
        assert bn.num_features == conv.in_channels, (
            'Input channel of Conv must match num_features of BatchNorm'
        )
        assert bn.affine, (
            'Only support fusing BatchNorm with affine set to True'
        )
        assert bn.track_running_stats, (
            'Only support fusing BatchNorm2d with tracking_running_stats set '
            'to True'
        )
        fused_module_class = fused_module_class_map.get((type(conv)), None)
        if fused_module_class is not None:
            return fused_module_class(bn, conv)
        else:
            raise NotImplementedError(
                "Cannot fuse train modules: {}".format((conv, bn)))
    else:
        return fusion_utils.fuse_bn_conv_eval(bn, conv)


def fuse_bn_conv_relu(is_qat, bn, conv, relu):
    r"""Given the bn and conv modules, fuses them and returns the fused module

    Args:
        is_qat: a flag for whether we are using quantization aware training fusion
        or post training quantization fusion
        bn: Spatial BN instance that needs to be fused with the conv
        conv: Module instance of type conv2d/conv3d
        relu: Module instance of type ReLU

    Examples::

        >>> b1 = nn.BatchNorm2d(10)
        >>> m1 = nn.Conv2d(10, 20, 3)
        >>> r1 = nn.ReLU(inplace=False)
        >>> m2 = fuse_conv_bn_relu(b1, m1, r1)
    """
    assert bn.training == conv.training == relu.training, (
        "BN and Conv both must be in the same mode (train or eval)."
    )
    fused_module: Optional[Type[nn.Sequential]] = None
    if is_qat:
        map_to_fused_module_train = {
            nn.Conv1d: BnConvReLU1d,
            nn.Conv2d: BnConvReLU2d,
            nn.Conv3d: BnConvReLU3d,
        }
        assert bn.num_features == conv.in_channels, (
            'Input channel of Conv must match num_features of BatchNorm'
        )
        assert bn.affine, (
            'Only support fusing BatchNorm with affine set to True'
        )
        assert bn.track_running_stats, (
            'Only support fusing BatchNorm with tracking_running_stats set '
            'to True'
        )
        fused_module = map_to_fused_module_train.get(type(conv), None)
        if fused_module is not None:
            return fused_module(bn, conv, relu)
        else:
            raise NotImplementedError(
                "Cannot fuse train modules: {}".format((bn, conv, relu)))
    else:
        map_to_fused_module_eval = {
            nn.Conv1d: ConvsReLU1d,
            nn.Conv2d: ConvsReLU2d,
            nn.Conv3d: ConvsReLU3d,
        }
        fused_module = map_to_fused_module_eval.get(type(conv), None)
        if fused_module is not None:
            fused_conv = fusion_utils.fuse_bn_conv_eval(bn, conv)
            return fused_module(fused_conv, relu)
        else:
            raise NotImplementedError(
                "Cannot fuse eval modules: {}".format((bn, conv, relu)))


def fuse_bn_linear(is_qat, bn, linear):
    r"""Given the bn and linear modules, fuses them and returns the fused module

    Args:
        is_qat: a flag for whether we are using quantization aware training fusion
        or post training quantization fusion
        bn: BatchNorm1d instance that needs to be fused with the linear layer
        linear: Module instance of type Linear

    Examples::

        >>> b1 = nn.BatchNorm1d(10)
        >>> m1 = nn.Linear(10, 20)
        >>> m2 = fuse_bn_linear(b1, m1)
    """
    assert(bn.training == linear.training), (
        "BN and Linear both must be in the same mode (train or eval)."
    )

    if is_qat:
        assert bn.num_features == linear.in_features, (
            "Output features of Linear must match num_features of BatchNorm1d"
        )
        assert bn.affine, (
            "Only support fusing BatchNorm1d with affine set to True"
        )
        assert bn.track_running_stats, (
            "Only support fusing BatchNorm1d with tracking_running_stats set "
            "to True"
        )
        return BnLinear1d(linear, bn)
    else:
        return fusion_utils.fuse_bn_linear_eval(linear, bn)


CUSTOM_PATTERN_TO_FUSER_METHOD: Dict[Pattern, Union[nn.Sequential, Callable]] = {
    (nn.Conv1d, nn.BatchNorm1d): reverse2(fuse_bn_conv),
    (nn.Conv2d, nn.BatchNorm2d): reverse2(fuse_bn_conv),
    (nn.Conv3d, nn.BatchNorm3d): reverse2(fuse_bn_conv),
    (nn.ReLU, (nn.Conv1d, nn.BatchNorm1d)): reverse3(fuse_bn_conv_relu),
    (nn.ReLU, (nn.Conv2d, nn.BatchNorm2d)): reverse3(fuse_bn_conv_relu),
    (nn.ReLU, (nn.Conv3d, nn.BatchNorm3d)): reverse3(fuse_bn_conv_relu),
    (nn.Linear, nn.BatchNorm1d): reverse2(fuse_bn_linear),
}
