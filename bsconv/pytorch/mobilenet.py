import re
import types

import torch.nn
import torch.nn.init

import bsconv.pytorch
from bsconv.pytorch.common import conv1x1_block, conv3x3_block, conv3x3_dw_block, conv5x5_dw_block, SEUnit, Classifier


###
#%% MobileNet building blocks
###


class DepthwiseSeparableConvBlock(torch.nn.Module):
    """
    Depthwise-separable convolution (DSC) block internally used in MobileNets.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride):
        super().__init__()

        self.conv_dw = conv3x3_dw_block(channels=in_channels, stride=stride)
        self.conv_pw = conv1x1_block(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        return x


class LinearBottleneck(torch.nn.Module):
    """
    Linear bottleneck block internally used in MobileNets.
    """
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 stride,
                 activation="relu6",
                 kernel_size=3,
                 use_se=False):
        super().__init__()
        self.use_res_skip = (in_channels == out_channels) and (stride == 1)
        self.use_se = use_se

        self.conv1 = conv1x1_block(in_channels=in_channels, out_channels=mid_channels, activation=activation)
        if kernel_size == 3:
            self.conv2 = conv3x3_dw_block(channels=mid_channels, stride=stride, activation=activation)
        elif kernel_size == 5:
            self.conv2 = conv5x5_dw_block(channels=mid_channels, stride=stride, activation=activation)
        else:
            raise ValueError
        if self.use_se:
            self.se_unit = SEUnit(channels=mid_channels, squeeze_factor=4, squeeze_activation="relu", excite_activation="hsigmoid")
        self.conv3 = conv1x1_block(in_channels=mid_channels, out_channels=out_channels, activation=None)

    def forward(self, x):
        if self.use_res_skip:
            residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_se:
            x = self.se_unit(x)
        x = self.conv3(x)
        if self.use_res_skip:
            x = x + residual
        return x


class MobileNetV1(torch.nn.Module):
    """
    Class for constructing MobileNetsV1.
    
    If you are in doubt, please use the high-level function `get_mobilenet` to
    obtain ready-to-use models.
    """
    def __init__(self,
                 num_classes,
                 init_conv_channels,
                 init_conv_stride,
                 channels,
                 strides,
                 in_channels=3,
                 in_size=(224, 224),
                 use_data_batchnorm=True):
        super().__init__()
        self.use_data_batchnorm = use_data_batchnorm
        self.in_size = in_size

        self.backbone = torch.nn.Sequential()

        # data batchnorm
        if self.use_data_batchnorm:
            self.backbone.add_module("data_bn", torch.nn.BatchNorm2d(num_features=in_channels))

        # init conv
        self.backbone.add_module("init_conv", conv3x3_block(in_channels=in_channels, out_channels=init_conv_channels, stride=init_conv_stride))

        # stages
        in_channels = init_conv_channels
        for stage_id, stage_channels in enumerate(channels):
            stage = torch.nn.Sequential()
            for unit_id, unit_channels in enumerate(stage_channels):
                stride = strides[stage_id] if unit_id == 0 else 1
                stage.add_module("unit{}".format(unit_id + 1), DepthwiseSeparableConvBlock(in_channels=in_channels, out_channels=unit_channels, stride=stride))
                in_channels = unit_channels
            self.backbone.add_module("stage{}".format(stage_id + 1), stage)
        self.backbone.add_module("global_pool", torch.nn.AdaptiveAvgPool2d(output_size=1))

        # classifier
        self.classifier = Classifier(in_channels=in_channels, num_classes=num_classes)

        self.init_params()

    def init_params(self):
        # backbone
        for name, module in self.backbone.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)

        # classifier
        self.classifier.init_params()

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


class MobileNetV2(torch.nn.Module):
    """
    Class for constructing MobileNetsV2.
    
    If you are in doubt, please use the high-level function `get_mobilenet` to
    obtain ready-to-use models.
    """
    def __init__(self,
                 num_classes,
                 init_conv_channels,
                 init_conv_stride,
                 channels,
                 mid_channels,
                 final_conv_channels,
                 strides,
                 in_channels=3,
                 in_size=(224, 224),
                 use_data_batchnorm=True):
        super().__init__()
        self.use_data_batchnorm = use_data_batchnorm
        self.in_size = in_size

        self.backbone = torch.nn.Sequential()

        # data batchnorm
        if self.use_data_batchnorm:
            self.backbone.add_module("data_bn", torch.nn.BatchNorm2d(num_features=in_channels))

        # init conv
        self.backbone.add_module("init_conv", conv3x3_block(in_channels=in_channels, out_channels=init_conv_channels, stride=init_conv_stride, activation="relu6"))

        # stages
        in_channels = init_conv_channels
        for stage_id, stage_channels in enumerate(channels):
            stage = torch.nn.Sequential()
            for unit_id, unit_channels in enumerate(stage_channels):
                stride = strides[stage_id] if unit_id == 0 else 1
                mid_channel = mid_channels[stage_id][unit_id]
                stage.add_module("unit{}".format(unit_id + 1), LinearBottleneck(in_channels=in_channels, mid_channels=mid_channel, out_channels=unit_channels, stride=stride))
                in_channels = unit_channels
            self.backbone.add_module("stage{}".format(stage_id + 1), stage)

        self.backbone.add_module("final_conv", conv1x1_block(in_channels=in_channels, out_channels=final_conv_channels, activation="relu6"))
        self.backbone.add_module("global_pool", torch.nn.AdaptiveAvgPool2d(output_size=1))

        # classifier
        self.classifier = Classifier(in_channels=final_conv_channels, num_classes=num_classes)

        self.init_params()

    def init_params(self):
        # backbone
        for name, module in self.backbone.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)

        # classifier
        self.classifier.init_params()

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


class MobileNetV3(torch.nn.Module):
    """
    Class for constructing MobileNetsV3.
    
    If you are in doubt, please use the high-level function `get_mobilenet` to
    obtain ready-to-use models.
    """
    def __init__(self,
                 num_classes,
                 init_conv_channels,
                 init_conv_stride,
                 final_conv_channels,
                 final_conv_se,
                 channels,
                 mid_channels,
                 strides,
                 se_units,
                 kernel_sizes,
                 activations,
                 dropout_rate=0.0,
                 in_channels=3,
                 in_size=(224, 224),
                 use_data_batchnorm=True):
        super().__init__()
        self.use_data_batchnorm = use_data_batchnorm
        self.in_size = in_size
        self.dropout_rate = dropout_rate

        self.backbone = torch.nn.Sequential()

        # data batchnorm
        if self.use_data_batchnorm:
            self.backbone.add_module("data_bn", torch.nn.BatchNorm2d(num_features=in_channels))

        # init conv
        self.backbone.add_module("init_conv", conv3x3_block(in_channels=in_channels, out_channels=init_conv_channels, stride=init_conv_stride, activation="hswish"))

        # stages
        in_channels = init_conv_channels
        for stage_id, stage_channels in enumerate(channels):
            stage = torch.nn.Sequential()
            for unit_id, unit_channels in enumerate(stage_channels):
                stride = strides[stage_id] if unit_id == 0 else 1
                mid_channel = mid_channels[stage_id][unit_id]
                use_se=se_units[stage_id][unit_id] == 1
                kernel_size = kernel_sizes[stage_id]
                activation = activations[stage_id]
                stage.add_module("unit{}".format(unit_id + 1), LinearBottleneck(in_channels=in_channels, mid_channels=mid_channel, out_channels=unit_channels, stride=stride, activation=activation, use_se=use_se, kernel_size=kernel_size))
                in_channels = unit_channels
            self.backbone.add_module("stage{}".format(stage_id + 1), stage)

        self.backbone.add_module("final_conv1", conv1x1_block(in_channels=in_channels, out_channels=final_conv_channels[0], activation="hswish"))
        in_channels = final_conv_channels[0]
        if final_conv_se:
            self.backbone.add_module("final_se", SEUnit(channels=in_channels, squeeze_factor=4, squeeze_activation="relu", excite_activation="hsigmoid"))
        self.backbone.add_module("final_pool", torch.nn.AdaptiveAvgPool2d(output_size=1))
        if len(final_conv_channels) > 1:
            self.backbone.add_module("final_conv2", conv1x1_block(in_channels=in_channels, out_channels=final_conv_channels[1], activation="hswish", use_bn=False))
            in_channels = final_conv_channels[1]
        if  self.dropout_rate != 0.0:
            self.backbone.add_module("final_dropout", torch.nn.Dropout(dropout_rate))

        # classifier
        self.classifier = Classifier(in_channels=in_channels, num_classes=num_classes)

        self.init_params()

    def init_params(self):
        # backbone
        for name, module in self.backbone.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)

        # classifier
        self.classifier.init_params()

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


###
#%% model definitions
###


def build_mobilenet_v1(num_classes, width_multiplier=1.0, cifar=False):
    """
    Construct a MobileNetV1 from the given set of parameters.
    
    If you are in doubt, please use the high-level function `get_mobilenet` to
    obtain ready-to-use models.
    
    Args:
        num_classes (int): Number of classes for the classification layer.
        width_multiplier (float): Multiplier for the number of channels.
        cifar (bool): if `True`, make the model suitable for the CIFAR10/100
            datasets. Otherwise, the model will be suited for ImageNet and
            fine-grained datasets.
        
    Returns:
        The constructed MobileNetV1.
    """

    init_conv_channels = 32
    channels = [[64], [128, 128], [256, 256], [512, 512, 512, 512, 512, 512], [1024, 1024]]

    if cifar:
        in_size = (32, 32)
        init_conv_stride = 1
        strides = [1, 1, 2, 2, 2]
    else:
        in_size = (224, 224)
        init_conv_stride = 2
        strides = [1, 2, 2, 2, 2]

    if width_multiplier != 1.0:
        channels = [[int(unit * width_multiplier) for unit in stage] for stage in channels]
        init_conv_channels = int(init_conv_channels * width_multiplier)

    return MobileNetV1(num_classes=num_classes,
                       init_conv_channels=init_conv_channels,
                       init_conv_stride=init_conv_stride,
                       channels=channels,
                       strides=strides,
                       in_size=in_size)


def build_mobilenet_v2(num_classes, width_multiplier=1.0, cifar=False):
    """
    Construct a MobileNetV2 from the given set of parameters.
    
    If you are in doubt, please use the high-level function `get_mobilenet` to
    obtain ready-to-use models.
    
    Args:
        num_classes (int): Number of classes for the classification layer.
        width_multiplier (float): Multiplier for the number of channels.
        cifar (bool): if `True`, make the model suitable for the CIFAR10/100
            datasets. Otherwise, the model will be suited for ImageNet and
            fine-grained datasets.
        
    Returns:
        The constructed MobileNetV2.
    """
    init_conv_channels = 32
    channels = [[16], [24, 24], [32, 32, 32], [64, 64, 64, 64, 96, 96, 96], [160, 160, 160, 320]]
    mid_channels = [[32], [96, 144], [144, 192, 192], [192, 384, 384, 384, 384, 576, 576], [576, 960, 960, 960]]
    final_conv_channels = 1280

    if cifar:
        in_size = (32, 32)
        init_conv_stride = 1
        strides = [1, 1, 2, 2, 2]
    else:
        in_size = (224, 224)
        init_conv_stride = 2
        strides = [1, 2, 2, 2, 2]

    if width_multiplier != 1.0:
        init_conv_channels = int(init_conv_channels * width_multiplier)
        channels = [[int(unit * width_multiplier) for unit in stage] for stage in channels]
        mid_channels = [[int(unit * width_multiplier) for unit in stage] for stage in mid_channels]
        if width_multiplier > 1.0:
            final_conv_channels = int(final_conv_channels * width_multiplier)

    return MobileNetV2(num_classes=num_classes,
                       init_conv_channels=init_conv_channels,
                       init_conv_stride=init_conv_stride,
                       channels=channels,
                       mid_channels=mid_channels,
                       strides=strides,
                       final_conv_channels=final_conv_channels,
                       in_size=in_size)


def build_mobilenet_v3(num_classes, version, width_multiplier=1.0, cifar=False, use_lightweight_head=True):
    """
    Construct a MobileNetV3 from the given set of parameters.
    
    If you are in doubt, please use the high-level function `get_mobilenet` to
    obtain ready-to-use models.
    
    Args:
        num_classes (int): Number of classes for the classification layer.
        version (str): can be `"small"` or `"large"` for MobileNetV3-small or
            MobileNetV3-large.
        width_multiplier (float): Multiplier for the number of channels.
        cifar (bool): if `True`, make the model suitable for the CIFAR10/100
            datasets. Otherwise, the model will be suited for ImageNet and
            fine-grained datasets.
        use_lightweight_head (bool): If `True`, use a smaller head than
            originally defined to reduce model complexity.
        
    Returns:
        The constructed MobileNetV3.
    """
    in_size = (224, 224)
    init_conv_channels = 16
    init_conv_stride = 2
    dropout_rate = 0.0

    if version == "small":
        channels = [[16], [24, 24], [40, 40, 40, 48, 48], [96, 96, 96]]
        mid_channels = [[16], [72, 88], [96, 240, 240, 120, 144], [288, 576, 576]]
        strides = [2, 2, 2, 2]
        kernel_sizes = [3, 3, 5, 5]
        activations = ["relu", "relu", "hswish", "hswish"]
        se_units = [[1], [0, 0], [1, 1, 1, 1, 1], [1, 1, 1]]
        if use_lightweight_head:
            final_conv_channels = [576]
        else:
            final_conv_channels = [576, 1024]
        final_conv_se = True
    elif version == "large":
        channels = [[16], [24, 24], [40, 40, 40], [80, 80, 80, 80, 112, 112], [160, 160, 160]]
        mid_channels = [[16], [64, 72], [72, 120, 120], [240, 200, 184, 184, 480, 672], [672, 960, 960]]
        strides = [1, 2, 2, 2, 2]
        kernel_sizes = [3, 3, 5, 3, 5]
        activations = ["relu", "relu", "relu", "hswish", "hswish"]
        se_units = [[0], [0, 0], [1, 1, 1], [0, 0, 0, 0, 1, 1], [1, 1, 1]]
        if use_lightweight_head:
            final_conv_channels = [960]
        else:
            final_conv_channels = [960, 1280]
        final_conv_se = False
    else:
        raise NotImplementedError

    if cifar:
        in_size = (32, 32)
        init_conv_stride = 1
        strides = [1, 2, 2, 2] if version == "small" else [1, 1, 2, 2, 2]

    if width_multiplier != 1.0:
        channels = [[int(unit * width_multiplier) for unit in stage] for stage in channels]
        mid_channels = [[int(unit * width_multiplier) for unit in stage] for stage in mid_channels]
        init_conv_channels = int(init_conv_channels * width_multiplier)
        if width_multiplier > 1.0:
            final_conv_channels[0] = int(final_conv_channels[0] * width_multiplier)

    return MobileNetV3(num_classes=num_classes,
                       init_conv_channels=init_conv_channels,
                       init_conv_stride=init_conv_stride,
                       final_conv_channels=final_conv_channels,
                       final_conv_se=final_conv_se,
                       channels=channels,
                       mid_channels=mid_channels,
                       strides=strides,
                       se_units=se_units,
                       kernel_sizes=kernel_sizes,
                       activations=activations,
                       dropout_rate=dropout_rate,
                       in_size=in_size)


def transform_mobilenetv1(net):
    """
    Transform a given MobileNetV1 `net` to its corresponding BSConv-U variant.
    
    For details, see paper Sect. 4.1.
    
    Args:
        net (PyTorch model): The MobileNetV1 to be transformed.
        
    Returns:
        The transformed MobileNet as BSConv-U variant.
    """
    
    class DSCFilter(bsconv.pytorch.replacers.ModuleFilter):
        def apply(self, module, name, full_name):
            return isinstance(module, DepthwiseSeparableConvBlock)

    class DSCToBSConvUTransformer(bsconv.pytorch.replacers.ModuleTransformer):
        def __init__(self, with_bn, bn_kwargs):
            self.with_bn = with_bn
            self.bn_kwargs = bn_kwargs

        def apply(self, module, name, full_name):
            return torch.nn.Sequential(
                bsconv.pytorch.modules.BSConvU(
                        in_channels=module.conv_dw.conv.in_channels,
                        out_channels=module.conv_pw.conv.out_channels,
                        kernel_size=module.conv_dw.conv.kernel_size,
                        stride=module.conv_dw.conv.stride,
                        padding=module.conv_dw.conv.padding,
                        dilation=module.conv_dw.conv.dilation,
                        bias=module.conv_dw.conv.bias is not None,
                        padding_mode=module.conv_dw.conv.padding_mode,
                        with_bn=self.with_bn,
                        bn_kwargs=self.bn_kwargs,
                    ),
                    torch.nn.BatchNorm2d(num_features=module.conv_pw.conv.out_channels),
                    torch.nn.ReLU(inplace=True)
                )

    class MobileNetV1Transformer(bsconv.pytorch.replacers.ModuleReplacer):
        def __init__(self, with_bn=False, bn_kwargs=None, **kwargs):
            super().__init__(**kwargs)
            self.add_rule(
                DSCFilter(),
                DSCToBSConvUTransformer(with_bn=with_bn, bn_kwargs=bn_kwargs),
            )

    transformer = MobileNetV1Transformer()
    net = transformer.apply(net)
    return net


def transform_mobilenetv2(net):
    """
    Transform a given MobileNetV2 or MobileNetV3 `net` to its corresponding
    BSConv-S (p=1/6) variant.
    
    For details, see paper, Sect. 4.2.
    
    Args:
        net (PyTorch model): The MobileNetV2 or MobileNetV3 to be transformed.
        
    Returns:
        The transformed MobileNet as BSConv-S (p=1/6) variant.
    """
    class LinearBottleneckFilter(bsconv.pytorch.replacers.ModuleFilter):
        def apply(self, module, name, full_name):
            return isinstance(module, LinearBottleneck)

    class LinearBottleneckToBSConvUTransformer(bsconv.pytorch.replacers.ModuleTransformer):
        def __init__(self, with_bn, bn_kwargs):
            self.with_bn = with_bn
            self.bn_kwargs = bn_kwargs

        def apply(self, module, name, full_name):
            def _reg_loss(self):
                W = module.conv3.conv.weight[:, :, 0, 0]
                WWt = torch.mm(W, torch.transpose(W, 0, 1))
                I = torch.eye(WWt.shape[0], device=WWt.device)
                return torch.norm(WWt - I, p="fro")
            module._reg_loss = types.MethodType(_reg_loss, module)
            return module

    class MobileNetV2Transformer(bsconv.pytorch.replacers.ModuleReplacer):
        def __init__(self, with_bn=False, bn_kwargs=None, **kwargs):
            super().__init__(**kwargs)
            self.add_rule(
                LinearBottleneckFilter(),
                LinearBottleneckToBSConvUTransformer(with_bn=with_bn, bn_kwargs=bn_kwargs),
            )
            self.add_rule(
                bsconv.pytorch.replacers.ModelFilter(),
                bsconv.pytorch.replacers.RegularizationMethodTransformer(),
            )

    transformer = MobileNetV2Transformer()
    net = transformer.apply(net)
    return net


def get_mobilenet(architecture, num_classes):
    """
    Return a MobileNet specified by the string `architecture`.
    
    Args:
        architecture (str): MobileNet architecture specification
            (see below for examples).
        num_classes (int): Number of classes for the classification layer.
        
    Returns:
        The constructed MobileNet.
    
    Examples for `architecture`:
        # MobileNetsV1
        mobilenetv1_w1
        mobilenetv1_w3d4
        mobilenetv1_w1d2
        mobilenetv1_w1d4
        
        # MobileNetsV1 + BSconv-U
        mobilenetv1_w1_bsconvu
        mobilenetv1_w3d4_bsconvu
        mobilenetv1_w1d2_bsconvu
        mobilenetv1_w1d4_bsconvu
        
        # MobileNetsV2
        mobilenetv2_w1
        mobilenetv2_w3d4
        mobilenetv2_w1d2
        mobilenetv2_w1d4
        
        # MobileNetsV2 + BSConv-S (p=1/6)
        mobilenetv2_w1_bsconvs_p1d6
        mobilenetv2_w3d4_bsconvs_p1d6
        mobilenetv2_w1d2_bsconvs_p1d6
        mobilenetv2_w1d4_bsconvs_p1d6
        
        # MobileNetsV3-small
        mobilenetv3_small_w1
        mobilenetv3_small_w3d4
        mobilenetv3_small_w1d2
        mobilenetv3_small_w7d20
        
        # MobileNetsV3-small + BSConv-S (p=1/6)
        mobilenetv3_small_w1_bsconvs_p1d6
        mobilenetv3_small_w3d4_bsconvs_p1d6
        mobilenetv3_small_w1d2_bsconvs_p1d6
        mobilenetv3_small_w7d20_bsconvs_p1d6
        
        # MobileNetsV3-large
        mobilenetv3_large_w1
        mobilenetv3_large_w3d4
        mobilenetv3_large_w1d2
        mobilenetv3_large_w7d20
        
        # MobileNetsV3-large + BSConv-S (p=1/6)
        mobilenetv3_large_w1_bsconvs_p1d6
        mobilenetv3_large_w3d4_bsconvs_p1d6
        mobilenetv3_large_w1d2_bsconvs_p1d6
        mobilenetv3_large_w7d20_bsconvs_p1d6
        
        # CIFAR MobileNetsV1
        cifar_mobilenetv1_w1
        cifar_mobilenetv1_w3d4
        cifar_mobilenetv1_w1d2
        cifar_mobilenetv1_w1d4
        
        # CIFAR MobileNetsV1 + BSconv-U
        cifar_mobilenetv1_w1_bsconvu
        cifar_mobilenetv1_w3d4_bsconvu
        cifar_mobilenetv1_w1d2_bsconvu
        cifar_mobilenetv1_w1d4_bsconvu
        
        # CIFAR MobileNetsV2
        cifar_mobilenetv2_w1
        cifar_mobilenetv2_w3d4
        cifar_mobilenetv2_w1d2
        cifar_mobilenetv2_w1d4
        
        # CIFAR MobileNetsV2 + BSConv-S (p=1/6)
        cifar_mobilenetv2_w1_bsconvs_p1d6
        cifar_mobilenetv2_w3d4_bsconvs_p1d6
        cifar_mobilenetv2_w1d2_bsconvs_p1d6
        cifar_mobilenetv2_w1d4_bsconvs_p1d6
        
        # CIFAR MobileNetsV3-small
        cifar_mobilenetv3_small_w1
        cifar_mobilenetv3_small_w3d4
        cifar_mobilenetv3_small_w1d2
        cifar_mobilenetv3_small_w7d20
        
        # CIFAR MobileNetsV3-small + BSConv-S (p=1/6)
        cifar_mobilenetv3_small_w1_bsconvs_p1d6
        cifar_mobilenetv3_small_w3d4_bsconvs_p1d6
        cifar_mobilenetv3_small_w1d2_bsconvs_p1d6
        cifar_mobilenetv3_small_w7d20_bsconvs_p1d6
        
        # CIFAR MobileNetsV3-large
        cifar_mobilenetv3_large_w1
        cifar_mobilenetv3_large_w3d4
        cifar_mobilenetv3_large_w1d2
        cifar_mobilenetv3_large_w7d20
        
        # CIFAR MobileNetsV3-large + BSConv-S (p=1/6)
        cifar_mobilenetv3_large_w1_bsconvs_p1d6
        cifar_mobilenetv3_large_w3d4_bsconvs_p1d6
        cifar_mobilenetv3_large_w1d2_bsconvs_p1d6
        cifar_mobilenetv3_large_w7d20_bsconvs_p1d6
    
    See https://github.com/zeiss-microscopy/BSConv/blob/master/bsconv/pytorch/README.md
    for a list of possible architectures.
    """
    
    # parse architecture string
    pattern = r"^(?P<cifar>cifar_)?mobilenet(?P<version>v1|v2|v3_small|v3_large)_w(?P<width_numerator>[0-9]+)(d(?P<width_denominator>[0-9]+))?(_(?P<bsconv_variant>bsconvu|bsconvs_p1d6))?$"
    match = re.match(pattern, architecture)
    if match is None:
        raise ValueError("Model architecture '{}' is not supported".format(architecture))
    cifar = (match.group("cifar") is not None)
    version = match.group("version")
    width_numerator = match.group("width_numerator")
    width_denominator = match.group("width_denominator")
    bsconv_variant = match.group("bsconv_variant")
    
    # determine the width_multiplier
    if width_denominator is None:
        width_multiplier = float(width_numerator)
    else:
        width_multiplier = float(width_numerator) / float(width_denominator)
    
    # base net
    if version == "v1":
        model = build_mobilenet_v1(num_classes=num_classes, width_multiplier=width_multiplier, cifar=cifar)
    elif version == "v2":
        model = build_mobilenet_v2(num_classes=num_classes, width_multiplier=width_multiplier, cifar=cifar)
    elif version == "v3_small":
        model = build_mobilenet_v3(num_classes=num_classes, version="small", width_multiplier=width_multiplier, cifar=cifar)
    elif version == "v3_large":
        model = build_mobilenet_v3(num_classes=num_classes, version="large", width_multiplier=width_multiplier, cifar=cifar)
    
    # apply BSConv
    if bsconv_variant is None:
        pass
    elif bsconv_variant == "bsconvu":
        if version == "v1":
            model = transform_mobilenetv1(model)
        else:
            raise ValueError("For MobileNetV1, only BSConv-U is supported")
    elif bsconv_variant.startswith("bsconvs_p1d6"):
        if version in ("v2", "v3_small", "v3_large"):
            model = transform_mobilenetv2(model)
        else:
            raise ValueError("For MobileNetV2/V3, only BSConv-S (p=1/6) is supported")

    return model
