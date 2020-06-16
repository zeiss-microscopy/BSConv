import types

import torch.nn
import torch.nn.init

import bsconv.pytorch
from bsconv.pytorch.common import conv1x1_block, conv3x3_block, conv3x3_dw_block, conv5x5_dw_block, SEUnit, Classifier


class DepthwiseSeparableConvBlock(torch.nn.Module):
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
        if final_conv_se:
            self.backbone.add_module("final_se", SEUnit(channels=final_conv_channels[0], squeeze_factor=4, squeeze_activation="relu", excite_activation="hsigmoid"))
        self.backbone.add_module("global_pool", torch.nn.AdaptiveAvgPool2d(output_size=1))
        self.backbone.add_module("final_conv2", conv1x1_block(in_channels=final_conv_channels[0], out_channels=final_conv_channels[1], activation="hswish", use_bn=False))

        # classifier
        self.classifier = Classifier(in_channels=final_conv_channels[1], num_classes=num_classes)

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

def build_mobilenet_v1(num_classes,
                       width_multiplier = 1.0,
                       cifar = False):

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


def build_mobilenet_v2(num_classes,
                       width_multiplier = 1.0,
                       cifar = False):

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

def build_mobilenet_v3(num_classes,
                       version,
                       width_multiplier = 1.0,
                       cifar = False):

    in_size = (224, 224)
    init_conv_channels = 16
    init_conv_stride = 2

    if version == "small":
        channels = [[16], [24, 24], [40, 40, 40, 48, 48], [96, 96, 96]]
        mid_channels = [[16], [72, 88], [96, 240, 240, 120, 144], [288, 576, 576]]
        strides = [2, 2, 2, 2]
        kernel_sizes = [3, 3, 5, 5]
        activations = ["relu", "relu", "hswish", "hswish"]
        se_units = [[1], [0, 0], [1, 1, 1, 1, 1], [1, 1, 1]]
        final_conv_channels = [576, 1024]
        final_conv_se = True
    elif version == "large":
        channels = [[16], [24, 24], [40, 40, 40], [80, 80, 80, 80, 112, 112], [160, 160, 160]]
        mid_channels = [[16], [64, 72], [72, 120, 120], [240, 200, 184, 184, 480, 672], [672, 960, 960]]
        strides = [1, 2, 2, 2, 2]
        kernel_sizes = [3, 3, 5, 3, 5]
        activations = ["relu", "relu", "relu", "hswish", "hswish"]
        se_units = [[0], [0, 0], [1, 1, 1], [0, 0, 0, 0, 1, 1], [1, 1, 1]]
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
                       in_size=in_size)


###
#%% bsconv replacer
###

def transform_mobilenetv1(net):

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

###
#%% baseline MobileNet CIFAR
###

def cifar_mobilenetv1_w1(num_classes):
    return build_mobilenet_v1(num_classes=num_classes, cifar=True)

def cifar_mobilenetv2_w1(num_classes):
    return build_mobilenet_v2(num_classes=num_classes, cifar=True)

def cifar_mobilenetv3_small_w1(num_classes):
    return build_mobilenet_v3(num_classes=num_classes, version="small", cifar=True)

def cifar_mobilenetv3_large_w1(num_classes):
    return build_mobilenet_v3(num_classes=num_classes, version="large", cifar=True)

###
#%% baseline MobileNet ImageNet
###

def mobilenetv1_w1(num_classes):
    return build_mobilenet_v1(num_classes=num_classes, cifar=False)

def mobilenetv2_w1(num_classes):
    return build_mobilenet_v2(num_classes=num_classes, cifar=False)

def mobilenetv3_small_w1(num_classes):
    return build_mobilenet_v3(num_classes=num_classes, version="small", cifar=False)

def mobilenetv3_large_w1(num_classes):
    return build_mobilenet_v3(num_classes=num_classes, version="large", cifar=False)

###
#%% bsconv MobileNet CIFAR
###

def cifar_mobilenetv1_w1_bsconvu(num_classes):
    net = cifar_mobilenetv1_w1(num_classes)
    return transform_mobilenetv1(net)

def cifar_mobilenetv2_w1_bsconvs(num_classes):
    net = cifar_mobilenetv2_w1(num_classes)
    return transform_mobilenetv2(net)

def cifar_mobilenetv3_small_w1_bsconvs(num_classes):
    net = cifar_mobilenetv3_small_w1(num_classes)
    return transform_mobilenetv2(net)

def cifar_mobilenetv3_large_w1_bsconvs(num_classes):
    net = cifar_mobilenetv3_large_w1(num_classes)
    return transform_mobilenetv2(net)


###
#%% bsconv MobileNet ImageNet
###

def mobilenetv1_w1_bsconvu(num_classes):
    net = mobilenetv1_w1(num_classes)
    return transform_mobilenetv1(net)

def mobilenetv2_w1_bsconvs(num_classes):
    net = mobilenetv2_w1(num_classes)
    return transform_mobilenetv2(net)

def mobilenetv3_small_w1_bsconvs(num_classes):
    net = mobilenetv3_small_w1(num_classes)
    return transform_mobilenetv2(net)

def mobilenetv3_large_w1_bsconvs(num_classes):
    net = mobilenetv3_large_w1(num_classes)
    return transform_mobilenetv2(net)
