import torch.nn
import torch.nn.init

import bsconv.pytorch
from bsconv.pytorch.common import conv1x1_block, conv3x3_block, conv7x7_block, Classifier

class InitUnitLarge(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.conv = conv7x7_block(in_channels=in_channels, out_channels=out_channels, stride=2)
        self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x

class InitUnitSmall(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 preact=False):
        super().__init__()

        self.conv = conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=1,
                use_bn=not preact,
                activation=None if preact else "relu")

    def forward(self, x):
        x = self.conv(x)
        return x

class PostActivation(torch.nn.Module):
    def __init__(self,
                 channels):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(num_features=channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        return x

class StandardUnit(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride):
        super().__init__()
        self.use_projection = (in_channels != out_channels) or (stride != 1)

        self.conv1 = conv3x3_block(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.conv2 = conv3x3_block(in_channels=out_channels, out_channels=out_channels, stride=1, activation=None)
        if self.use_projection:
            self.pojection = conv1x1_block(in_channels=in_channels, out_channels=out_channels, stride=stride, activation=None)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        if self.use_projection:
            residual = self.pojection(x)
        else:
            residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + residual
        x = self.relu(x)
        return x

class PreactUnit(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride):
        super().__init__()
        self.use_projection = (in_channels != out_channels) or (stride != 1)

        self.bn = torch.nn.BatchNorm2d(num_features=in_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv1 = conv3x3_block(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.conv2 = conv3x3_block(in_channels=out_channels, out_channels=out_channels, use_bn=False, activation=None)
        if self.use_projection:
            self.pojection = conv1x1_block(in_channels=in_channels, out_channels=out_channels, stride=stride, use_bn=False, activation=None)

    def forward(self, x):
        if self.use_projection:
            x = self.bn(x)
            x = self.relu(x)
            residual = self.pojection(x)
        else:
            residual = x
            x = self.bn(x)
            x = self.relu(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + residual
        return x

class ResNet(torch.nn.Module):
    def __init__(self,
                 channels,
                 num_classes,
                 preact=False,
                 init_unit_channels=64,
                 use_init_unit_large=True,
                 in_channels=3,
                 in_size=(224, 224),
                 use_data_batchnorm=True):
        super().__init__()
        self.use_data_batchnorm = use_data_batchnorm
        self.use_init_unit_large = use_init_unit_large
        self.in_size = in_size

        self.backbone = torch.nn.Sequential()

        # data batchnorm
        if self.use_data_batchnorm:
            self.backbone.add_module("data_bn", torch.nn.BatchNorm2d(num_features=in_channels))

        # init unit
        if self.use_init_unit_large:
            self.backbone.add_module("init_unit", InitUnitLarge(in_channels=in_channels, out_channels=init_unit_channels))
        else:
            self.backbone.add_module("init_unit", InitUnitSmall(in_channels=in_channels, out_channels=init_unit_channels, preact=preact))

        # stages
        in_channels = init_unit_channels
        for stage_id, stage_channels in enumerate(channels):
            stage = torch.nn.Sequential()
            for unit_id, unit_channels in enumerate(stage_channels):
                stride = 2 if (unit_id == 0) and (stage_id != 0) else 1
                if preact:
                    stage.add_module("unit{}".format(unit_id + 1), PreactUnit(in_channels=in_channels, out_channels=unit_channels, stride=stride))
                else:
                    stage.add_module("unit{}".format(unit_id + 1), StandardUnit(in_channels=in_channels, out_channels=unit_channels, stride=stride))
                in_channels = unit_channels
            self.backbone.add_module("stage{}".format(stage_id + 1), stage)
        if preact:
            self.backbone.add_module("final_activation", PostActivation(in_channels))
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

def build_resnet(num_classes,
                 units_per_stage,
                 width_multiplier = 1.0,
                 preact = False,
                 cifar = False):
    if cifar:
        # todo: check whether init_channels have to be scaled based on width_factor (guess resnet: yes; wrn: no)
        init_unit_channels = 16
        channels_in_stage = [16, 32, 64]
        channels = [[int(ch * width_multiplier)] * rep for (ch, rep) in zip(channels_in_stage, units_per_stage)]
        use_init_unit_large = False
        in_size = (32, 32)
    else:
        init_unit_channels = 64
        channels_in_stage = [64, 128, 256, 512]
        channels = [[int(ch * width_multiplier)] * rep for (ch, rep) in zip(channels_in_stage, units_per_stage)]
        use_init_unit_large = True
        in_size = (224, 224)

    net = ResNet(channels=channels,
                 num_classes=num_classes,
                 preact=preact,
                 init_unit_channels=init_unit_channels,
                 use_init_unit_large=use_init_unit_large,
                 in_size=in_size)
    return net

###
#%% bsconv replacer
###

def replace_bsconv_u(net):
    replacer = bsconv.pytorch.BSConvU_Replacer()
    net = replacer.apply(net)
    return net

def replace_bsconv_s(net, p=0.25):
    replacer = bsconv.pytorch.BSConvS_Replacer(p=p)
    net = replacer.apply(net)
    return net


###
#%% ResNets CIFAR
###

def cifar_resnet20(num_classes):
    return build_resnet(num_classes=num_classes, units_per_stage=[3, 3, 3], cifar=True)

def cifar_resnet56(num_classes):
    return build_resnet(num_classes=num_classes, units_per_stage=[9, 9, 9], cifar=True)

def cifar_resnet110(num_classes):
    return build_resnet(num_classes=num_classes, units_per_stage=[18, 18, 18], cifar=True)


def cifar_resnet20_bsconvu(num_classes):
    net = cifar_resnet20(num_classes)
    return replace_bsconv_u(net)

def cifar_resnet56_bsconvu(num_classes):
    net = cifar_resnet56(num_classes)
    return replace_bsconv_u(net)

def cifar_resnet110_bsconvu(num_classes):
    net = cifar_resnet110(num_classes)
    return replace_bsconv_u(net)


###
#%% ResNets ImageNet
###

def resnet10(num_classes):
    return build_resnet(num_classes=num_classes, units_per_stage=[1, 1, 1, 1], cifar=False)

def resnet18(num_classes):
    return build_resnet(num_classes=num_classes, units_per_stage=[2, 2, 2, 2], cifar=False)

def resnet26(num_classes):
    return build_resnet(num_classes=num_classes, units_per_stage=[3, 3, 3, 3], cifar=False)

def resnet34(num_classes):
    return build_resnet(num_classes=num_classes, units_per_stage=[3, 4, 6, 3], cifar=False)

def resnet68(num_classes):
    return build_resnet(num_classes=num_classes, units_per_stage=[3, 4, 23, 3], cifar=False)

def resnet102(num_classes):
    return build_resnet(num_classes=num_classes, units_per_stage=[3, 8, 36, 3], cifar=False)


def resnet10_bsconvu(num_classes):
    net = resnet10(num_classes)
    return replace_bsconv_u(net)

def resnet18_bsconvu(num_classes):
    net = resnet18(num_classes)
    return replace_bsconv_u(net)

def resnet26_bsconvu(num_classes):
    net = resnet26(num_classes)
    return replace_bsconv_u(net)

def resnet34_bsconvu(num_classes):
    net = resnet34(num_classes)
    return replace_bsconv_u(net)

def resnet68_bsconvu(num_classes):
    net = resnet68(num_classes)
    return replace_bsconv_u(net)

def resnet102_bsconvu(num_classes):
    net = resnet102(num_classes)
    return replace_bsconv_u(net)


###
#%% WideResNets CIFAR
###

def cifar_wrn40_3(num_classes):
    return build_resnet(num_classes=num_classes, units_per_stage=[6, 6, 6], preact=True, width_multiplier=3.0, cifar=True)

def cifar_wrn40_8(num_classes):
    return build_resnet(num_classes=num_classes, units_per_stage=[6, 6, 6], preact=True, width_multiplier=8.0, cifar=True)


def cifar_wrn40_3_bsconvu(num_classes):
    net = cifar_wrn40_3(num_classes)
    return replace_bsconv_u(net)

def cifar_wrn40_8_bsconvu(num_classes):
    net = cifar_wrn40_8(num_classes)
    return replace_bsconv_u(net)