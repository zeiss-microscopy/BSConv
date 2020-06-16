from .resnet import *
from .mobilenet import *

models = {
    'resnet10': resnet10,
    'resnet18': resnet18,
    'resnet26': resnet26,
    'resnet34': resnet34,
    'resnet68': resnet68,
    'resnet102': resnet102,

    'resnet10_bsconvu': resnet10_bsconvu,
    'resnet18_bsconvu': resnet18_bsconvu,
    'resnet26_bsconvu': resnet26_bsconvu,
    'resnet34_bsconvu': resnet34_bsconvu,
    'resnet68_bsconvu': resnet68_bsconvu,
    'resnet102_bsconvu': resnet102_bsconvu,

    'mobilenetv1_w1': mobilenetv1_w1,
    'mobilenetv2_w1': mobilenetv2_w1,
    'mobilenetv3_small_w1': mobilenetv3_small_w1,
    'mobilenetv3_large_w1': mobilenetv3_large_w1,

    'mobilenetv1_w1_bsconvu': mobilenetv1_w1_bsconvu,
    'mobilenetv2_w1_bsconvs': mobilenetv2_w1_bsconvs,
    'mobilenetv3_small_w1_bsconvs': mobilenetv3_small_w1_bsconvs,
    'mobilenetv3_large_w1_bsconvs': mobilenetv3_large_w1_bsconvs,

    'cifar_resnet20': cifar_resnet20,
    'cifar_resnet56': cifar_resnet56,
    'cifar_resnet110': cifar_resnet110,

    'cifar_resnet20_bsconvu': cifar_resnet20_bsconvu,
    'cifar_resnet56_bsconvu': cifar_resnet56_bsconvu,
    'cifar_resnet110_bsconvu': cifar_resnet110_bsconvu,

    'cifar_wrn40_3': cifar_wrn40_3,
    'cifar_wrn40_8': cifar_wrn40_8,

    'cifar_wrn40_3_bsconvu': cifar_wrn40_3_bsconvu,
    'cifar_wrn40_8_bsconvu': cifar_wrn40_8_bsconvu,

    'cifar_mobilenetv1_w1': cifar_mobilenetv1_w1,
    'cifar_mobilenetv2_w1': cifar_mobilenetv2_w1,
    'cifar_mobilenetv3_small_w1': cifar_mobilenetv3_small_w1,
    'cifar_mobilenetv3_large_w1': cifar_mobilenetv3_large_w1,

    'cifar_mobilenetv1_w1_bsconvu': cifar_mobilenetv1_w1_bsconvu,
    'cifar_mobilenetv2_w1_bsconvs': cifar_mobilenetv2_w1_bsconvs,
    'cifar_mobilenetv3_small_w1_bsconvs': cifar_mobilenetv3_small_w1_bsconvs,
    'cifar_mobilenetv3_large_w1_bsconvs': cifar_mobilenetv3_large_w1_bsconvs
}


def get_model(architecture, **kwargs):
    architecture = architecture.lower()
    if architecture not in models:
        raise NotImplementedError("Model: {} not supported".format(architecture))
    model = models[architecture](**kwargs)
    return model
