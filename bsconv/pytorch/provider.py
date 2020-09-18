from .resnet import get_resnet
from .mobilenet import *

models = {
    'mobilenetv1_w1': mobilenetv1_w1,
    'mobilenetv2_w1': mobilenetv2_w1,
    'mobilenetv3_small_w1': mobilenetv3_small_w1,
    'mobilenetv3_large_w1': mobilenetv3_large_w1,

    'mobilenetv1_w1_bsconvu': mobilenetv1_w1_bsconvu,
    'mobilenetv2_w1_bsconvs': mobilenetv2_w1_bsconvs,
    'mobilenetv3_small_w1_bsconvs': mobilenetv3_small_w1_bsconvs,
    'mobilenetv3_large_w1_bsconvs': mobilenetv3_large_w1_bsconvs,

    'cifar_mobilenetv1_w1': cifar_mobilenetv1_w1,
    'cifar_mobilenetv2_w1': cifar_mobilenetv2_w1,
    'cifar_mobilenetv3_small_w1': cifar_mobilenetv3_small_w1,
    'cifar_mobilenetv3_large_w1': cifar_mobilenetv3_large_w1,

    'cifar_mobilenetv1_w1_bsconvu': cifar_mobilenetv1_w1_bsconvu,
    'cifar_mobilenetv2_w1_bsconvs': cifar_mobilenetv2_w1_bsconvs,
    'cifar_mobilenetv3_small_w1_bsconvs': cifar_mobilenetv3_small_w1_bsconvs,
    'cifar_mobilenetv3_large_w1_bsconvs': cifar_mobilenetv3_large_w1_bsconvs,
}


def get_model(architecture, **kwargs):
    architecture = architecture.lower()

    if architecture in models:
        # get MobileNets using lookup table above
        model = models[architecture](**kwargs)
    elif ("resnet" in architecture) or ("wrn" in architecture):
        # get ResNets by parsing the architecture string
        model = get_resnet(architecture=architecture, **kwargs)
    return model
