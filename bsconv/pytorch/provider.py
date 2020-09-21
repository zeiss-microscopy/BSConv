import bsconv.pytorch.mobilenet
import bsconv.pytorch.resnet


def get_model(architecture, **kwargs):
    # parse architecture string using the appropriate parser
    architecture = architecture.lower()
    if "mobilenet" in architecture:
        model = bsconv.pytorch.mobilenet.get_mobilenet(architecture=architecture, **kwargs)
    elif ("resnet" in architecture) or ("wrn" in architecture):
        model = bsconv.pytorch.resnet.get_resnet(architecture=architecture, **kwargs)
    else:
        raise ValueError("Model architecture '{}' is not supported".format(architecture))
    return model