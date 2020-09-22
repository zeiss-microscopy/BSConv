BSConv Implementation for PyTorch
=================================

We provide several ways to obtain BSConv-models:

* [**Ready-to-use model definitions**](#ready-to-use-model-definitions)
    * suited for models which require special considerations when transforming them to BSConv variants (e.g., MobileNets, ResNets-50 and larger, EfficientNets, etc.)
    * can be used to reproduce the results reported in the paper
* [**BSConv as general drop-in replacement**](#bsconv-as-general-drop-in-replacement)
    * replaces convolutions in existing model definitions by BSConv
    * suited for CNNs which use regular convolutions (without groups, bottlenecks, etc.), e.g. ResNets (up to ResNet-34), VGGs, DenseNets, etc.
    * for other models (e.g. MobileNets, ResNets-50 and larger, EfficientNets, etc.) use our ready-to-use model definitions (see above)
* [**BSConv PyTorch modules**](#bsconv-pytorch-modules)
    * these modules can be used instead of regular convolution layers
    * suited for building custom models from scratch


Ready-to-Use Model Definitions
------------------------------

Ready-to-use model definitions (both for baseline models and their BSConv variants) can be obtained via
```python
import bsconv.pytorch
model = bsconv.pytorch.get_model('cifar_resnet110_bsconvu', num_classes=100)
```

Currently, available architectures are:

    # ResNets
    resnet(10|18|26|34|68|102)
    resnet(10|18|26|34|68|102)_bsconvu
    resnet(10|18|26|34|68|102)_bsconvs_pXdY            # BSConv-S p=X/Y

    # Pre-Activation ResNets (aka ResNetsV2)
    preresnet(10|18|26|34|68|102)
    preresnet(10|18|26|34|68|102)_bsconvu
    preresnet(10|18|26|34|68|102)_bsconvs_pXdY         # BSConv-S p=X/Y

    # MobileNetsV1 (support BSConv-U)
    mobilenetv1_wXdY                                   # width w=X/Y
    mobilenetv1_wXdY_bsconvu                           # width w=X/Y
    
    # MobileNetsV2 (support BSConv-S with p=1/6)
    mobilenetv2_wXdY                                   # width w=X/Y
    mobilenetv2_wXdY_bsconvs_p1d6                      # width w=X/Y
    
    # MobileNetsV3 (support BSConv-S with p=1/6)
    mobilenetv3_small_wXdY                             # width w=X/Y
    mobilenetv3_large_wXdY                             # width w=X/Y
    mobilenetv3_small_wXdY_bsconvs_p1d6                # width w=X/Y
    mobilenetv3_large_wXdY_bsconvs_p1d6                # width w=X/Y

    # CIFAR ResNets
    cifar_resnet(20|56|110|302|602)
    cifar_resnet(20|56|110|302|602)_bsconvu
    cifar_resnet(20|56|110|302|602)_bsconvs_pXdY       # BSConv-S p=X/Y
    
    # CIFAR Pre-Activation ResNets (aka ResNetsV2)
    cifar_preresnet(20|56|110|302|602)
    cifar_preresnet(20|56|110|302|602)_bsconvu
    cifar_preresnet(20|56|110|302|602)_bsconvs_pXdY    # BSConv-S p=X/Y
    
    # CIFAR WideResNets
    cifar_wrn(16|28|40)_(1|2|3|...)
    cifar_wrn(16|28|40)_(1|2|3|...)_bsconvu
    cifar_wrn(16|28|40)_(1|2|3|...)_bsconvs_pXdY       # BSConv-S p=X/Y
    
    # CIFAR MobileNetsV1 (support BSConv-U)
    cifar_mobilenetv1_wXdY                             # width w=X/Y
    cifar_mobilenetv1_wXdY_bsconvu                     # width w=X/Y
    
    # CIFAR MobileNetsV2 (support BSConv-S with p=1/6)
    cifar_mobilenetv2_wXdY                             # width w=X/Y
    cifar_mobilenetv2_wXdY_bsconvs_p1d6                # width w=X/Y
    
    # CIFAR MobileNetsV3 (support BSConv-S with p=1/6)
    cifar_mobilenetv3_small_wXdY                       # width w=X/Y
    cifar_mobilenetv3_large_wXdY                       # width w=X/Y
    cifar_mobilenetv3_small_wXdY_bsconvs_p1d6          # width w=X/Y
    cifar_mobilenetv3_large_wXdY_bsconvs_p1d6          # width w=X/Y
    
Concrete examples (i.e., architecture strings which can be passed to `bsconv.pytorch.get_model`) are:

    # ResNets
    resnet10
    resnet18
    resnet26
    resnet34
    resnet68
    resnet102
    
    # ResNets + BSConv-U
    resnet10_bsconvu
    resnet18_bsconvu
    resnet26_bsconvu
    resnet34_bsconvu
    resnet68_bsconvu
    resnet102_bsconvu
    
    # ResNets + BSConv-S (p=1/4)
    resnet10_bsconvs_p1d4
    resnet18_bsconvs_p1d4
    resnet26_bsconvs_p1d4
    resnet34_bsconvs_p1d4
    resnet68_bsconvs_p1d4
    resnet102_bsconvs_p1d4
    
    # ResNets + BSConv-S (p=1/8)
    resnet10_bsconvs_p1d8
    resnet18_bsconvs_p1d8
    resnet26_bsconvs_p1d8
    resnet34_bsconvs_p1d8
    resnet68_bsconvs_p1d8
    resnet102_bsconvs_p1d8

    ################################################################################

    # Pre-Activation ResNets (aka ResNetsV2)
    preresnet10
    preresnet18
    preresnet26
    preresnet34
    preresnet68
    preresnet102
    
    # Pre-Activation ResNets (aka ResNetsV2) + BSConv-U
    preresnet10_bsconvu
    preresnet18_bsconvu
    preresnet26_bsconvu
    preresnet34_bsconvu
    preresnet68_bsconvu
    preresnet102_bsconvu
    
    # Pre-Activation ResNets (aka ResNetsV2) + BSConv-S (p=1/4)
    preresnet10_bsconvs_p1d4
    preresnet18_bsconvs_p1d4
    preresnet26_bsconvs_p1d4
    preresnet34_bsconvs_p1d4
    preresnet68_bsconvs_p1d4
    preresnet102_bsconvs_p1d4
    
    # Pre-Activation ResNets (aka ResNetsV2) + BSConv-S (p=1/8)
    preresnet10_bsconvs_p1d8
    preresnet18_bsconvs_p1d8
    preresnet26_bsconvs_p1d8
    preresnet34_bsconvs_p1d8
    preresnet68_bsconvs_p1d8
    preresnet102_bsconvs_p1d8

    ################################################################################

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

    ################################################################################

    # CIFAR ResNets
    cifar_resnet20
    cifar_resnet56
    cifar_resnet110
    cifar_resnet302
    cifar_resnet602
    
    # CIFAR ResNets + BSConv-U
    cifar_resnet20_bsconvu
    cifar_resnet56_bsconvu
    cifar_resnet110_bsconvu
    cifar_resnet302_bsconvu
    cifar_resnet602_bsconvu
    
    # CIFAR ResNets + BSConv-S (p=1/4)
    cifar_resnet20_bsconvs_p1d4
    cifar_resnet56_bsconvs_p1d4
    cifar_resnet110_bsconvs_p1d4
    cifar_resnet302_bsconvs_p1d4
    cifar_resnet602_bsconvs_p1d4
    
    # CIFAR ResNets + BSConv-S (p=1/8)
    cifar_resnet20_bsconvs_p1d8
    cifar_resnet56_bsconvs_p1d8
    cifar_resnet110_bsconvs_p1d8
    cifar_resnet302_bsconvs_p1d8
    cifar_resnet602_bsconvs_p1d8
    
    ################################################################################
    
    # CIFAR Pre-Activation ResNets (aka ResNetsV2)
    cifar_preresnet20
    cifar_preresnet56
    cifar_preresnet110
    cifar_preresnet302
    cifar_preresnet602
    
    # CIFAR Pre-Activation ResNets (aka ResNetsV2) + BSConv-U
    cifar_preresnet20_bsconvu
    cifar_preresnet56_bsconvu
    cifar_preresnet110_bsconvu
    cifar_preresnet302_bsconvu
    cifar_preresnet602_bsconvu
    
    # CIFAR Pre-Activation ResNets (aka ResNetsV2) + BSConv-S (p=1/4)
    cifar_preresnet20_bsconvs_p1d4
    cifar_preresnet56_bsconvs_p1d4
    cifar_preresnet110_bsconvs_p1d4
    cifar_preresnet302_bsconvs_p1d4
    cifar_preresnet602_bsconvs_p1d4
    
    # CIFAR Pre-Activation ResNets (aka ResNetsV2) + BSConv-S (p=1/8)
    cifar_preresnet20_bsconvs_p1d8
    cifar_preresnet56_bsconvs_p1d8
    cifar_preresnet110_bsconvs_p1d8
    cifar_preresnet302_bsconvs_p1d8
    cifar_preresnet602_bsconvs_p1d8
    
    ################################################################################
    
    # CIFAR WideResNets-16
    cifar_wrn16_1
    cifar_wrn16_2
    cifar_wrn16_4
    cifar_wrn16_6
    cifar_wrn16_8
    cifar_wrn16_10
    cifar_wrn16_12
    
    # CIFAR WideResNets-16 + BSConv-U
    cifar_wrn16_1_bsconvu
    cifar_wrn16_2_bsconvu
    cifar_wrn16_4_bsconvu
    cifar_wrn16_6_bsconvu
    cifar_wrn16_8_bsconvu
    cifar_wrn16_10_bsconvu
    cifar_wrn16_12_bsconvu
    
    # CIFAR WideResNets-16 + BSConv-S (p=1/4)
    cifar_wrn16_1_bsconvs_p1d4
    cifar_wrn16_2_bsconvs_p1d4
    cifar_wrn16_4_bsconvs_p1d4
    cifar_wrn16_6_bsconvs_p1d4
    cifar_wrn16_8_bsconvs_p1d4
    cifar_wrn16_10_bsconvs_p1d4
    cifar_wrn16_12_bsconvs_p1d4
    
    # CIFAR WideResNets-16 + BSConv-S (p=1/8)
    cifar_wrn16_1_bsconvs_p1d8
    cifar_wrn16_2_bsconvs_p1d8
    cifar_wrn16_4_bsconvs_p1d8
    cifar_wrn16_6_bsconvs_p1d8
    cifar_wrn16_8_bsconvs_p1d8
    cifar_wrn16_10_bsconvs_p1d8
    cifar_wrn16_12_bsconvs_p1d8
    
    # CIFAR WideResNets-28
    cifar_wrn28_1
    cifar_wrn28_2
    cifar_wrn28_4
    cifar_wrn28_6
    cifar_wrn28_8
    cifar_wrn28_10
    cifar_wrn28_12
    
    # CIFAR WideResNets-28 + BSConv-U
    cifar_wrn28_1_bsconvu
    cifar_wrn28_2_bsconvu
    cifar_wrn28_4_bsconvu
    cifar_wrn28_6_bsconvu
    cifar_wrn28_8_bsconvu
    cifar_wrn28_10_bsconvu
    cifar_wrn28_12_bsconvu
    
    # CIFAR WideResNets-28 + BSConv-S (p=1/4)
    cifar_wrn28_1_bsconvs_p1d4
    cifar_wrn28_2_bsconvs_p1d4
    cifar_wrn28_4_bsconvs_p1d4
    cifar_wrn28_6_bsconvs_p1d4
    cifar_wrn28_8_bsconvs_p1d4
    cifar_wrn28_10_bsconvs_p1d4
    cifar_wrn28_12_bsconvs_p1d4
    
    # CIFAR WideResNets-28 + BSConv-S (p=1/8)
    cifar_wrn28_1_bsconvs_p1d8
    cifar_wrn28_2_bsconvs_p1d8
    cifar_wrn28_4_bsconvs_p1d8
    cifar_wrn28_6_bsconvs_p1d8
    cifar_wrn28_8_bsconvs_p1d8
    cifar_wrn28_10_bsconvs_p1d8
    cifar_wrn28_12_bsconvs_p1d8
    
    # CIFAR WideResNets-40
    cifar_wrn40_1
    cifar_wrn40_2
    cifar_wrn40_4
    cifar_wrn40_6
    cifar_wrn40_8
    cifar_wrn40_10
    cifar_wrn40_12
    
    # CIFAR WideResNets-40 + BSConv-U
    cifar_wrn40_1_bsconvu
    cifar_wrn40_2_bsconvu
    cifar_wrn40_4_bsconvu
    cifar_wrn40_6_bsconvu
    cifar_wrn40_8_bsconvu
    cifar_wrn40_10_bsconvu
    cifar_wrn40_12_bsconvu
    
    # CIFAR WideResNets-40 + BSConv-S (p=1/4)
    cifar_wrn40_1_bsconvs_p1d4
    cifar_wrn40_2_bsconvs_p1d4
    cifar_wrn40_4_bsconvs_p1d4
    cifar_wrn40_6_bsconvs_p1d4
    cifar_wrn40_8_bsconvs_p1d4
    cifar_wrn40_10_bsconvs_p1d4
    cifar_wrn40_12_bsconvs_p1d4
    
    # CIFAR WideResNets-40 + BSConv-S (p=1/8)
    cifar_wrn40_1_bsconvs_p1d8
    cifar_wrn40_2_bsconvs_p1d8
    cifar_wrn40_4_bsconvs_p1d8
    cifar_wrn40_6_bsconvs_p1d8
    cifar_wrn40_8_bsconvs_p1d8
    cifar_wrn40_10_bsconvs_p1d8
    cifar_wrn40_12_bsconvs_p1d8
    
    ################################################################################
    
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
    
If you use BSConv-S variants, you must add the regularization loss to your classification loss:

```python
# get model
model = bsconv.pytorch.get_model('resnet34_bsconvs_p1d4')

# training loop
...
output = model(images)
loss = criterion(output, target)

# THIS LINE MUST BE ADDED, everything else remains unchanged
loss += model.reg_loss(alpha=0.1)

optimizer.zero_grad()
loss.backward()
optimizer.step()
...
```


BSConv as General Drop-in Replacement
-------------------------------------

Using BSConv as a drop-in replacement requires the following steps:
1. Load an existing model definition
2. Replace convolution layers by BSConv modules
3. Add regularization loss (BSConv-S only)


### Step 1: Load an existing model definition

Currently supported are models based on regular convolution layers (`torch.nn.Conv2d`) without any bottleneck structures or group convolutions.
This includes ResNets (up to ResNet-34), VGGs, and DenseNets.
Code for larger ResNets (ResNet-50, ResNet-101, ResNet-152) will follow soon.

#### Using `torchvision` ([https://github.com/pytorch/vision](https://github.com/pytorch/vision))

Currently supported torchvision models are:
* ResNet-18, ResNet-34
* VGG-11, VGG-13, VGG-16, VGG-19 (with and without batch normalization)

Example (ResNet-18):

```python
import torchvision.models
model = torchvision.models.resnet18()
```

#### Using `pytorchcv` ([https://github.com/osmr/imgclsmob/tree/master/pytorch](https://github.com/osmr/imgclsmob/tree/master/pytorch))


Example (ResNet-18):

```python
import pytorchcv.model_provider
model = pytorchcv.model_provider.get_model("resnet18")
```

A full list of supported pytorchcv models will follow soon.

### Step 2: Replace convolution layers by BSConv modules

Replace each `torch.nn.Conv2d` by BSConv modules:

For unconstrained BSConv (BSConv-U):

```python
replacer = bsconv.pytorch.BSConvU_Replacer()
model = replacer.apply(model)
```

For subspace BSConv (BSConv-S):

```python
replacer = bsconv.pytorch.BSConvS_Replacer()
model = replacer.apply(model)
```

### Step 3: Add regularization loss (BSConv-S only)

When calculating the loss, the orthonormal regularization can easily be added with only one line of code.
The contribution to the global loss is determined by `alpha` (see paper for details).

```python
# training loop
...
output = model(images)
loss = criterion(output, target)

# THIS LINE MUST BE ADDED, everything else remains unchanged
loss += model.reg_loss(alpha=0.1)

optimizer.zero_grad()
loss.backward()
optimizer.step()
...
```

That's all you need to do in your training script!


BSConv PyTorch Modules
----------------------

We provide two PyTorch modules `bsconv.pytorch.BSConvU` (unconstrained BSConv) and `bsconv.pytorch.BSConvS` (subspace BSConv) which can be used instead of `torch.nn.Conv2d` layers.

### Example 1: Building a simple custom model with unconstrained BSConv-U modules:

```python
import torch
import bsconv.pytorch

class SimpleNet(torch.nn.Module):

    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = torch.nn.Sequential(
            bsconv.pytorch.BSConvU(3, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.ReLU(inplace=True),
            bsconv.pytorch.BSConvU(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU(inplace=True),
            bsconv.pytorch.BSConvU(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.ReLU(inplace=True),
            bsconv.pytorch.BSConvU(128, 256, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.ReLU(inplace=True),
            bsconv.pytorch.BSConvU(256, 512, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.ReLU(inplace=True),
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```

### Example 2: Building a simple custom model with subspace BSConv-S modules:

To easily apply the orthonormal regularization loss to each module, the model has to be derived as usual from `torch.nn.Module` but also from the mixin class `bsconv.pytorch.BSConvS_ModelRegLossMixin`.

```python
import torch
import bsconv.pytorch

class SimpleNet(torch.nn.Module, bsconv.pytorch.BSConvS_ModelRegLossMixin):

    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = torch.nn.Sequential(
            # using a BSConv-U module as the first conv layer,
            # since compressing a 3 channel input with BSConv-S would be overkill
            bsconv.pytorch.BSConvU(3, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(num_features=32),
            torch.nn.ReLU(inplace=True),
            bsconv.pytorch.BSConvS(32, 64, kernel_size=3, stride=2, padding=1, p=0.5),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU(inplace=True),
            bsconv.pytorch.BSConvS(64, 128, kernel_size=3, stride=2, padding=1, p=0.25),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.ReLU(inplace=True),
            bsconv.pytorch.BSConvS(128, 256, kernel_size=3, stride=2, padding=1, p=0.25),
            torch.nn.BatchNorm2d(num_features=256),
            torch.nn.ReLU(inplace=True),
            bsconv.pytorch.BSConvS(256, 512, kernel_size=3, stride=2, padding=1, p=0.25),
            torch.nn.BatchNorm2d(num_features=512),
            torch.nn.ReLU(inplace=True),
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```

Finally, you must add the regularization loss to your classification loss:

```python
# training loop
...
output = model(images)
loss = criterion(output, target)

# THIS LINE MUST BE ADDED, everything else remains unchanged
loss += model.reg_loss(alpha=0.1)

optimizer.zero_grad()
loss.backward()
optimizer.step()
...
```
