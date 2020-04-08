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

Coming soon.


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

#### Using `pytorchcv` ([https://github.com/osmr/imgclsmob](https://github.com/osmr/imgclsmob/tree/master/pytorch/pytorchcv))


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

### Example 1: Building a simple custom model with unconstrained BSConvU modules:

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

### Example 2: Building a simple custom model with subspace BSConvS modules:

To easily apply the orthonormal regularization loss to each module, the model has to be derived as usual from `torch.nn.Module` but also from the mixin class `bsconv.pytorch.BSConvS_ModelRegLossMixin`.

```python
import torch
import bsconv.pytorch

class SimpleNet(torch.nn.Module, bsconv.pytorch.BSConvS_ModelRegLossMixin):

    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = torch.nn.Sequential(
            # using a BSConvU module as the first conv layer,
            # since compressing a 3 channel input with BSConvS would be overkill
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
