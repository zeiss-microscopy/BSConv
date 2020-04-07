# BSConv implementation for pytorch

We provide several implementation variants for BSConv: 
1. A replacer functionality, which substitudes each standard convolution layer in existing networks by a BSConv module. 
2. A layer module, which can be used to build custom models from scratch.
3. A modification functionality for individual models such as MobileNets to reproduce the results reported in the paper.

Status: ...
Todo: ...

## BSConv as drop-in replacement for ResNets, VGGs, DenseNets

1. Load an existing model
2. Replace Conv2d layers by BSConv modules
3. Add the regularization loss (only BSConv-S)

Concrete examples will follow soon.


### 1. Load an existing model definition

#### Using `torchvision` ([https://github.com/pytorch/vision](https://github.com/pytorch/vision))

Currently supported are models with `torch.nn.Conv2d` layers without any bottleneck structure.
For torchvision these are:
* ResNet-18, ResNet-34
* VGG-11, VGG-13, VGG-16, VGG-19 (with and without batch normalization)

Example (ResNet-18):

    import torchvision.models
    model = torchvision.models.resnet18()


#### Using `pytorchcv` ([https://github.com/osmr/imgclsmob](https://github.com/osmr/imgclsmob/tree/master/pytorch/pytorchcv))

Currently supported are models with `torch.nn.Conv2d` layers without any bottleneck structure.

Example (ResNet-18):

    import pytorchcv.model_provider
    model = torchvision.models.resnet18()

	
### 2. Replace Conv2d
	
Replace each `torch.nn.Conv2d` by BSConv modules:

For unconstrained BSConv (BSConv-U):

	replacer = bsconv.pytorch.BSConvU_Replacer()
	model = replacer.apply(model)

For subspace BSConv (BSConv-S):
	
	replacer = bsconv.pytorch.BSConvS_Replacer()
	model = replacer.apply(model)

	
### 3. Add regularization loss (BSConv-S only)
	
When calucalting the loss, the regularization can easily be added with a weighting coefficient with only one modified line of code!
	
```python
...
output = model(images)
loss = criterion(output, target)

# THIS LINE MUST BE ADDED, everthing else remains unchanged
loss += model.reg_loss(alpha=0.1)

optimizer.zero_grad()
loss.backward()
optimizer.step()
...
```
	
That's all you need to do in your training script!


## Building your own model from scratch with BSConv

We provide 2 pytorch modules `bsconv.pytorch.BSConvU` and `bsconv.pytorch.BSConvS` which can be used instead of `torch.nn.Conv2d` layers.
Building your custom network:

	import torch
	import bsconv.pytorch
	
	model = torch.nn.Sequential()
	
	model.add(bsconv.pytorch.modules.bsconv_u())
	
	https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py

Note that if your network uses subspace BSConv-S layers (`bsconv.pytorch.BSConvS`), you must add the regularization loss to your classification loss:
	
	loss = loss + ...
	
Concrete examples will follow soon.
	
## Improving MobileNets with BSConv

Coming soon.