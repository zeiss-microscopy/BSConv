Blueprint Separable Convolutions (BSConv)
=========================================

This repository provides code and trained models for the [CVPR 2020 paper](https://arxiv.org/abs/2003.13549v2):

    Rethinking Depthwise Separable Convolutions: How Intra-Kernel Correlations Lead to Improved MobileNets
    Daniel Haase*, Manuel Amthor*
    CVPR 2020
    arXiv:2003.13549

![Demo GIF](demo.gif)

**Please note that the code provided here is work-in-progress. Therefore, many features are still missing or may change between versions.**

Requirements
------------

* Python>=3.6
* PyTorch>=1.0.0 (support for other frameworks might be added later)

Installation
------------

```bash
pip install bsconv
```

Usage
-----

[See here for PyTorch usage details](bsconv/pytorch/README.md).


Change Log
----------

### 0.2.0 (2020-04-16)

* BSConv for PyTorch:
    * removed activation and added option for normalization of PW layers in BSConv-S (issue #1) (**API change**)
    * added option for normalization of PW layers in BSConv-U (**API change**)
    * ensure that BSConv-S never uses more mid channels (= M') than input channels (M) and add parameter `min_mid_channels` (= M'_min) (**API change**)
    * added model profiler for parameter and FLOP counting
    * replacer now shows number of old and new model parameters

### 0.1.0 (2020-04-08)

* first public version
* includes modules `BSConvU` and `BSConvS` for PyTorch
* includes replacers `BSConvU_Replacer` and `BSConvS_Replacer` for PyTorch
