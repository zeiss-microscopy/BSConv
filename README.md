Blueprint Separable Convolutions (BSConv)
=========================================

This repository provides code and trained models for the [CVPR 2020 paper](https://arxiv.org/abs/2003.13549v2):

    Rethinking Depthwise Separable Convolutions: How Intra-Kernel Correlations Lead to Improved MobileNets
    Daniel Haase*, Manuel Amthor*
    CVPR 2020
    arXiv:2003.13549

**Please note that the code provided here is work-in-progress. Therefore, many features are still missing or may change bewteen versions.**

Requirements
------------

* Python>=3.6
* PyTorch>=1.0.0 (support for other frameworks might be added later)

Installation
------------

### Installation via PyPI

```bash
pip install bsconv
```

### Installation via Wheel

1. Download `bsconv-x.y.z.whl`.
2. Install the downloaded Wheel via `pip`
```bash
pip install bsconv-x.y.z.whl
```

Usage
-----

[See here for PyTorch usage details](bsconv/pytorch/README.md).


Change Log
----------

### 0.1.0 (2020-04-??)

* first version
* includes modules `BSConvU` and `BSConvS` for PyTorch
* includes replacers `BSConvU_Replacer` and `BSConvS_Replacer` for PyTorch
