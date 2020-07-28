~~You can now [find us at CVPR 2020](http://cvpr20.com/event/rethinking-depthwise-separable-convolutions-how-intra-kernel-correlations-lead-to-improved-mobilenets/). Our live Q&A sessions are on [June 18, 2020 @ 5pm - 7pm PDT](https://everytimezone.com/s/2c61ec42) [(click here to join)](http://cvpr20.com/event/rethinking-depthwise-separable-convolutions-how-intra-kernel-correlations-lead-to-improved-mobilenets/) and [June 19, 2020 @ 5am - 7am PDT](https://everytimezone.com/s/99be4dea) [(click here to join)](http://cvpr20.com/event/rethinking-depthwise-separable-convolutions-how-intra-kernel-correlations-lead-to-improved-mobilenets2nd-time/). We are looking forward to seeing you at CVPR!~~

***CVPR 2020 is now over, and we thank you for all the interesting discussions! Our presentation video is [available on YouTube](https://www.youtube.com/watch?v=nC6C-74xmbY). We will continue the development of the code and models in this repository, so stay tuned!***

---

Blueprint Separable Convolutions (BSConv)
=========================================

This repository provides code and trained models for the CVPR 2020 paper ([official](http://openaccess.thecvf.com/content_CVPR_2020/html/Haase_Rethinking_Depthwise_Separable_Convolutions_How_Intra-Kernel_Correlations_Lead_to_Improved_CVPR_2020_paper.html), [arXiv](https://arxiv.org/abs/2003.13549)):

> **Rethinking Depthwise Separable Convolutions: How Intra-Kernel Correlations Lead to Improved MobileNets**
>
> Daniel Haase\*, Manuel Amthor\*

![Teaser GIF](teaser.gif)

Table of Contents
-----------------

1. [Overview](#blueprint-separable-convolutions-bsconv)
2. [Results](#results)
    1. [CIFAR100 - ResNets](#cifar100---resnets)
    2. [CIFAR100 - WRN-28](#cifar100---wideresnets-wrn-28)
    2. [CIFAR100 - WRN-40](#cifar100---wideresnets-wrn-40)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Change Log](#change-log)
7. [Citation](#citation)

Results
-------

### CIFAR100 - ResNets

![CIFAR100 ResNet Params Plot](plots/cifar100-resnet-params.png)
![CIFAR100 ResNet FLOPs Plot](plots/cifar100-resnet-flops.png)

| Model                          |   Accuracy (top-1) |   Params [M] |   FLOPs [M] |
|:-------------------------------|-------------------:|-------------:|------------:|
| `cifar_resnet20`               |              68.59 |         0.28 |       41.42 |
| `cifar_resnet56`               |              71.31 |         0.86 |      127.39 |
| `cifar_resnet110`              |              71.29 |         1.74 |      256.34 |
| `cifar_resnet302`              |              72.22 |         4.85 |      714.83 |
| `cifar_resnet602`              |              71.22 |         9.71 |     1431.22 |
|                                |                    |              |             |
| `cifar_resnet20_bsconvu`       |              64.41 |         0.05 |        7.86 |
| `cifar_resnet56_bsconvu`       |              69.43 |         0.13 |       21.42 |
| `cifar_resnet110_bsconvu`      |              71.16 |         0.24 |       41.77 |
| `cifar_resnet302_bsconvu`      |              72.67 |         0.67 |      114.12 |
| `cifar_resnet602_bsconvu`      |              73.48 |         1.33 |      227.17 |
|                                |                    |              |             |
| `cifar_resnet20_bsconvs_p1d4`  |              62.03 |         0.03 |        5.66 |
| `cifar_resnet56_bsconvs_p1d4`  |              68.72 |         0.08 |       15.37 |
| `cifar_resnet110_bsconvs_p1d4` |              71.15 |         0.16 |       29.93 |
| `cifar_resnet302_bsconvs_p1d4` |              72.53 |         0.43 |       81.70 |
| `cifar_resnet602_bsconvs_p1d4` |              72.37 |         0.85 |      162.60 |

### CIFAR100 - WideResNets (WRN-28)

![CIFAR100 WRN-28 Params Plot](plots/cifar100-wrn28-params.png)
![CIFAR100 WRN-28 FLOPs Plot](plots/cifar100-wrn28-flops.png)

| Model                         |   Accuracy (top-1) |   Params [M] |   FLOPs [M] |
|:------------------------------|-------------------:|-------------:|------------:|
| `cifar_wrn28_1`               |              69.00 |         0.38 |       55.72 |
| `cifar_wrn28_2`               |              73.38 |         1.48 |      215.80 |
| `cifar_wrn28_3`               |              75.25 |         3.31 |      479.96 |
| `cifar_wrn28_4`               |              76.85 |         5.87 |      848.44 |
| `cifar_wrn28_8`               |              78.07 |        23.40 |     3365.72 |
| `cifar_wrn28_10`              |              78.58 |        36.54 |     5250.36 |
| `cifar_wrn28_12`              |              79.04 |        52.59 |     7552.34 |
|                               |                    |              |             |
| `cifar_wrn28_1_bsconvu`       |              66.21 |         0.06 |       10.09 |
| `cifar_wrn28_2_bsconvu`       |              71.78 |         0.20 |       34.09 |
| `cifar_wrn28_3_bsconvu`       |              73.79 |         0.44 |       71.46 |
| `cifar_wrn28_4_bsconvu`       |              75.29 |         0.75 |      122.45 |
| `cifar_wrn28_8_bsconvu`       |              77.15 |         2.87 |      462.76 |
| `cifar_wrn28_10_bsconvu`      |              78.04 |         4.44 |      714.70 |
| `cifar_wrn28_12_bsconvu`      |              78.30 |         6.36 |     1021.17 |
|                               |                    |              |             |
| `cifar_wrn28_1_bsconvs_p1d4`  |              64.65 |         0.04 |        7.26 |
| `cifar_wrn28_2_bsconvs_p1d4`  |              71.55 |         0.13 |       21.48 |
| `cifar_wrn28_3_bsconvs_p1d4`  |              74.42 |         0.26 |       42.25 |
| `cifar_wrn28_4_bsconvs_p1d4`  |              76.22 |         0.43 |       69.84 |
| `cifar_wrn28_8_bsconvs_p1d4`  |              79.49 |         1.58 |      248.36 |
| `cifar_wrn28_10_bsconvs_p1d4` |              79.56 |         2.42 |      378.52 |
| `cifar_wrn28_12_bsconvs_p1d4` |              80.26 |         3.44 |      535.94 |

### CIFAR100 - WideResNets (WRN-40)

![CIFAR100 WRN-40 Params Plot](plots/cifar100-wrn40-params.png)
![CIFAR100 WRN-40 FLOPs Plot](plots/cifar100-wrn40-flops.png)

| Model                         |   Accuracy (top-1) |   Params [M] |   FLOPs [M] |
|:------------------------------|-------------------:|-------------:|------------:|
| `cifar_wrn40_1`               |              70.34 |         0.57 |       84.38 |
| `cifar_wrn40_2`               |              74.13 |         2.26 |      329.74 |
| `cifar_wrn40_3`               |              75.70 |         5.06 |      735.79 |
| `cifar_wrn40_4`               |              77.55 |         8.97 |     1302.81 |
| `cifar_wrn40_8`               |              78.33 |        35.79 |     5180.42 |
| `cifar_wrn40_10`              |              78.49 |        55.90 |     8084.96 |
|                               |                    |              |             |
| `cifar_wrn40_1_bsconvu`       |              68.98 |         0.09 |       14.61 |
| `cifar_wrn40_2_bsconvu`       |              72.41 |         0.30 |       49.42 |
| `cifar_wrn40_3_bsconvu`       |              74.91 |         0.64 |      103.90 |
| `cifar_wrn40_4_bsconvu`       |              76.42 |         1.12 |      178.29 |
| `cifar_wrn40_8_bsconvu`       |              78.01 |         4.29 |      675.09 |
| `cifar_wrn40_10_bsconvu`      |              78.45 |         6.64 |     1043.03 |
|                               |                    |              |             |
| `cifar_wrn40_1_bsconvs_p1d4`  |              67.66 |         0.06 |       10.49 |
| `cifar_wrn40_2_bsconvs_p1d4`  |              73.19 |         0.18 |       31.09 |
| `cifar_wrn40_3_bsconvs_p1d4`  |              75.83 |         0.37 |       61.40 |
| `cifar_wrn40_4_bsconvs_p1d4`  |              76.97 |         0.63 |      101.66 |
| `cifar_wrn40_8_bsconvs_p1d4`  |              79.51 |         2.32 |      362.33 |
| `cifar_wrn40_10_bsconvs_p1d4` |              80.21 |         3.56 |      552.44 |

Requirements
------------

* `Python>=3.6`
* `PyTorch>=1.0.0` (support for other frameworks will be added later)

Installation
------------

```bash
pip install --upgrade bsconv
```

Usage
-----

![Demo GIF](demo.gif)

**[See here for PyTorch usage details](bsconv/pytorch/README.md).**

Support for other frameworks will be added later.

Please note that the code provided here is work-in-progress. Therefore, some features may be missing or may change between versions.

Change Log
----------

### 0.3.0 (2020-06-16)

* BSConv for PyTorch:
    * added ready-to-use model definitions (MobileNetV1, MobileNetV2, MobileNetsV3, ResNets and WRNs and their BSConv variants for CIFAR and ImageNet/fine-grained datasets)
    * added training script for CIFAR and ImageNet/fine-grained datasets
    * added class for the StanfordDogs dataset

### 0.2.0 (2020-04-16)

* BSConv for PyTorch:
    * removed activation and added option for normalization of PW layers in BSConv-S (issue #1) (**API change**)
    * added option for normalization of PW layers in BSConv-U (**API change**)
    * ensure that BSConv-S never uses more mid channels (= M') than input channels (M) and added parameter `min_mid_channels` (= M'_min) (**API change**)
    * added model profiler for parameter and FLOP counting
    * replacer now shows number of old and new model parameters

### 0.1.0 (2020-04-08)

* first public version
* BSConv for PyTorch:
    * modules `BSConvU` and `BSConvS`
    * replacers `BSConvU_Replacer` and `BSConvS_Replacer`

Citation
--------

If you find this work useful in your own research, please cite the paper as:

    @InProceedings{Haase_2020_CVPR,
        author = {Haase, Daniel and Amthor, Manuel},
        title = {Rethinking Depthwise Separable Convolutions: How Intra-Kernel Correlations Lead to Improved {MobileNets}},
        booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        month = {June},
        year = {2020}
    }
