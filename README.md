~~You can now [find us at CVPR 2020](http://cvpr20.com/event/rethinking-depthwise-separable-convolutions-how-intra-kernel-correlations-lead-to-improved-mobilenets/). Our live Q&A sessions are on [June 18, 2020 @ 5pm - 7pm PDT](https://everytimezone.com/s/2c61ec42) [(click here to join)](http://cvpr20.com/event/rethinking-depthwise-separable-convolutions-how-intra-kernel-correlations-lead-to-improved-mobilenets/) and [June 19, 2020 @ 5am - 7am PDT](https://everytimezone.com/s/99be4dea) [(click here to join)](http://cvpr20.com/event/rethinking-depthwise-separable-convolutions-how-intra-kernel-correlations-lead-to-improved-mobilenets2nd-time/). We are looking forward to seeing you at CVPR!~~

***CVPR 2020 is now over, and we thank you for all the interesting discussions! We will continue the development of the code and models in this repository, so stay tuned!***

---

Blueprint Separable Convolutions (BSConv)
=========================================

This repository provides code and trained models for the CVPR 2020 paper ([official](http://openaccess.thecvf.com/content_CVPR_2020/html/Haase_Rethinking_Depthwise_Separable_Convolutions_How_Intra-Kernel_Correlations_Lead_to_Improved_CVPR_2020_paper.html), [arXiv](https://arxiv.org/abs/2003.13549)):

> **Rethinking Depthwise Separable Convolutions: How Intra-Kernel Correlations Lead to Improved MobileNets**
>
> Daniel Haase\*, Manuel Amthor\*

![Teaser GIF](teaser.gif)

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
        title = {Rethinking Depthwise Separable Convolutions: How Intra-Kernel Correlations Lead to Improved MobileNets},
        booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        month = {June},
        year = {2020}
    }
