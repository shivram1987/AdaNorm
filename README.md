# AdaNorm Optimizer for CNNs (WACV 2023)

This repository contains the code of AdaNorm optimizers (i.e., AdamNorm, diffGradNorm, RadamNorm, and AdaBeliefNorm).


## Abstract

The stochastic gradient descent (SGD) optimizers are generally used to train the convolutional neural networks (CNNs). In recent years, several adaptive momentum based SGD optimizers have been introduced, such as Adam, diffGrad, Radam and AdaBelief. However, the existing SGD optimizers do not exploit the gradient norm of past iterations and lead to poor convergence and performance. In this paper, we propose a novel AdaNorm based SGD optimizers by correcting the norm of gradient in each iteration based on the adaptive training history of gradient norm. By doing so, the proposed optimizers are able to maintain high and representive gradient throughout the training and solves the low and atypical gradient problems. The proposed concept is generic and can be used with any existing SGD optimizer. We show the efficacy of the proposed AdaNorm with four state-of-the-art optimizers, including Adam, diffGrad, Radam and AdaBelief. We depict the performance improvement due to the proposed optimizers using three CNN models, including VGG16, ResNet18 and ResNet50, on three benchmark object recognition datasets, including CIFAR10, CIFAR100 and TinyImageNet.




## Citation
@inproceedings{dubey2023adanorm,<br/>
  title={AdaNorm: Adaptive Gradient Norm Correction based Optimizer for CNNs},<br/>
  author={Dubey, Shiv Ram and Singh, Satish Kumar and Chaudhuri, Bidyut Baran},<br/>
  booktitle={IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},<br/>
  year={2023}<br/>
}
