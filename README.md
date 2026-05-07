# Advanced Neural Networks Project

This repository contains the code and notebooks for a neural networks project comparing classic convolutional models with a Vision Transformer. The main idea is to look at how different architectures behave under fixed training-time budgets, especially when moving from simpler image datasets like MNIST to more complex datasets like CIFAR-10 and Tiny ImageNet.

Most of the work is organized as notebooks, with the reusable model definitions kept in Python files under `model/`.

## Repository Structure

```text
.
|-- data_utils.py
|-- requirements.txt
`-- model/
    |-- alexnet.py
    |-- le_net_5.py
    |-- vision_transformers.py
    |-- resnet.py
    |-- custom_architecture.py
    |-- validation_lenet5.ipynb
    |-- validation_alexnet.ipynb
    |-- validation_vision_transformers.ipynb
    |-- lenet_alexnet_mnist.ipynb
    |-- lenet_alexnet_cifar10.ipynb
    |-- Anet_Vit_Cifar10.ipynb
    |-- Anet_Vit_Imagenet.ipynb
    |-- experiment_1.ipynb
    `-- data/
```

## Main Files

`data_utils.py` contains helper functions for downloading and extracting datasets from Kaggle. Right now it includes utilities for Tiny ImageNet and CIFAR-10. The Tiny ImageNet helper is the one called when the file is run directly.

`requirements.txt` lists the Python packages used across the project. The notebooks also install or import some common deep learning tools directly, especially when running in Google Colab.

## Model Definitions

`model/le_net_5.py` defines a LeNet-5 style convolutional network. It was originally designed around MNIST-like images, but the implementation has been adjusted so it can also work with CIFAR-10 by changing the number of input channels and output classes.

`model/alexnet.py` defines `AlexNet4803`, a custom AlexNet-style model. It keeps the general AlexNet structure: convolutional layers, ReLU activations, pooling, local response normalization, adaptive average pooling, and a large fully connected classifier.

`model/vision_transformers.py` defines a small Vision Transformer implementation from scratch. It includes a patch embedding layer, transformer blocks using multi-head attention, a class token, positional encoding, and a final classification head.

`model/resnet.py` and `model/custom_architecture.py` are currently placeholders. They are present for possible future architecture experiments, but they do not contain model code yet.

## Validation Notebooks

`model/validation_lenet5.ipynb` runs validation experiments for LeNet-5. It tests learning rates and dropout values using cross-validation so the final comparison notebooks can use a stronger configuration.

`model/validation_alexnet.ipynb` is used to tune the AlexNet-style model. It focuses especially on the first convolution layer settings, such as stride and padding, while keeping the rest of the architecture mostly fixed.

`model/validation_vision_transformers.ipynb` validates Vision Transformer settings on CIFAR-10 and Tiny ImageNet. It sweeps through hyperparameters such as the number of attention heads, dropout rate, and MLP ratio.

## Experiment Notebooks

`model/lenet_alexnet_mnist.ipynb` compares LeNet-5 and AlexNet on MNIST using a fixed training-time budget. This gives a baseline view of how a smaller classic CNN compares with a much larger AlexNet-style model on a simpler dataset.

`model/lenet_alexnet_cifar10.ipynb` repeats the LeNet-5 versus AlexNet comparison on CIFAR-10. This is useful because CIFAR-10 is more visually complex than MNIST, so the efficiency and accuracy tradeoffs change.

`model/Anet_Vit_Cifar10.ipynb` compares AlexNet and the Vision Transformer on CIFAR-10 under the same fixed time budget. The notebook uses the best or selected settings from the validation notebooks.

`model/Anet_Vit_Imagenet.ipynb` compares AlexNet and the Vision Transformer on Tiny ImageNet-200. This is the largest dataset experiment in the repository and uses a longer fixed training-time budget.

`model/experiment_1.ipynb` is a short experiment notebook describing the broader comparison between LeNet and AlexNet on simple versus more complex datasets.

## Data Folder

`model/data/` stores downloaded dataset files. The current repository includes raw MNIST files under `model/data/MNIST/raw/`. Other datasets, such as CIFAR-10 and Tiny ImageNet, are expected to be downloaded when running the notebooks or helper scripts.

## How To Run

The notebooks are the main entry point. They are written to work well in Colab and often include a setup cell that clones or pulls this repository before importing local model files.

For a local setup, install the dependencies first:

```bash
pip install -r requirements.txt
```

Then open the notebooks in `model/` and run the validation notebooks before the final comparison notebooks if you want to reproduce the same workflow.

