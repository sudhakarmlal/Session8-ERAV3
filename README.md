# Session 9

# Advanced Concepts, Data Augmentation & Visualizations

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 2.0](https://img.shields.io/badge/torch-v2.0-brightgreen)](https://pytorch.org/docs/stable/index.html)
[![Torchvision 0.15](https://img.shields.io/badge/torchvision-v0.15-green)](https://pytorch.org/vision/stable/index.html)
[![TQDM](https://img.shields.io/badge/tqdm-v4.65.0-yellowgreen)](https://tqdm.github.io/)
[![Albumentations](https://img.shields.io/badge/Albumentations-v1.3.1-yellow)](https://albumentations.ai/docs/)

<br>

# Task

Write a new network that
1. has the architecture to C1C2C3C40 (No MaxPooling, but 3 convolutions, where the last one has a stride of 2 instead) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)
2. total RF must be more than 44
3. one of the layers must use Depthwise Separable Convolution
4. one of the layers must use Dilated Convolution
5. use GAP (compulsory):- add FC after GAP to target #of classes (optional)
6. use albumentation library and apply:
    - horizontal flip
    - shiftScaleRotate
    - coarseDropout (max_holes = 1, max_height=16px, max_width=16, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
7. achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.
8. make sure you're following code-modularity

<br>

# Solution

This repository contains a model trained and validated on `CIFAR10 dataset` using Advanced Convolutions and Data Augmentation Techniques. We have `used Dilated kernels and Depthwise Separable Convolution` to train the model and achieve 85% Test Accuracy.

<br>

## File Contents

1. `model.py` - This file contains a model created using Dilated Kernels and Depthwise Separable Convolution and applying skip connections in forward function.

2. `utils.py` - This file contains all the necessary utility functions and methods.

3. `backprop.py` - This file contains necessary train and test functions for the model as well as plotting graphs and misclassified images function.

4. `dataset.py` - This file contains data loaders and data augmentation methods (train_transforms and test_transforms).

<br>

# Applying Albumentations library

We have used [Albumentations](https://albumentations.ai/docs/) library for Data Augmentation in this assignment.

### Features :
- Albumentations is a fast and flexible image augmentation library.
- Albumentations is written in Python, and it is licensed under the MIT license. 

<br>

```python
class Album_Train_transform():

    def __init__(self):
        self.albumentations_transform = A.Compose([
            A.Normalize(mean=[0.4914, 0.4822, 0.4471],std=[0.2469, 0.2433, 0.2615]),
            A.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.2),
            A.ToGray(p=0.1),
            A.Downscale(0.8, 0.95, p=0.2),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),
            A.CoarseDropout(max_holes=1, max_height=16, max_width=16, p=0.2, fill_value=0),
            # Since we have normalized in the first step, mean and std are already 0, so fill_value = 0
            ToTensorV2()
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.albumentations_transform(image=img)['image']
        return img 
```

![Image Augmentations](../Results/Session%209/augmentation_images.png)


<br>

# Depthwise Separable Convolution

The Depthwise Separable Convolution is implemented using simple convolution blocks in the model.

```python
self.convlayer = nn.Sequential(
        nn.Conv2d(input_c, output_c, 3, groups=output_c, padding=1, bias=False),
        nn.BatchNorm2d(output_c),
        nn.Dropout(dropout),
        nn.ReLU(),
        nn.Conv2d(output_c, output_c, 1, groups=1, padding=0, bias=False)
    )
```

<br>

# Forward function (Skip connection)

The Skip connection is implemented in forward function as below.

```python
def forward(self, x):

    x = self.conv_block1(x)
    x = self.trans_block1(x)
    x = x + self.conv_block2(x)
    x = self.trans_block2(x)
    x = x + self.conv_block3(x)
    x = self.trans_block3(x)
    x = x + self.conv_block4(x)
    x = self.trans_block4(x)
    x = self.out_block(x)
    
    return x
```

<br>

# Model Summary

```python
================================================================
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 24, 32, 32]             648
       BatchNorm2d-2           [-1, 24, 32, 32]              48
           Dropout-3           [-1, 24, 32, 32]               0
              ReLU-4           [-1, 24, 32, 32]               0
            Conv2d-5           [-1, 24, 32, 32]             216
       BatchNorm2d-6           [-1, 24, 32, 32]              48
           Dropout-7           [-1, 24, 32, 32]               0
              ReLU-8           [-1, 24, 32, 32]               0
            Conv2d-9           [-1, 24, 32, 32]             576
           Conv2d-10           [-1, 32, 30, 30]           6,912
      BatchNorm2d-11           [-1, 32, 30, 30]              64
          Dropout-12           [-1, 32, 30, 30]               0
             ReLU-13           [-1, 32, 30, 30]               0
           Conv2d-14           [-1, 32, 30, 30]             288
      BatchNorm2d-15           [-1, 32, 30, 30]              64
          Dropout-16           [-1, 32, 30, 30]               0
             ReLU-17           [-1, 32, 30, 30]               0
           Conv2d-18           [-1, 32, 30, 30]           1,024
           Conv2d-19           [-1, 32, 30, 30]             288
      BatchNorm2d-20           [-1, 32, 30, 30]              64
          Dropout-21           [-1, 32, 30, 30]               0
             ReLU-22           [-1, 32, 30, 30]               0
           Conv2d-23           [-1, 32, 30, 30]           1,024
           Conv2d-24           [-1, 64, 26, 26]          18,432
      BatchNorm2d-25           [-1, 64, 26, 26]             128
          Dropout-26           [-1, 64, 26, 26]               0
             ReLU-27           [-1, 64, 26, 26]               0
           Conv2d-28           [-1, 64, 26, 26]             576
      BatchNorm2d-29           [-1, 64, 26, 26]             128
          Dropout-30           [-1, 64, 26, 26]               0
             ReLU-31           [-1, 64, 26, 26]               0
           Conv2d-32           [-1, 64, 26, 26]           4,096
           Conv2d-33           [-1, 64, 26, 26]             576
      BatchNorm2d-34           [-1, 64, 26, 26]             128
          Dropout-35           [-1, 64, 26, 26]               0
             ReLU-36           [-1, 64, 26, 26]               0
           Conv2d-37           [-1, 64, 26, 26]           4,096
           Conv2d-38           [-1, 96, 18, 18]          55,296
      BatchNorm2d-39           [-1, 96, 18, 18]             192
          Dropout-40           [-1, 96, 18, 18]               0
             ReLU-41           [-1, 96, 18, 18]               0
           Conv2d-42           [-1, 96, 18, 18]             864
      BatchNorm2d-43           [-1, 96, 18, 18]             192
          Dropout-44           [-1, 96, 18, 18]               0
             ReLU-45           [-1, 96, 18, 18]               0
           Conv2d-46           [-1, 96, 18, 18]           9,216
           Conv2d-47           [-1, 96, 18, 18]             864
      BatchNorm2d-48           [-1, 96, 18, 18]             192
          Dropout-49           [-1, 96, 18, 18]               0
             ReLU-50           [-1, 96, 18, 18]               0
           Conv2d-51           [-1, 96, 18, 18]           9,216
           Conv2d-52             [-1, 96, 2, 2]          82,944
      BatchNorm2d-53             [-1, 96, 2, 2]             192
          Dropout-54             [-1, 96, 2, 2]               0
             ReLU-55             [-1, 96, 2, 2]               0
AdaptiveAvgPool2d-56             [-1, 96, 1, 1]               0
           Conv2d-57             [-1, 10, 1, 1]             970
          Flatten-58                   [-1, 10]               0
       LogSoftmax-59                   [-1, 10]               0
================================================================
Total params: 199,562
Trainable params: 199,562
Non-trainable params: 0
================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 12.72
Params size (MB): 0.76
Estimated Total Size (MB): 13.49
================================================================
```
<br>

# Receptive Field Calculations

| Block  | Conv layer | Layers | Kernel | R_in | N_in | J_in | Stride | Padding | Dilation | Eff. Kernel size | R_out | N_out | J_out |
|--------|------------|--------|--------|------|------|------|--------|---------|----------|------------------|-------|-------|-------|
| Input  | conv0      | 0      |        |      |      |      |        |         |          |                  | 1     | 32    | 1     |
| C1     | conv1      | 1      | 3      | 1    | 32   | 1    | 1      | 1       | 1        | 3                | 3     | 32    | 1     |
|        | conv2      | 2      | 3      | 3    | 32   | 1    | 1      | 1       | 1        | 3                | 5     | 32    | 1     |
|        | trans1     | 3      | 3      | 5    | 32   | 1    | 1      | 0       | 1        | 3                | 7     | 30    | 1     |
| C2     | conv3      | 4      | 3      | 7    | 30   | 1    | 1      | 1       | 1        | 3                | 9     | 30    | 1     |
|        | conv4      | 5      | 3      | 9    | 30   | 1    | 1      | 1       | 1        | 3                | 11    | 30    | 1     |
|        | trans2     | 6      | 3      | 11   | 30   | 1    | 1      | 0       | 2        | 5                | 15    | 26    | 1     |
| C3     | conv5      | 7      | 3      | 15   | 26   | 1    | 1      | 1       | 1        | 3                | 17    | 26    | 1     |
|        | conv6      | 8      | 3      | 17   | 26   | 1    | 1      | 1       | 1        | 3                | 19    | 26    | 1     |
|        | trans3     | 9      | 3      | 19   | 26   | 1    | 1      | 0       | 4        | 9                | 27    | 18    | 1     |
| C4     | conv7      | 10     | 3      | 27   | 18   | 1    | 1      | 1       | 1        | 3                | 29    | 18    | 1     |
|        | conv8      | 11     | 3      | 29   | 18   | 1    | 1      | 1       | 1        | 3                | 31    | 18    | 1     |
|        | trans4     | 12     | 3      | 31   | 18   | 1    | 1      | 0       | 8        | 17               | 47    | 2     | 1     |
| GAP    | gap        | 13     | 2      | 47   | 2    | 1    | 1      | 0       | 1        | 2                | 48    | 1     | 1     |
| Output | output     | 14     | 1      | 48   | 1    | 1    | 1      | 0       | 1        | 1                | 48    | 1     | 1     |

<br>

# Results

Best Training Accuracy : `81.04`   
Best Test Accuracy : `85.08`

![Results](../Results/Session%209/Results.png)

<br>

# Misclassified Images

![Misclassified Images](../Results/Session%209/misclassified_images.png)

<br>

# Training Testing Logs

```python
EPOCH: 38
Loss=0.5958437323570251 Batch_id=390 Accuracy=80.87: 100%|██████████| 391/391 [00:28<00:00, 13.82it/s]

Test set: Average loss: 0.4386, Accuracy: 8493/10000 (84.93%)

Epoch 00039: reducing learning rate of group 0 to 1.0000e-05.
EPOCH: 39
Loss=0.6107109189033508 Batch_id=390 Accuracy=81.04: 100%|██████████| 391/391 [00:28<00:00, 13.68it/s]

Test set: Average loss: 0.4376, Accuracy: 8498/10000 (84.98%)

EPOCH: 40
Loss=0.5270119309425354 Batch_id=390 Accuracy=80.69: 100%|██████████| 391/391 [00:27<00:00, 14.02it/s]

Test set: Average loss: 0.4354, Accuracy: 8508/10000 (85.08%)

EPOCH: 41
Loss=0.618690013885498 Batch_id=390 Accuracy=81.01: 100%|██████████| 391/391 [00:29<00:00, 13.04it/s]

Test set: Average loss: 0.4369, Accuracy: 8506/10000 (85.06%)

EPOCH: 42
Loss=0.5880504846572876 Batch_id=390 Accuracy=80.79: 100%|██████████| 391/391 [00:28<00:00, 13.89it/s]

Test set: Average loss: 0.4372, Accuracy: 8494/10000 (84.94%)

Epoch 00043: reducing learning rate of group 0 to 1.0000e-06.
EPOCH: 43
Loss=0.4305812418460846 Batch_id=390 Accuracy=80.87: 100%|██████████| 391/391 [00:27<00:00, 14.06it/s]

Test set: Average loss: 0.4350, Accuracy: 8505/10000 (85.05%)
```

