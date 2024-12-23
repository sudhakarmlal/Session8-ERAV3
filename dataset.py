import torch                 
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
import os

import cv2

import albumentations as A 
from albumentations.pytorch.transforms import ToTensorV2

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
            # Since we have normalized in the first step, mean is already 0, so fill_value = 0
            ToTensorV2()
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.albumentations_transform(image=img)['image']
        return img 


class Album_Test_transform():

    def __init__(self):
        self.albumentations_transform = A.Compose([
            A.Normalize(mean=[0.4914, 0.4822, 0.4471],std=[0.2469, 0.2433, 0.2615]),
            ToTensorV2()
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.albumentations_transform(image=img)['image']
        return img 
    

class CIFAR10:

    mean = (0.49139968, 0.48215827, 0.44653124)
    std = (0.24703233, 0.24348505, 0.26158768)
    classes = None

    def __init__(self, batch_size=128):
        self.batch_size = batch_size

        self.loader_kwargs = {'batch_size':batch_size,'num_workers':os.cpu_count()-1,'pin_memory':True}
        self.train_loaders , self.test_loaders = self.get_loaders()


    def get_train_loader(self):
        train_data = datasets.CIFAR10('data',train=True,download=True,transform=Album_Train_transform())

        if self.classes is None:
            self.classes = {i:c for i,c in enumerate(train_data.classes)}
        
        self.train_loader = torch.utils.data.DataLoader(train_data , shuffle = True , **self.loader_kwargs)
        return self.train_loader


    def get_test_loader(self):
        test_data = datasets.CIFAR10('data',train=False,download=True,transform=Album_Test_transform())

        if self.classes is None:
            self.classes = {i:c for i,c in enumerate(test_data.classes)}
        
        self.test_loader = torch.utils.data.DataLoader(test_data , shuffle = True , **self.loader_kwargs)
        return self.test_loader
    
    def get_loaders(self):
        return self.get_train_loader() , self.get_test_loader()