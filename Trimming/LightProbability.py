#outlines 
import torch as pt
import numpy as np
import cv2
import os

DEVICE = "cuda:0"

class FFModel(pt.nn.Module):
    def __init__(self):
        super(FFModel, self).__init__()
        self.activation = pt.nn.ReLU()
        self.sig = pt.nn.Sigmoid()
        self.flatten = pt.nn.Flatten()
        self.conv1 = pt.nn.Conv2d(1,128,(3,3),1)
        self.conv2 = pt.nn.Conv2d(128,32,(3,3),1)
        self.conv3 = pt.nn.Conv2d(32,8,(3,3),1)
        self.conv4 = pt.nn.Conv2d(8,4,(3,3),1)
        self.convArry = [self.conv1,self.conv2,self.conv3,self.conv4]
        self.maxPool = pt.nn.MaxPool2d((2,2),2)
        self.linear1 = pt.nn.Linear(1404,512)
        self.linear2 = pt.nn.Linear(512,128)
        self.linear3 = pt.nn.Linear(128,1)
        self.linearArry = [self.linear1,self.linear2,self.linear3]
        
    def forward(self, x):
        for conv in self.convArry:
            x = conv(x)
            x = self.activation(x)
            x = self.maxPool(x)
        x = self.flatten(x)
        for linear in self.linearArry:
            x = linear(x)
            if linear != self.linearArry[-1]:
                x = self.activation(x)
        x = self.sig(x)
        return x

