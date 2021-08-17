#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import cv2
import random
import numpy as np


data=os.listdir('./curve')


for name in data:

    img=cv2.imread(os.path.join('./curve',name))
    mask=cv2.imread(os.path.join('./mask',name))

    w,h= img.shape()
    num1=2**w//3
    num2=2//3

    img1=img[:num1,:]
    img2=img[num2:,:]
    overlap=img[num2:num1,:]
    img1_mask=mask[:num1,:]
    img2_mask=mask[num2:,:]
    overlap_mask=mask[num2:num1,:]

    cv2.imwrite(os.path.join('./target_dir',name), img1)
    cv2.imwrite(os.path.join('./source_dir', name), img2)
    cv2.imwrite(os.path.join('./mask_dir', name), mask)
