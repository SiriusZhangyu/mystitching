#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import cv2
import random

import numpy as np

data=os.listdir('selected_curve')
matrix=[]



for name in data:

    cut=[]
    img=cv2.imread(os.path.join('./selected_curve',name))
    w,h= img.shape()
    num1=np.randint(0,w//2 )
    num2=np.randint(w//2,w)

    img1=img[:num2,:]
    img2=img[num1:,:]
    mask=img[num1:num2,:]

    cv2.imwrite(os.path.join('./target_dir',name), img1)
    cv2.imwrite(os.path.join('./source_dir', name), img2)
    cv2.imwrite(os.path.join('./mask_dir', name), mask)
