#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import torch
import torch.utils.data as data
import numpy as np
import cv2


class mydataset(data.Dataset):

    def __init__(self, img1_path, img2_path, mask1_path, mask2_path):
        self.img1_path = img1_path
        self.img2_path = img2_path
        self.mask1_path =mask1_path
        self.mask2_path=mask2_path


    def __len__(self):

        return len(os.listdir(self.img1_path))

    def __getitem__(self, index):
        img_lists=os.listdir(self.img1_path)
        img1 = cv2.imread(os.path.join(self.img1_path,img_lists[index]),0)

        img2 = cv2.imread(os.path.join(self.img2_path, img_lists[index]),0)

        mask1=cv2.imread(os.path.join(self.mask1_path,img_lists[index]),0)
        mask2=cv2.imread(os.path.join(self.mask2_path,img_lists[index]),0)

        img1 = torch.Tensor(img1)
        img1 = img1.unsqueeze(0)

        img2 = torch.Tensor(img2)
        img2 = img2.unsqueeze(0)

        ret, mask1 = cv2.threshold(mask1, 127, 1, cv2.THRESH_BINARY)
        mask1 = torch.Tensor(mask1)

        mask2 = torch.Tensor(mask2)

        return img1, img2, mask1, mask2
