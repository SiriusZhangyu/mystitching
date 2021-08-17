#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import cv2
import math
import json
import random
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)


class Matcher(nn.Module):
    def __init__(self):
        super(Matcher, self).__init__()
        # self.encoder = torchvision.models.__dict__['resnet50'](pretrained=True)
        self.eps = 1e-8
        self.num_classes = 1

        self.enc1 = _EncoderBlock(1, 64)
        self.enc2 = _EncoderBlock(64, 128)
        self.enc3 = _EncoderBlock(128, 256)
        self.enc4 = _EncoderBlock(256, 512, dropout=True)

        self.dec3 = _DecoderBlock(1536, 512, 256)
        self.dec2 = _DecoderBlock(256, 128, 64)
        self.dec1 = nn.Conv2d(64, self.num_classes, kernel_size=1)

    def nn_concat(self, src_feat, tar_feat):
        """
        For each pixel in source feature, we find its nearest feature vector in the target feature and concatenate them.
        src_feat: B*C*H1*W1
        tar_feat: B*C*H2*W2
        concat_feat: B*2C*H1*W1
        """
        src_feat_flat = src_feat.flatten(start_dim=2).transpose(2,1)
        tar_feat_flat = tar_feat.flatten(start_dim=2).transpose(2,1)
        dist_mat = torch.cdist(src_feat_flat, tar_feat_flat)
        tar_idx = dist_mat.argmin(dim=2)
        nearest_tar_feat = torch.zeros_like(src_feat_flat)
        for b in range(src_feat.shape[0]):
            nearest_tar_feat[b] = tar_feat_flat[b][tar_idx[b]]
        nearest_tar_feat = nearest_tar_feat.transpose(2,1).reshape(*src_feat.shape)
        concat_feat = torch.cat((src_feat, nearest_tar_feat), 1)
        return concat_feat

    def forward(self, src_img, tar_img):
        src_feat1 = self.enc1(src_img)
        src_feat2 = self.enc2(src_feat1)
        src_feat3 = self.enc3(src_feat2)
        src_feat4 = self.enc4(src_feat3)

        tar_feat1 = self.enc1(tar_img)
        tar_feat2 = self.enc2(tar_feat1)
        tar_feat3 = self.enc3(tar_feat2)
        tar_feat4 = self.enc4(tar_feat3)

        concat_feat3 = self.nn_concat(src_feat3, tar_feat3)
        concat_feat4 = self.nn_concat(src_feat4, tar_feat4)

        dec_feat = self.dec3(torch.cat([concat_feat3, F.interpolate(concat_feat4, concat_feat3.size()[2:], mode='bilinear')], 1))
        dec_feat = self.dec2(dec_feat)
        dec_feat = self.dec1(dec_feat)
        pred_mask = F.upsample(dec_feat, src_img.size()[2:], mode='bilinear')

        return pred_mask
