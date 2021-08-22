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
        self.num_classes = 2
        self.encoder = torch.load('rotationinvariantmodel.pth')
        for param in self.encoder :
            param.requires_grad = False

        self.enc1 = self.encoder.layer3
        self.enc2 = self.encoder.layer4


        self.dec4 = _DecoderBlock(6144,3072, 1536)
        self.dec3 = _DecoderBlock(1536, 768, 384)
        self.dec2 = _DecoderBlock(384, 192, 96)
        self.dec1 = nn.Conv2d(96, self.num_classes, kernel_size=1)

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


        tar_feat1 = self.enc1(tar_img)
        tar_feat2 = self.enc2(tar_feat1)


        concat_feat1 = self.nn_concat(src_feat1, tar_feat1)
        concat_feat2 = self.nn_concat(src_feat2, tar_feat2)

        dec_feat = self.dec4(torch.cat([concat_feat1, F.interpolate(concat_feat2, concat_feat1.size()[2:], mode='bilinear')], 1))
        dec_feat = self.dec3(dec_feat)
        dec_feat = self.dec2(dec_feat)
        dec_feat = self.dec1(dec_feat)
        pred_mask = F.upsample(dec_feat, src_img.size()[2:], mode='bilinear')

        return pred_mask
