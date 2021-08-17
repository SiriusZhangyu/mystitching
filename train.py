#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from dataset import mydataset
from model import Matcher

torch.cuda.empty_cache()


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    source_root='./source'
    sys_root='./sys_img'
    source_gt_root='./source_gt'
    sys_gt_root='./sys_gt'


    train_dataset = mydataset(img1_path=source_root, img2_path=sys_root, mask1_path=source_gt_root,mask2_path=sys_gt_root)

    batch_size = 1
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=0)


    net = Matcher()
    net.to(device)


    epochs = 50
    save_path = './mystitch.pth'
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    _loss = nn.MSELoss()
    for epoch in range(epochs):
        # train
        net.train()
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            source_img, sys_img,source_gt,sys_gt = data

            optimizer.zero_grad()



            output=net(source_img.to(device),sys_img.to(device))


            loss=_loss(source_gt.to(device),output)

            print(loss)
            loss.backward()
            optimizer.step()


        net.eval()

    torch.save(net, save_path)
    print('Finished Training')


if __name__ == '__main__':
    main()