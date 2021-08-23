

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
import torchvision.models as models
from en2de import Matcher
import en2de

torch.cuda.empty_cache()

def main():
    torch.cuda.empty_cache()
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

    encoder = en2de.Encoder(models.__dict__['resnet50'])
    encoder.load_state_dict(torch.load('rotationinvariantmodel.pth'))


    net = Matcher()
    net.to(device)
    epochs = 1
    save_path = 'en2de.pth'
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    _loss = nn.CrossEntropyLoss()

    Loss_list = []

    for epoch in range(epochs):
        # train
        net.train()
        train_bar = tqdm(train_loader)
        l_sum=0.0
        for step, data in enumerate(train_bar):
            source_img, sys_img,source_gt,sys_gt = data
            optimizer.zero_grad()

            x1,x11,x2,x22 =encoder(source_img,sys_img)

            output=net(source_img.to(device), x1.to(device),x11.to(device),x2.to(device),x22.to(device))

            loss=_loss(output , source_gt.to(device, dtype=torch.int64))

            loss.backward()
            optimizer.step()
            l_sum += loss

        Loss_list.append(l_sum)


        net.eval()

    x=range(0,100)
    plt.plot(x, Loss_list, 'o-')
    plt.title('Test loss vs. epoches')
    plt.ylabel('Test loss')
    plt.savefig('loss.png')

    torch.save(net, save_path)

    print('Finished Training')


if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()
