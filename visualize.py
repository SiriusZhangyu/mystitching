import os
import json

import cv2
import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from dataset import mydataset

from torchvision import utils as vutils




def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.cuda.empty_cache()

    net=torch.load('mystitch.pth')


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
    net.to(device)

    train_bar = tqdm(train_loader)
    for step, data in enumerate(train_bar):
        if step > 10:
            break
        source_img, sys_img, source_gt, sys_gt = data
        output = net(source_img.to(device), sys_img.to(device))
        print(output.size())
        output=output.detach().cpu().numpy()
        a=output.shape[2]
        b=output.shape[3]

        output=np.max(output, axis=1).reshape(a,b)
        print(output.shape)

        output = (((output - np.min(output)) / (np.max(output) - np.min(output))) * 255).astype(np.uint8)
        cv2.applyColorMap(output, cv2.COLORMAP_JET)

        cv2.imwrite(os.path.join('./save',str(step)+'.png'),output)



if __name__ == '__main__':
    test()
