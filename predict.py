import glob
import numpy as np
import torch
from utils.test_dataset import Band_Loader
import os
import re
import cv2
from model.unet_model import UNet
import matplotlib.pyplot as plt

def predict_rader(net, device, data_path, batch_size=4):
    
    # 读取所有图片路径
    band_dataset = Band_Loader(data_path)
    
    test_loader = torch.utils.data.DataLoader(dataset=band_dataset, batch_size=batch_size, shuffle=False)
    
    # 测试模式
    net.eval()
    save_path = '/home/aclab/Biechee/UNET/Main_Use/result/'
    i = 1

    # 遍歷所有圖片
    for files in range(len(band_dataset)):
        for band in test_loader:
            # print(test_path)
            # 保存结果地址
            save_res_path = save_path + str(i) + ".png"
            i += 1

            band = band.to(device=device, dtype=torch.float32)

            # 预测
            pred = net(band)
            # 提取结果
            pred = np.array(pred.data.cpu()[0])[0]
            # print(pred.shape)
            # 处理结果
            # 将单通道图像复制到三个通道
            rgb_image = np.stack((pred,)*3, axis=-1)
            rgb_image_uint8 = (rgb_image * 255).astype(np.uint8)

            # 保存图片
            cv2.imwrite(save_res_path, rgb_image_uint8)
        


if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    net = UNet(n_channels=4, n_classes=3)

    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('/home/aclab/Biechee/UNET/Main_Use/best_model.pth', map_location=device))
    
    data_path = "/home/aclab/Biechee/UNET/Main_Use/data/short_test/test"

    predict_rader(net, device, data_path)

    
        