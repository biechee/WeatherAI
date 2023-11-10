import numpy as np
import torch
import cv2
import os
import glob
import re
from torch.utils.data import Dataset
import random

class RASA_Loader(Dataset):
    def __init__(self, data_path):
        #排序規則
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]

        #初始化定義
        self.data_path = data_path
        self.bands_path = sorted(glob.glob(os.path.join(data_path, 'bands/*.npz')), key=alphanum_key)
        
        self.label_path = sorted(glob.glob(os.path.join(data_path, 'label/*.png')), key=alphanum_key)


    def __getitem__(self, index):
        # 根據index讀取npz
        bands_path = np.load(self.bands_path[index])
        label_path = self.label_path[index]
        
        label = cv2.imread(label_path)

        if label.max() > 1:
            label = label / 255

        # [-1]之後要刪掉
        latitude = bands_path['latitude']
        longitude = bands_path['longitude']
        albedo_01 = bands_path['albedo_01']
        albedo_02 = bands_path['albedo_02']
        albedo_03 = bands_path['albedo_03']
        albedo_04 = bands_path['albedo_04']
        albedo_05 = bands_path['albedo_05']
        albedo_06 = bands_path['albedo_06']
        tbb_07 = bands_path['tbb_07']
        tbb_08 = bands_path['tbb_08']
        tbb_09 = bands_path['tbb_09']
        tbb_10 = bands_path['tbb_10']
        tbb_11 = bands_path['tbb_11']
        tbb_12 = bands_path['tbb_12']
        tbb_13 = bands_path['tbb_13']
        tbb_14 = bands_path['tbb_14']
        tbb_15 = bands_path['tbb_15']
        tbb_16 = bands_path['tbb_16']
        # print(label.shape)
        # print(latitude.shape)

        bands = [
            # bands_path['albedo_01'],
            # bands_path['albedo_02'],
            # bands_path['albedo_03'],
            # bands_path['albedo_04'],
            # bands_path['albedo_05'],
            # bands_path['albedo_06'],
            bands_path['tbb_07'],
            # bands_path['tbb_08'],
            bands_path['tbb_09'],
            # bands_path['tbb_10'],
            # bands_path['tbb_11'],
            # bands_path['tbb_12'],
            bands_path['tbb_13'],
            # bands_path['tbb_14'],
            # bands_path['tbb_15'],
            bands_path['tbb_16']
        ]
        
        # 叠加 bands 数据
        bands = np.stack(bands, axis=-1)

        # 假设 label_data 是一个包含 label 数据的 NumPy 数组

        # print("Training data shape:", bands.shape)
        bands = bands.reshape(bands.shape[2], bands.shape[0], bands.shape[1])
        label = label.reshape(label.shape[2], label.shape[0], label.shape[1])
        # print(bands.shape)
        
        

        # # 根据image_path生成label_path
        # label_path = self.label_path[index]
        

        # # 读取训练图片和标签图片
        
        # label = cv2.imread(label_path)
        
        
        return bands, label
        # return 1
    
    def __len__(self):
        # 返回訓練集大小
        return len(self.bands_path)
    


if __name__ == "__main__":
    rasa_dataset = RASA_Loader("/home/aclab/Biechee/UNET/Main_Use/data/band_rader_Near_size/train")
    print("數據個數：", len(rasa_dataset))
    print(rasa_dataset[7])
    