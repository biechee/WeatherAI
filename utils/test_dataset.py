import numpy as np
import torch
import cv2
import os
import glob
import re
from torch.utils.data import Dataset
import random

class Band_Loader(Dataset):
    def __init__(self, data_path):
        #排序規則
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]

        #初始化定義
        self.data_path = data_path
        self.bands_path = sorted(glob.glob(os.path.join(data_path, '*.npz')), key=alphanum_key)

    def __getitem__(self, index):
        # 根據index讀取npz
        bands_path = np.load(self.bands_path[index])

        bands = [
            # bands_path['albedo_01'][:-1],
            # bands_path['albedo_02'][:-1],
            # bands_path['albedo_03'][:-1],
            # bands_path['albedo_04'][:-1],
            # bands_path['albedo_05'][:-1],
            # bands_path['albedo_06'][:-1],
            bands_path['tbb_07'][:-1],
            # bands_path['tbb_08'][:-1],
            bands_path['tbb_09'][:-1],
            # bands_path['tbb_10'][:-1],
            # bands_path['tbb_11'][:-1],
            # bands_path['tbb_12'][:-1],
            bands_path['tbb_13'][:-1],
            # bands_path['tbb_14'][:-1],
            # bands_path['tbb_15'][:-1],
            bands_path['tbb_16'][:-1]
        ]
        
        # 叠加 bands 数据
        bands = np.stack(bands, axis=-1)

        # 假设 label_data 是一个包含 label 数据的 NumPy 数组

        # print("Training data shape:", bands.shape)
        bands = bands.reshape(bands.shape[2], bands.shape[0], bands.shape[1])
        
        return bands
    
    def __len__(self):
        # 返回訓練集大小
        return len(self.bands_path)
    

if __name__ == "__main__":
    rasa_dataset = Band_Loader("/home/aclab/Biechee/UNET/Main_Use/data/test")
    print("數據個數：", len(rasa_dataset))
    # print(rasa_dataset[0])
    