import numpy as np
import cv2
import os
import glob
import re
from torch.utils.data import Dataset
from tqdm import tqdm

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
        
        # label = cv2.imread(label_path)
        label = self.extract_color(label_path)

        # if label.max() > 1:
        #     label = label / 255

        latitude = bands_path['latitude']
        longitude = bands_path['longitude']
        # print(label.shape)
        
        lat_bounds = [20.5, 26.5]  # 例如：從 20 到 50 度
        lon_bounds = [118.0, 124.0]  # 例如：從 100 到 150 度
        latitude_range = np.where((latitude >= lat_bounds[0]) & (latitude <= lat_bounds[1]))[0]
        longitude_range = np.where((longitude >= lon_bounds[0]) & (longitude <= lon_bounds[1]))[0]

        latitude = latitude[latitude_range[0]:latitude_range[len(latitude_range) - 1]]
        longitude = longitude[longitude_range[0]:longitude_range[len(longitude_range) - 1]]
        
        bands = [
            # bands_path['albedo_01'],
            # bands_path['albedo_02'],
            # bands_path['albedo_03'],
            # bands_path['albedo_04'],
            # bands_path['albedo_05'],
            # bands_path['albedo_06'],
            bands_path['tbb_07'][latitude_range, :][:, longitude_range],
            # bands_path['tbb_08'],
            bands_path['tbb_09'][latitude_range, :][:, longitude_range],
            # bands_path['tbb_10'],
            # bands_path['tbb_11'],
            # bands_path['tbb_12'],
            bands_path['tbb_13'][latitude_range, :][:, longitude_range],
            # bands_path['tbb_14'],
            # bands_path['tbb_15'],
            bands_path['tbb_16'][latitude_range, :][:, longitude_range]
        ]
        # print(bands[0].shape)
        
        # 叠加 bands 数据
        bands = np.stack(bands, axis=-1)

        bands = bands.reshape(bands.shape[2], bands.shape[0], bands.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        # print(bands.shape)
        # print(label.shape)
        return bands, label
        # return 1
    
    def __len__(self):
        # 返回訓練集大小
        return len(self.bands_path)
    
    def extract_color(self, image):
        # 定义颜色映射
        colors_hex = [
            "#000000", "#07FDFD", "#0695FD", "#0203F9", "#00FF00", 
            "#00C800", "#019500", "#FEFD02", "#FEC801", "#FD7A00", 
            "#FB0100", "#C70100", "#950100", "#FA03FA", "#9800F6"
        ]
        # dBZ 阈值
        dbz_values = np.linspace(0, 65, len(colors_hex))
        # 将十六进制颜色转换为 RGB
        colors_rgb = [tuple(int(h[i:i+2], 16) for i in (1, 3, 5)) for h in colors_hex]
        image_data = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        # 创建一个与原始图像同形状的新数组
        new_image_data = np.zeros_like(image_data[:, :, 0])

        # 将颜色列表转换为 Numpy 数组以便使用广播
        colors_array = np.array(colors_rgb)
        # 检查是否有 alpha 通道
        if image_data.shape[2] == 4:
            has_alpha = True
        else:
            has_alpha = False
            raise ValueError("Image does not have an alpha channel.")

        # 通过颜色和透明度处理每个像素
        for i in range(image_data.shape[0]):
            for j in range(image_data.shape[1]):
                    if has_alpha and image_data[i, j][3] == 0:
                        # 如果像素是透明的，则设置 dBZ 为一个特定的值，例如 -1
                        new_image_data[i, j] = 0
                    else:
                        # 对于非透明像素，找到最接近的 dBZ 值
                        new_image_data[i, j] = self.find_nearest_dbz(image_data[i, j], colors_array, dbz_values)

        return new_image_data

    def find_nearest_dbz(self, pixel, colors, dbz_values):
        distances = np.sqrt(np.sum((colors - pixel[:3]) ** 2, axis=1))
        index_of_smallest = np.argmin(distances)
        return dbz_values[index_of_smallest]
    
    

if __name__ == "__main__":
    rasa_dataset = RASA_Loader("/home/aclab/Biechee/UNET/Main_Use/data/band_rader_Near_size/train")
    print("數據個數：", len(rasa_dataset))
    print(rasa_dataset[7])
    