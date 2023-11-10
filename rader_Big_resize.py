import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from PIL import Image
from tqdm import tqdm

# 输入文件夹和输出文件夹路径
input_folder = '/home/aclab/Biechee/UNET/Main_Use/data/band_rader_Big_size/temp'
output_folder = '/home/aclab/Biechee/UNET/Main_Use/data/band_rader_Big_size/label'

# 获取输入文件夹中的所有文件
file_list = os.listdir(input_folder)

# 循环处理每个文件
for file_name in tqdm(file_list):
    # 仅处理图像文件，可以根据需要扩展支持的文件类型
    if file_name.endswith('.png'):
        # 读取雷达图像
        rader_image = plt.imread(os.path.join(input_folder, file_name))

        # 创建新的Figure
        fig = plt.figure()

        # 添加地图和轴
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        # 设置地图范围，这里使用了台湾的经纬度范围
        map_extent = [114.48, 126.96, 18.74, 31.2]
        rader_extent = [115.0, 126.5, 17.75, 29.25]  # 雷达图像的经纬度范围
        ax.set_extent(map_extent)

        # 调整Figure的大小以匹配合并后的图像大小
        fig = plt.gcf()
        fig.set_size_inches(624 / 100, 624 / 100)

        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)

        # 从数组中创建图像并调整大小
        new_rader_image = Image.fromarray((rader_image * 255).astype('uint8')).resize((624, 624))

        # 将黑色像素转换为透明像素
        new_rader_image = new_rader_image.convert("RGBA")
        data = new_rader_image.getdata()
        new_data = []
        for item in data:
            if item[0] == 0 and item[1] == 0 and item[2] == 0:
                new_data.append((0, 0, 0, 0))  # 将黑色像素转换为透明像素
            else:
                new_data.append(item)
        new_rader_image.putdata(new_data)
        plt.axis('off')

        # 在地图上显示处理后的雷达图像
        ax.imshow(new_rader_image, origin='upper', extent=rader_extent, transform=ccrs.PlateCarree(), zorder=2)

        # 保存处理后的图像到输出文件夹
        output_file = os.path.join(output_folder, file_name)
        plt.savefig(output_file, format='png', transparent=True, pad_inches=0)

        plt.close()  # 关闭当前图形窗口以释放内存
