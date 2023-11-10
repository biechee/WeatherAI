
from model.unet_model import UNet
from utils.Near_dataset_dbz import RASA_Loader
from torch import optim
import torch.nn as nn
import torch
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_net(net, device, data_path, val_data_path, epochs=50, batch_size=4, lr=0.00001):
    # 加载训练集
    rasa_dataset = RASA_Loader(data_path)
    val_dataset = RASA_Loader(val_data_path)
    

    train_loader = torch.utils.data.DataLoader(dataset=rasa_dataset,
                                               batch_size=batch_size, 
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    # 定义RMSprop算法
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8, betas=(0.9, 0.999))
    # 定义Loss算法 
    criterion = nn.CrossEntropyLoss()
    # best_loss统计，初始化为正无穷
    best_val_loss = float('inf')
    # 儲存loss
    train_losses = []
    val_losses = []
    # 训练epochs次  
    for epoch in tqdm(range(epochs)):
        # 训练模式
        net.train()
        running_loss = 0.0
        # 按照batch_size开始训练
        for bands, label in tqdm(train_loader):
            # print(label.shape)
            # print(bands.shape)
            # time.wait(21)
            # _ = input()
            optimizer.zero_grad()
            # 将数据拷贝到device中
            bands = bands.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            pred = net(bands)
            # 计算loss
            loss = criterion(pred, label)
            
            # 更新参数
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print('Loss/train', epoch_loss)

        # 验证模式
        net.eval()
        running_loss = 0.0
        with torch.no_grad():
            for bands, label in val_loader:
                # print(label.shape)
                # time.wait(21)
                bands = bands.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
                pred = net(bands)
                loss = criterion(pred, label)
                running_loss += loss.item()
        epoch_loss = running_loss / len(val_loader)
        val_losses.append(epoch_loss)
        print('Loss/val', epoch_loss)
        # 保存验证损失最小的网络参数
        if epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
            torch.save(net.state_dict(), '/home/aclab/Biechee/UNET/Main_Use/best_model_val.pth')
            

    # 繪製訓練和驗證損失的折線圖
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True)
    # 設定圖片儲存路徑和檔名
    save_path = '/home/aclab/Biechee/UNET/Main_Use/loss_plot.png'  # 指定儲存路徑和檔名

    # 儲存圖片
    plt.savefig(save_path)
    

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道1，分类为1。
    net = UNet(n_channels=4, n_classes=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 指定训练集地址，开始训练
    data_path = "/home/aclab/Biechee/UNET/Main_Use/data/band_rader_Near_size/train"
    val_data_path = "/home/aclab/Biechee/UNET/Main_Use/data/band_rader_Near_size/val"
    #查看當前資料夾
    # print(os.getcwd())
    # print(os.listdir(data_path))
    train_net(net, device, data_path, val_data_path)
