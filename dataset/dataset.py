import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from dataset.loader import ReadBMPFile
import os
import scipy.io as io
import random
class dataset(data_utils.Dataset):
    def __init__(self, from_img = True) -> None:
        super().__init__()
        self.x_data = []
        self.y_data = []
#        file = '../data/img1/img1.bmp'
        if from_img:
            num = list(range(1,101))
            random.shuffle(num)
            data1 = []
            for i in num:
                strimg = 'img' + str(i)
                file1 = '../data/' + strimg + '/' + strimg + '.bmp'
                file_y = '../data/' + strimg + '/' + strimg + '_epithelial.mat'
                filepath = os.path.join(os.path.dirname(__file__) + '/' + file1)
                file_y = os.path.join(os.path.dirname(__file__) + '/' + file_y)
                data = ReadBMPFile(filepath)
                rotate = torch.rand(361)
                mirrir = torch.rand(361)
                # print(data.R[0])
                r = np.array(data.R, dtype=np.float32) / 256.
                g = np.array(data.G, dtype=np.float32) / 256.
                b = np.array(data.B, dtype=np.float32) / 256.
                # img1 = [r , g, b]
                img1 = [0.65*r+0.70*g+0.29*b,0.07*r+0.99*g+0.11*b,0.27*r+0.57*g+0.78*b] #转换到H&E空间
                img1 = np.array(img1, dtype=np.float32).copy()
    #            print(img1[0])
                # num1 = torch.normal(1,1,(3,500,500))
                # num2 = torch.normal(1,1,(3,500,500))
    #            print(num1[0])
                img1 = torch.from_numpy(img1)
                # img1 = img1 * num1 * num2
                zero1 = torch.zeros((3,500,13),dtype=torch.float32)
                zero2 = torch.zeros((3,13,513),dtype=torch.float32)
                img1 = torch.cat((img1, zero1), dim=2).clone()
                img1 = torch.cat((img1,zero2),dim=1).clone()
                img1 = torch.chunk(img1, 19, dim=1)
                img1 = torch.stack(img1, dim=0).clone()
                img1 = torch.chunk(img1, 19, dim=3)
                img1 = torch.cat(img1, dim=0).clone()
                for t in range(361):  #随机镜像反转
                    if rotate[t] > 0.5:
                        img1[t].transpose(1,2)
                    if mirrir[t] > 0.5:
                        img1[t] = torch.flip(img1[t],dims=[2])
#                self.x_data.append(img1.clone())
                data1.append(img1.numpy().copy())
                print(io.loadmat(file_y)['detection'].shape)
                self.y_data.append(io.loadmat(file_y)['detection'].shape[0] != 0)
                # data1.append(np.array([img1.clone(), self.y_data[i-1]]))
            data1 = np.array(data1).copy()
            self.y_data = np.array(self.y_data,dtype=np.float32).copy()
            self.x_data = torch.from_numpy(data1).clone()
            io.savemat('xdata.mat',{'data':data1}) # 存入指定文件中，下次提取加快速度
            io.savemat('ydata.mat',{'data':self.y_data})
            self.y_data = torch.from_numpy(self.y_data).clone()
            # print(self.y_data)
        else:
            data1 = io.loadmat('xdata.mat')['data']
            data2 = io.loadmat('ydata.mat')['data']
            self.x_data = torch.Tensor(data1).clone()
            self.y_data = torch.Tensor(data2)[0].clone()
        print(self.y_data)
        print(self.x_data.shape)
        print(self.x_data[0,1])
        print(self.x_data[1,1])
#        print(img1[0][0])
    def __len__(self):
        return 100
    def __getitem__(self, index):
        return self.x_data[index] , self.y_data[index]
if __name__ == "__main__":
    d = dataset(True)
