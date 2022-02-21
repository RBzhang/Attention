from operator import attrgetter
from turtle import forward
import torch
import torch.utils.data as data_utils
import torch.nn as nn
from dataset.dataset import dataset
import torch.nn.functional as F
class CNN_net(torch.nn.Module):
    def __init__(self) -> None:
        super(CNN_net,self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3,36,kernel_size=4,padding=0), nn.ReLU(), nn.MaxPool2d(2,2),
        nn.Conv2d(36,48,kernel_size=3,padding=0), nn.ReLU(),nn.MaxPool2d(2,2))
        self.layer2 = nn.Sequential(nn.Linear(48 * 5 * 5, 512,bias=False), nn.ReLU(), nn.Dropout(p=0.5))
        self.layer3 = nn.Sequential(nn.Linear(512,512,bias=False), nn.ReLU(), nn.Dropout(p=0.5))
        self.linearV = nn.Linear(512,128,bias=False)
        self.linearU = nn.Linear(512,128,bias=False)
        self.attention = nn.Linear(128,1,bias=False)
        # self.softmax = nn.Softmax(dim=1)
        self.layer4 = nn.Linear(512,1,bias=False)
    def forward(self,x , gate = True):
        x = self.layer1(x)
        x = x.view(-1, 48 * 5 * 5)
        # print(x.shape)
        # x = x.view(361,-1)
        # print(x[0])
        x = self.layer2(x)
        x = self.layer3(x)
        # print("x-1")
        # print(x[0])
        if gate:
            a = torch.tanh(self.linearV(x)) * torch.sigmoid(self.linearU(x))
        else:
            a = torch.tanh(self.linearV(x))
        a = self.attention(a)
        a = torch.transpose(a, 0,1)
        # print(a)
        a = torch.softmax(a,1)

        x = torch.mm(a , x)
        # print("x")
        # print(x)
        
        x = self.layer4(x)
        # print("x1")
        # print(x)
        # print("w")
        # print(self.layer4.weight)
        x = torch.sigmoid(x)
        # print("w")
        # print(self.layer4[])
        # print(x.shape)
        return x[0]

if __name__=="__main__":
    # net = CNN_net()
    # x = dataset(False)
    # x = data_utils.DataLoader(dataset=x, batch_size=1,shuffle=False)
    # _ , data = list(enumerate(x , 0))[0]
    # x , y = data
    # # x = torch.rand((361,3,27,27))
    # x = net(x[0])
    x = torch.Tensor([2])
    y = torch.sigmoid(x)
    print(x)
    print(y)