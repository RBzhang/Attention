from PIL import Image
import numpy as np
import scipy.io as io
import random
import addition_trainf
import torch
import torchvision.transforms as transforms

class loader_img:
    transform_train = transforms.Compose([
        addition_trainf.RandomHEStain(),
        addition_trainf.HistoNormalize(),
        addition_trainf.RandomRotate(),
        addition_trainf.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
    transform_test = transforms.Compose([
        addition_trainf.HistoNormalize(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    def __init__(self, train=False, shaffle_bag=False) -> None:
        super().__init__()
        if train:
            normal_form = self.transform_train
        else:
            normal_form = self.transform_test
        self.img = []
        self.label = []
        for dir in range(1,101):
            i = 'img' + str(dir)
            file_bmp = './data/' + i + '/' + i + '.bmp'
            with open(file_bmp,'rb') as f:
                with Image.open(f) as img:
                    img = img.convert('RGB')
            dir_epithelial = './data/' + i + '/' + i + '_epithelial.mat'
            with open(dir_epithelial,'rb') as f:
                mat_epithelial = io.loadmat(f)['detection']
            label_cell = []
            for (x,y) in mat_epithelial:
                x = np.round(x)
                y = np.round(y)
                if train:
                    x = x + np.round(np.random.normal(0,3,1)[0])
                    y = y + np.round(np.random.normal(0,3,1)[0])
                    # print(x)
                if x < 13:
                    x = 0
                elif x > 500 - 14:
                    x = 500 - 27
                else:
                    x = x - 13
                if y < 13:
                    y = 0
                elif y > 500 - 14:
                    y = 500 - 27
                else:
                    y = y - 13
                label_cell.append(normal_form(img.crop((x,  y , x+27,y + 27))))
            dir_fibroblast = './data/' + i + '/' + i + '_fibroblast.mat'
            dir_inflammatory = './data/' + i + '/' + i + '_inflammatory.mat'
            dir_others = './data/' + i + '/' + i + '_others.mat'
            with open(dir_fibroblast,'rb') as f:
                mat_fibroblast = io.loadmat(f)['detection']
            with open(dir_inflammatory,'rb') as f:
                mat_inflammatory = io.loadmat(f)['detection']
            with open(dir_others, 'rb') as f:
                mat_others = io.loadmat(f)['detection']
            all_others = np.concatenate((mat_fibroblast,mat_inflammatory, mat_others),axis=0)
            others_cell = []
            for (x,y) in all_others:
                x = np.round(x)
                y = np.round(y)
                if train:
                    x = x + np.round(np.random.normal(0,3,1)[0])
                    y = y + np.round(np.random.normal(0,3,1)[0])
                if x < 13:
                    x = 0
                elif x > 500 - 14:
                    x = 500 - 27
                else:
                    x = x - 13
                if y < 13:
                    y = 0
                elif y > 500 - 14:
                    y = 500 - 27
                else:
                    y = y - 13
                others_cell.append(normal_form(img.crop((x,y , x+27,  y + 27))))
            cell = label_cell + others_cell
            
            label = np.concatenate((np.ones(len(label_cell)),np.zeros(len(others_cell))),axis=0)
            if shaffle_bag:
                zip_bag = list(zip(cell,label))
                random.shuffle(zip_bag)
                cell, label = zip(*zip_bag)
            cell = torch.stack(cell)
            if train:
                self.img.append(cell)
                self.label.append([max(label),label])
            else:
                self.img.append(cell)
                self.label.append([max(label),label])

if __name__=="__main__":
    t = loader_img(train=True, shaffle_bag=True)
    print(t.img[0].shape)
    print(t.img[1].shape)