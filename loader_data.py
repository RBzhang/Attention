from imgloader import loader_img
import torch
import numpy as np
import torch.utils.data as data_util
from k_fold_cross import make_k_fold
data_train = loader_img(train=True,shaffle_bag=True)
data_test = loader_img(train=False,shaffle_bag=False)
class loader(data_util.Dataset):
    def __init__(self, folds , train = True) -> None:
        super().__init__()
        r = np.random.RandomState(3)
        if train:
            data = data_train
        else:
            data = data_test
        # print(data.img[0])
        # print(data.img[0].shape)
        self.x_data = []
        self.y_data = []
        for i in folds:
            self.x_data.append(data.img[i])
            self.y_data.append(data.label[i][0])
            if train:
                self.x_data.append(data.img[i])
                self.y_data.append(data.label[i][0])
        bag = list(zip(self.x_data,self.y_data))
        r.shuffle(bag)
        self.x_data, self.y_data = zip(*bag)
        self.y_data = torch.Tensor(self.y_data)
        print(self.y_data.shape)
    def __len__(self):
        return len(self.y_data)
    def __getitem__(self, index) :
        return self.x_data[index] , self.y_data[index]
if __name__=="__main__":
    x = list(range(100))
    t, e = make_k_fold(10,x)
    print(t)
    print(e)
    t = loader(t[0],train=True)
