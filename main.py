from dataset.dataset import dataset
import torch.utils.data as data_utils
import torch
import torch.nn as nn
import torch.optim as optim
from letCNN import CNN_net
from k_fold_cross import make_k_fold
from imgloader import loader_img
from loader_data import loader
# data = data_utils.DataLoader(dataset=dataset(False),batch_size=1,shuffle=False)
# data = list(enumerate(data, 0))
critetion = nn.BCELoss()
gpu_avi = torch.cuda.is_available()
num = list(range(100))
train_folds, test_folds = make_k_fold(10,num)
def train(model, i, j):
    model.train()
    optimizer = optim.Adam(model.parameters(),lr=0.0001,betas=(0.9,0.999),weight_decay=0.0005)
    dir = train_folds[i]
    data = data_utils.DataLoader(dataset=loader(dir,train=True),batch_size=1,shuffle=False)
    loss_all = 0
    for idx in range(10):
        for _ , (inputs, target) in enumerate(data,0):
            if gpu_avi:
                inputs = inputs.to("cuda:0")
                target = target.to("cuda:0")
            outputs = model(inputs[0],gate=True)
            loss = critetion(outputs,target)
            loss_all += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    print('[%d] , [%d] train loss is %2.5f' % (j + 1, i + 1, loss_all / 1800))

def test(model, i, j , call):
    model.eval()
    dir = test_folds[i]
    data = data_utils.DataLoader(dataset=loader(dir,train=False),batch_size=1,shuffle=False)
    accuracy = 0
    for _, (inputs,target) in enumerate(data, 0):
        if gpu_avi:
            inputs = inputs.to("cuda:0")
            target = target.to("cuda:0")
        output = model(inputs[0],gate=True)
        print(output, target)
        if output[0].data > 0.5:
            if target[0].data > 0.9:
                accuracy += 1
        else:
            if target[0].data < 0.1:
                accuracy += 1
    print('[%d] accuracy [%d] is [%2.5f]' % (call, i + 1, accuracy / len(dir)))
    return accuracy
if __name__=="__main__":
    ende = 0.
    for idx in range(5):
        num = 0.
        for i in range(10):
            model = CNN_net()
            if gpu_avi:
                model = model.to("cuda:0")
            for j in range(10):
                train(model,i,j)
                acc = test(model, i, j , idx)
            num += acc
        print("the [%d] acc is [%2.5f]" % (i , num / 10))
        ende += num
    print("the resut is %2.5f" % (ende / 5))