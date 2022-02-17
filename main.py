from dataset.dataset import dataset
import torch.utils.data as data_utils
import torch
import torch.nn as nn
import torch.optim as optim
from letCNN import CNN_net
data = data_utils.DataLoader(dataset=dataset(False),batch_size=1,shuffle=False)
data = list(enumerate(data, 0))

critetion = nn.BCELoss()
gpu_avi = torch.cuda.is_available()
def train(model, times, j):
    model.train()
    optimizer = optim.Adam(model.parameters(),lr=0.0001,betas=(0.9,0.999),weight_decay=0.0005)
    loss_sum = 0
    for index in range(10):
        for i in range(100):
            if int(i / 10) != j:
                optimizer.zero_grad()
                _, (inputs, y_pred) = data[i]
                if gpu_avi:
                    inputs = inputs.to("cuda:0")
                    y_pred = y_pred.to("cuda:0")
                outputs = model(inputs[0],gate=False)
                # if gpu_avi:
                #     outputs = outputs.to("cuda:0")
                # print(outputs, y_pred)
                loss = critetion(outputs, y_pred)
                loss_sum += loss.item()
                
                loss.backward()
                optimizer.step()
    print("the train loss [%d] , [%d]: %2.7f" % (times, j , loss_sum / (90 * 10)))
def test(model, times, j , ra):
    model.eval()
    loss_num = 0
    accry = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(10 * j , 10* j + 10):
        _, (inputs, y_pred) = data[i]
        
        if gpu_avi:
            inputs = inputs.to("cuda:0")
            y_pred = y_pred.to("cuda:0")
        outputs = model(inputs[0],gate=False)
        print(outputs)
        # if gpu_avi:
        #     outputs = outputs.to("cuda:0")
        # if (outputs[0].data > 0.5 and y_pred[0].data > 0.9) or (outputs[0].data < 0.5 and y_pred[0].data < 0.1):
        #     accry += 1
        if (outputs[0].data > 0.5 and y_pred[0].data > 0.9):
            TP += 1
        elif (outputs[0].data > 0.5 and y_pred[0].data < 0.1):
            FN += 1
        elif (outputs[0].data < 0.5 and y_pred[0].data > 0.9):
            FP += 1
        else :
            TN += 1
        loss = critetion(outputs, y_pred)
#        loss_num += loss.item()
#    print("the test loss: %2.7f" % (loss_num / 10))
    accuracy = (TP + TN) / 10
    if (TP + FP) > 0:
        precision = TP / (TP + FP)
    else:
        precision = 1.
    if (TP + FN) > 0:
        recall = TP / (TP + FN)
    else:
        recall = 1.
    if(TN + FP) > 0 :
        sp = TN / (TN +FP)
    else:
        sp = 1.
    print("%d accrucy ,precision, recall,SP is %2.5f , %2.5f, %2.5f , %2.5f" % (ra , accuracy ,precision, recall, sp ))
    return accuracy, precision,recall,sp
if __name__=="__main__":
#    model = CNN_net()
    num = 0
    accuracy = 0
    precision = 0
    recall = 0
    sp = 0
    for i in range(5):
        a = 0
        b = 0
        c = 0
        d = 0
        for j in range(10):
            model = CNN_net()
            if gpu_avi:
                model.to("cuda:0")
            for ra in range(20):
                train(model, i, j)
                accu , pre, rec , s= test(model, i , j , ra)
            a += accu
            b += pre
            c += rec
            d += s
        accuracy += a
        precision += b
        recall += c
        sp += d
        print("accuracy rande %d is %2.5f %% , %2.5f %% , %2.5f %% , %2.5f %%" % (i , 10 * a , 10 * b , 10 * c, 10 * d))
    print("the 5 times is %2.5f %% , %2.5f %% , %2.5f %% , %2.5f %%" % (2 * accuracy , 2 * precision , 2 * recall, 2 * sp))