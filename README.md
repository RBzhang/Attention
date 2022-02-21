# AMIL
复现论文"Attention-based Deep Multiple Instance Learning"<br>
## 文件介绍

loader.py是读取.bmp图像文件的代码<br>
dataset.py是利用从图像文件读取出来的数据构建神经网络的输入数据代码<br>
letCNN.py文件是模型代码<br>
main.py文件是主程序<br>

## 结果分析

下面只列出COLON CANCER数据集的实验结果<br>

|MOTHOD|Accracy|
|----|----|----|----|----|
|attention|90.400%|
|attention-gated|85.000%|