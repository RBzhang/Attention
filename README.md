# AMIL
复现论文"Attention-based Deep Multiple Instance Learning"<br>
## 文件介绍

k_fold_cross.py是拆分数据集的代码，用于十折交叉验证<br>
addition_trainf.py是对图像进行预处理的模块<br>
imgloader.py是从原始图像中提取数据并进行处理的代码，将图像以Tensor的形式加载到内存中<br>
loader_data.py利用上一个模块的数据加载数据<br>
letCNN.py文件是模型代码<br>
main.py文件是主程序<br>

## 结果分析

下面只列出COLON CANCER数据集的实验结果<br>

|MOTHOD|Accracy|
|----|----|
|attention|90.400%|
|attention-gated|89.000%|