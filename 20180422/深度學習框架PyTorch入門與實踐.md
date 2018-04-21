# 深度學習框架PyTorch入門與實踐
```
https://github.com/chenyuntc/pytorch-book
```

```
1 PyTorch簡介
1.1 PyTorch的誕生
1.2常見的深度學習框架簡介
1.2.1 Theano 
1.2.2 TensorFlow 
1.2.3 Keras 
1.2.4 Caffe/Caffe2 
1.2.5 MXNet 
1.2.6 CNTK 
1.2.7其他框架
1.3屬於動態圖的未來
1.4為什麼選擇PyTorch 
1.5星火燎原
1.6 fast.ai放棄Keras+TensorFlow選擇PyTorch 
```
```
2快速入門
2.1安裝與配置
2.1.1安裝PyTorch 
[1]使用pip安裝

[2]使用anaconda安裝
conda install pytorch torchversion -c soumith

官方網址:conda install pytorch torchvision -c pytorch
http://pytorch.org/

官方範例:https://github.com/pytorch/examples
```
### PyTorch Examples

A repository showcasing examples of using [PyTorch](https://github.com/pytorch/pytorch)

- MNIST Convnets
- Word level Language Modeling using LSTM RNNs
- Training Imagenet Classifiers with Residual Networks
- Generative Adversarial Networks (DCGAN)
- Variational Auto-Encoders
- Superresolution using an efficient sub-pixel convolutional neural network
- Hogwild training of shared ConvNets across multiple processes on MNIST
- Training a CartPole to balance in OpenAI Gym with actor-critic
- Natural Language Inference (SNLI) with GloVe vectors, LSTMs, and torchtext
- Time sequence prediction - create an LSTM to learn Sine waves

Additionally, a list of good examples hosted in their own repositories:

- [Neural Machine Translation using sequence-to-sequence RNN with attention (OpenNMT)](https://github.com/OpenNMT/OpenNMT-py)

```
2.1.2學習環境配置
2.2 PyTorch入門第一步
2.2.1 Tensor 
2.2.2 Autograd ：自動微分
2.2.3神經網絡
2.2.4小試牛刀：CIFAR-10分類

3 Tensor和autograd 
3.1 Tensor 
3.1.1基礎操作
3.1.2 Tensor和Numpy 
3.1.3內部結構
3.1.4其他有關Tensor的話題
3.1.5小試牛刀：線性回歸
3.2 autograd 
3.2.1 Variable 
3.2.2計算圖
3.2.3擴展autograd 
3.2.4小試牛刀：用Variable實現線性回歸

4神經網絡工具箱nn 
4.1 nn.Module 
4.2常用的神經網絡層
4.2.1圖像相關層
4.2.2激活函數
4.2.3循環神經網絡層
4.2.4損失函數
4.3優化器
4.4 nn.functional 
4.5初始化策略
4.6 nn.Module深入分析
4.7 nn和autograd的關係
4.8小試牛刀：用50行代碼搭建ResNet 

5 PyTorch中常用的工具
5.1數據處理
5.2計算機視覺工具包：torchvision 
5.3可視化工具
5.3.1 Tensorboard 
5.3.2 visdom 
5.4使用GPU加速：cuda 
5.5持久化

6 PyTorch實戰指南
6.1編程實戰：貓和狗二分類
6.1.1比賽介紹
6.1.2文件組織架構
6.1.3關於__init__.py 
6.1.4數據加載
6.1.5模型定義
6.1.6工具函數
6.1.7配置文件
6.1.8 main.py 
6.1.9使用
6.1 .10爭議
6.2 PyTorch Debug指南
6.2.1 ipdb介紹
6.2.2在PyTorch中Debug 

7 AI插畫師：生成對抗網絡
7.1 GAN的原理簡介
7.2用GAN生成動漫頭像
7.3實驗結果分析

8 AI藝術家：神經網絡風格遷移
8.1風格遷移原理介紹
8.2用PyTorch實現風格遷移
8.3實驗結果分析

9 AI詩人：用RNN寫詩
9.1自然語言處理的基礎知識
9.1.1詞向量
9.1.2 RNN 
9.2 CharRNN 
9.3用PyTorch實現CharRNN 
9.4實驗結果分析

10 Image Caption：讓神經網絡看圖講故事
10.1圖像描述介紹
10.2數據
10.2.1數據介紹
10.2.2圖像數據處理
10.2.3數據加載
10.3模型與訓練
10.4實驗結果分析

11展望與未來
11.1 PyTorch的局限與發展
11.2使用建議
```
