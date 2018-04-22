# 使用Ubuntu_16.04_64bit_TensorFlow_jupyter.ova image


```
作業系統:Ubuntu 16.04_64 bit
硬體:CPU only
```
### 建置過程(1)安裝anacoda
```
Ubuntu 16.04_64 bit
CPU:2
RAM:2G
Python:2.7  ?? Python:3.5
```
### 建置過程(2)安裝 Tensorflow

```
sudo apt-get update
sudo apt-get install libcupti-dev
sudo apt-get install python-py python-dev 
sudo apt-get install python-pip
sudo pip install --upgrade pip
sudo pip install tensorflow
```

#### 升級 Tensorflow

# 安裝pytorch平台
```
[1]使用pip安裝
pip3 install torch torchvision

pip install torch torchvision
[2]使用anaconda安裝
conda install pytorch torchversion -c soumith

官方網址:conda install pytorch torchvision -c pytorch(在linux OK| Winodows 不OK)

[3]使用Docker安裝:Docker image
Dockerfile is supplied to build images with cuda support and cudnn v7. Build as usual

docker build -t pytorch -f docker/pytorch/Dockerfile .
You can also pull a pre-built docker image from Docker Hub and run with nvidia-docker, but this is not currently maintained and will pull PyTorch 0.2.

nvidia-docker run --rm -ti --ipc=host pytorch/pytorch:latest
```
# 安裝課本範例
```
git clone https://github.com/chenyuntc/PyTorch-book.git
cd pytorch-book 
pip install -r requirements.txt====>一些錯誤
```
進入ipython

import torch as t

# 安裝Websecurity範例

Web安全之机器学习入门

https://github.com/duoergun0729/1book

git clone https://github.com/duoergun0729/2book.git

git clone https://github.com/duoergun0729/3book.git

git clone https://github.com/duoergun0729/4book.git

git clone https://github.com/duoergun0729/nlp.git

```
MP21710	TensorFlow+Keras深度學習人工智慧實務應用	林大貴 著	978-986-434-216-7	590	2017/6/9	博碩
MP11626	深度學習快速入門—使用TensorFlow	Giancarlo Zaccone
```
```
Deep Learning：用Python進行深度學習的基礎理論實作
https://github.com/oreilly-japan/deep-learning-from-scratch
作者： 斎藤康毅  譯者： 吳嘉芳
出版社：歐萊禮  出版日期：2017/08/17
第一章 Python入門   第二章 感知器 
第三章 神經網路   第四章 神經網路的學習  第五章 誤差反向傳播法 
第六章 與學習有關的技巧  第七章 卷積神經網路  第八章 深度學習 
附錄A Softmax-with-Loss層的計算圖 
git clone https://github.com/oreilly-japan/deep-learning-from-scratch
```
```
Python3 for Data Science 
https://www.udemy.com/python-for-data-science/?couponCode=PYTHON-DATA-SCI10
https://github.com/udemy-course/python-data-science-intro
線上學習程式碼
http://nbviewer.jupyter.org/github/udemy-course/python-data-science-intro/tree/master/
```
