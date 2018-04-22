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