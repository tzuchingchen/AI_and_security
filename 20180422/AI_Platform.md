


# 使用Ubuntu_16.04_64bit_TensorFlow_jupyter.ova image
### 建置
```
Ubuntu 16.04_64bit
CPU:2
RAM:2G
Python:2.7  ?? Python:3.5
```
## 安裝 Tensorflow

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

