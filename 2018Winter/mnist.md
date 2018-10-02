#  

```
《21个项目玩转深度学习———基于TensorFlow的实践详解》
https://github.com/hzy46/Deep-Learning-21-Examples

上課用的image==>Ubuntu 16.04_64bit_TensorFlow.ova
```

https://github.com/hzy46/Deep-Learning-21-Examples/blob/master/chapter_1/download.py
```
# coding:utf-8

# 從tensorflow.examples.tutorials.mnist引入模組。這是TensorFlow為了教學MNIST而提前編制的程式
from tensorflow.examples.tutorials.mnist import input_data

# 從MNIST_data/中讀取MNIST數據。這條語句在資料不存在時，會自動執行下載
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 查看訓練數據的大小
print(mnist.train.images.shape)  # (55000, 784)
print(mnist.train.labels.shape)  # (55000, 10)

# 查看驗證資料的大小
print(mnist.validation.images.shape)  # (5000, 784)
print(mnist.validation.labels.shape)  # (5000, 10)

# 查看測試資料的大小
print(mnist.test.images.shape)  # (10000, 784)
print(mnist.test.labels.shape)  # (10000, 10)

# 列印出第0幅圖片的向量表示
print(mnist.train.images[0, :])

# 列印出第0幅圖片的標籤
print(mnist.train.labels[0, :])
```

### input_data
```
MNIST机器学习入门- TensorFlow 官方文档中文版
http://wiki.jikexueyuan.com/project/tensorflow-zh/tutorials/mnist_beginners.html

https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/input_data.py

https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py
```
### 執行畫面
```
ksu@ksu:~$ gedit download.py
ksu@ksu:~$ python2 download.py 
Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
Extracting MNIST_data/train-images-idx3-ubyte.gz
Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
..........................
```
```
ksu@ksu:~$ ls
Desktop    download.py  examples.desktop  Music     Public     TensorFlow_Code
Documents  Downloads    MNIST_data        Pictures  Templates  Videos
```
```
ksu@ksu:~$ cd MNIST_data/
ksu@ksu:~/MNIST_data$ ls
t10k-images-idx3-ubyte.gz  train-images-idx3-ubyte.gz
t10k-labels-idx1-ubyte.gz  train-labels-idx1-ubyte.gz
ksu@ksu:~/MNIST_data$ ls -al
total 11344
drwxr-xr-x  2 ksu ksu    4096  十   2 20:24 .
drwxr-xr-x 21 ksu ksu    4096  十   2 20:24 ..
-rw-rw-r--  1 ksu ksu 1648877  十   2 20:24 t10k-images-idx3-ubyte.gz
-rw-rw-r--  1 ksu ksu    4542  十   2 20:24 t10k-labels-idx1-ubyte.gz
-rw-rw-r--  1 ksu ksu 9912422  十   2 20:24 train-images-idx3-ubyte.gz
-rw-rw-r--  1 ksu ksu   28881  十   2 20:24 train-labels-idx1-ubyte.gz
```


