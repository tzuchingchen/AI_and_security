# 下載教科書程式碼

git clone https://github.com/hzy46/Deep-Learning-21-Examples.git
```
Cloning into 'Deep-Learning-21-Examples'...
remote: Enumerating objects: 7004, done.
remote: Total 7004 (delta 0), reused 0 (delta 0), pack-reused 7004
Receiving objects: 100% (7004/7004), 138.53 MiB | 6.27 MiB/s, done.
Resolving deltas: 100% (161/161), done.
Checking connectivity... done.
```
```
Deep-Learning-21-Examples  Downloads         Pictures         Videos
Desktop                    examples.desktop  Public
Documents                  MNIST_data        Templates
download.py                Music             TensorFlow_Code
```
```
ksu@ksu:~$ cd Deep-Learning-21-Examples/

ksu@ksu:~/Deep-Learning-21-Examples$ ls
chapter_1   chapter_13  chapter_17  chapter_20  chapter_5  chapter_9
chapter_10  chapter_14  chapter_18  chapter_21  chapter_6  README.md
chapter_11  chapter_15  chapter_19  chapter_3   chapter_7
chapter_12  chapter_16  chapter_2   chapter_4   chapter_8

ksu@ksu:~/Deep-Learning-21-Examples$ cd chapter_1

ksu@ksu:~/Deep-Learning-21-Examples/chapter_1$ ls
convolutional.py  label.py   save_pic.py
download.py       README.md  softmax_regression.py
```

# MNIST 機器學習入門

### 下載MNIST資料集，並列印一些基本資訊：python download.py
```
ksu@ksu:~/Deep-Learning-21-Examples/chapter_1$ python download.py 
Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
Extracting MNIST_data/train-images-idx3-ubyte.gz
Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
(55000, 784)
(55000, 10)
(5000, 784)
(5000, 10)
(10000, 784)
(10000, 10)
[ 0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.38039219  0.37647063
  0.3019608   0.46274513  0.2392157   0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.35294119  0.5411765
  0.92156869  0.92156869  0.92156869  0.92156869  0.92156869  0.92156869
  0.98431379  0.98431379  0.97254908  0.99607849  0.96078438  0.92156869
  0.74509805  0.08235294  0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.
  0.54901963  0.98431379  0.99607849  0.99607849  0.99607849  0.99607849
  0.99607849  0.99607849  0.99607849  0.99607849  0.99607849  0.99607849
  0.99607849  0.99607849  0.99607849  0.99607849  0.74117649  0.09019608
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.88627458  0.99607849  0.81568635
  0.78039223  0.78039223  0.78039223  0.78039223  0.54509807  0.2392157
  0.2392157   0.2392157   0.2392157   0.2392157   0.50196081  0.8705883
  0.99607849  0.99607849  0.74117649  0.08235294  0.          0.          0.
  0.          0.          0.          0.          0.          0.
  0.14901961  0.32156864  0.0509804   0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.13333334  0.83529419  0.99607849  0.99607849  0.45098042  0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.32941177  0.99607849  0.99607849  0.91764712  0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.32941177  0.99607849  0.99607849  0.91764712  0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.41568631  0.6156863   0.99607849  0.99607849  0.95294124  0.20000002
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.09803922  0.45882356  0.89411771
  0.89411771  0.89411771  0.99215692  0.99607849  0.99607849  0.99607849
  0.99607849  0.94117653  0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.26666668  0.4666667   0.86274517
  0.99607849  0.99607849  0.99607849  0.99607849  0.99607849  0.99607849
  0.99607849  0.99607849  0.99607849  0.55686277  0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.14509805  0.73333335  0.99215692
  0.99607849  0.99607849  0.99607849  0.87450987  0.80784321  0.80784321
  0.29411766  0.26666668  0.84313732  0.99607849  0.99607849  0.45882356
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.44313729
  0.8588236   0.99607849  0.94901967  0.89019614  0.45098042  0.34901962
  0.12156864  0.          0.          0.          0.          0.7843138
  0.99607849  0.9450981   0.16078432  0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.66274512  0.99607849  0.6901961   0.24313727  0.          0.
  0.          0.          0.          0.          0.          0.18823531
  0.90588242  0.99607849  0.91764712  0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.07058824  0.48627454  0.          0.          0.
  0.          0.          0.          0.          0.          0.
  0.32941177  0.99607849  0.99607849  0.65098041  0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.54509807  0.99607849  0.9333334   0.22352943  0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.
  0.82352948  0.98039222  0.99607849  0.65882355  0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.94901967  0.99607849  0.93725497  0.22352943  0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.
  0.34901962  0.98431379  0.9450981   0.33725491  0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.
  0.01960784  0.80784321  0.96470594  0.6156863   0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.01568628  0.45882356  0.27058825  0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.        ]
[ 0.  0.  0.  0.  0.  0.  0.  1.  0.  0.]

```


### 將MNIST資料集保存為圖片:python save_pic.py
```
ksu@ksu:~/Deep-Learning-21-Examples/chapter_1$ python save_pic.py
Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
Traceback (most recent call last):
  File "save_pic.py", line 25, in <module>
    scipy.misc.toimage(image_array, cmin=0.0, cmax=1.0).save(filename)
AttributeError: 'module' object has no attribute 'toimage'

```
### 圖像標籤的獨熱表示

列印MNIST資料集中圖片的標籤：python label.py
```
ksu@ksu:~/Deep-Learning-21-Examples/chapter_1$ python label.py
Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
mnist_train_0.jpg label: 7
mnist_train_1.jpg label: 3
mnist_train_2.jpg label: 4
mnist_train_3.jpg label: 6
mnist_train_4.jpg label: 1
mnist_train_5.jpg label: 8
mnist_train_6.jpg label: 1
mnist_train_7.jpg label: 0
mnist_train_8.jpg label: 9
mnist_train_9.jpg label: 8
mnist_train_10.jpg label: 0
mnist_train_11.jpg label: 3
mnist_train_12.jpg label: 1
mnist_train_13.jpg label: 2
mnist_train_14.jpg label: 7
mnist_train_15.jpg label: 0
mnist_train_16.jpg label: 2
mnist_train_17.jpg label: 9
mnist_train_18.jpg label: 6
mnist_train_19.jpg label: 0

```
### Softmax 回歸:python softmax_regression.py
```
ksu@ksu:~/Deep-Learning-21-Examples/chapter_1$ python softmax_regression.py
Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
2018-10-02 21:34:55.271561: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2018-10-02 21:34:55.271697: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2018-10-02 21:34:55.271761: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
start training...
0.9018
```
### 兩層卷積網路分類:python convolutional.py
```
ksu@ksu:~/Deep-Learning-21-Examples/chapter_1$ python convolutional.py
Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
2018-10-02 21:35:13.497129: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2018-10-02 21:35:13.497189: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2018-10-02 21:35:13.497206: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
step 0, training accuracy 0.1
step 100, training accuracy 0.86
step 200, training accuracy 0.86
step 300, training accuracy 0.98
step 400, training accuracy 0.92
step 500, training accuracy 1
step 600, training accuracy 0.94
step 700, training accuracy 0.96
step 800, training accuracy 0.98
step 900, training accuracy 0.94
step 1000, training accuracy 1
step 1100, training accuracy 0.94
step 1200, training accuracy 0.98
step 1300, training accuracy 0.98
step 1400, training accuracy 0.94
step 1500, training accuracy 1
step 1600, training accuracy 0.98
step 1700, training accuracy 0.96
step 1800, training accuracy 0.92
step 1900, training accuracy 0.96
step 2000, training accuracy 1
step 2100, training accuracy 1
step 2200, training accuracy 1
step 2300, training accuracy 0.96
step 2400, training accuracy 0.98
step 2500, training accuracy 0.98
step 2600, training accuracy 0.98
step 2700, training accuracy 0.98
step 2800, training accuracy 0.98

```
