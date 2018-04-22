# {教科書}深度學習框架PyTorch入門與實踐
```
深度學習框架PyTorch:入門與實踐 深度学习框架PyTorch:入门与实践
陳雲 電子工業出版社  出版日期:2018-01-01

https://github.com/chenyuntc/pytorch-book
```

**基礎部分**（前五章）講解PyTorch內容，這部份介紹了PyTorch中主要的的模組，和深度學習中常用的一些工具。對於這部分內容，這裡利用Jupyter Notebook作為教學工具，讀者可以結合notebook修改運行，反復實驗。

- 第二章介紹如何安裝PyTorch和配置學習環境。同時提供了一個快速入門教程，基於官方的教程簡化並更新內容，讀者可以花費大約1到2小時的時間快速完成入門任務，而後根據需求再選擇深入閱讀後續相關章節的內容。
- 第三章介紹了PyTorch中多維陣列Tensor和動態圖autograd/Variable的使用，並配以例子，讓讀者分別使用Tensor和autograd實現線性回歸，比較二者的不同點。除了介紹這二者的基礎使用之外，本章還對Tensor的底層設計，以及autograd的計算圖原理進行比較深入分析，希望能使得讀者能對這些底層知識有更全面的掌握。
- 第四章介紹了PyTorch中神經網路模組nn的基礎用法，同時講解了神經網路中“層”，“損失函數”，“優化器”等，最後帶領讀者用不到50行的代碼搭建出曾奪得ImageNet冠軍的ResNet。
- 第五章介紹了PyTorch中資料載入，GPU加速，持久化和視覺化等相關工具。



## 基礎部分

```
1 PyTorch簡介
1.1 PyTorch的誕生
1.2常見的深度學習框架簡介
1.2.1 Theano  1.2.2 TensorFlow  1.2.3 Keras 
1.2.4 Caffe/Caffe2  1.2.5 MXNet  1.2.6 CNTK  1.2.7其他框架
1.3屬於動態圖的未來


1.4為什麼選擇PyTorch 
http://pytorch.org/about/

1.5星火燎原
1.6 fast.ai放棄Keras+TensorFlow選擇PyTorch 
```

# 2快速入門

2.1安裝與配置

## 2.1.1安裝PyTorch 

### [1]使用pip安裝
```
pip3 install torch torchvision

pip install torch torchvision
```
### [2]使用anaconda安裝
```
conda install pytorch torchversion -c soumith

官方網址:conda install pytorch torchvision -c pytorch
```

### [3]使用Docker安裝:Docker image

Dockerfile is supplied to build images with cuda support and cudnn v7. Build as usual
```
docker build -t pytorch -f docker/pytorch/Dockerfile .
```

You can also pull a pre-built docker image from Docker Hub and run with nvidia-docker,
but this is not currently maintained and will pull PyTorch 0.2.
```
nvidia-docker run --rm -ti --ipc=host pytorch/pytorch:latest
```

### [4]安裝別人打包過的套件
```
https://zhuanlan.zhihu.com/p/26871672

# for CPU only packages
conda install -c peterjc123 pytorch-cpu

# for Windows 10 and Windows Server 2016, CUDA 8
conda install -c peterjc123 pytorch

# for Windows 10 and Windows Server 2016, CUDA 9
conda install -c peterjc123 pytorch cuda90

# for Windows 7/8/8.1 and Windows Server 2008/2012, CUDA 8
conda install -c peterjc123 pytorch_legacy


如果不能忍受conda那蝸牛爬般的網速的話，那麼大家可以嘗試以下兩種途徑：

1.  添加清華源，然後使用conda進行安裝。

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/peterjc123/

# for CPU only packages
conda install pytorch-cpu

# for Windows 10 and Windows Server 2016, CUDA 8
conda install pytorch

# for Windows 10 and Windows Server 2016, CUDA 9
conda install pytorch cuda90

# for Windows 7/8/8.1 and Windows Server 2008/2012, CUDA 8
conda install pytorch_legacy
2. 從百度雲進行下載，大家下載之後，鍵入如下幾條指令：

cd "下載包的路徑"
conda install numpy mkl cffi
conda install --offline pytorch????.tar.bz2
注：檔案名說明：

一般為以下兩種形式

PACKAGENAME-VERSION-PYTHON_VERSIONcuCUDA_VERSION.tar.bz

或

PACKAGENAME-VERSION-PYTHON_VERSION_cudaCUDA_VERSION_cudnnCUDNN_VERSIONHASH_REVISION.tar.bz2

PACKAGENAME 分為 pytorch 和 pytorch_legacy， 分別為NT內核版本10和6的兩類系統進行編譯；VERSION 代表 pytorch 的版本；而PYTHON則代表python程式的版本，主要分為3.5和3.6；CUDA_VERSION和CUDNN_VERSION分別代表CUDA和cuDNN編譯的版本；REVISION代表修訂號。請自行選擇合適的版本進行安裝。

安裝之後，也千萬要注意，要在主代碼的最外層包上

if __name__ == '__main__':
這個判斷，可以參照我昨天文章中的例子，因為PyTorch的多執行緒庫在Windows下工作還不正常。

```

echo %PATH%

加入到PATH==> PATH =  D:\Anaconda3\Scripts;%PATH%

echo %PATH%

D:\Anaconda3\pkgs底下有許多安裝的套件

執行的程式放在 D:\Anaconda3\Scripts   如conda

把你測試用的程式碼放在D:\data2018\pytorch
在D:\data2018目錄底下執行jupyter notebook


Please note that PyTorch uses shared memory to share data between processes, so if torch multiprocessing is used (e.g.
for multithreaded data loaders) the default shared memory segment size that container runs with is not enough, and you
should increase shared memory size either with `--ipc=host` or `--shm-size` command line options to `nvidia-docker run`.

## 課本教材環境配置

1. 安裝[PyTorch](http://pytorch.org)，請從官網選擇指定的版本安裝即可，一鍵安裝（即使你使用anaconda，也建議使用pip）。更多的安裝方式請參閱書中說明。

2. 複製github上的檔案

   ```python
   git clone https://github.com/chenyuntc/PyTorch-book.git
   ```

3. 安裝協力廠商依賴包

   ```python
   cd pytorch-book && pip install -r requirements.txt
   ```

### pytorch官方文件
```
http://pytorch.org/

https://github.com/pytorch/pytorch

http://pytorch.org/docs/stable/index.html

```
## ipython

快捷鍵

魔術方法: %
%timeit
%hist
%paste
%cat
%run
%env
%magic

%debug ===>ipdb debugger


## PyTorch深論

PyTorch是一個函式庫由底下元件(components)所組成:

<table>
<tr>
    <td><b> torch </b></td>
    <td> a Tensor library like NumPy, with strong GPU support </td>
</tr>
<tr>
    <td><b> torch.autograd </b></td>
    <td> a tape-based automatic differentiation library that supports all differentiable Tensor operations in torch </td>
</tr>
<tr>
    <td><b> torch.nn </b></td>
    <td> a neural networks library deeply integrated with autograd designed for maximum flexibility </td>
</tr>
<tr>
    <td><b> torch.multiprocessing  </b></td>
    <td> Python multiprocessing, but with magical memory sharing of torch Tensors across processes. Useful for data loading and Hogwild training. </td>
</tr>
<tr>
    <td><b> torch.utils </b></td>
    <td> DataLoader, Trainer and other utility functions for convenience </td>
</tr>
<tr>
    <td><b> torch.legacy(.nn/.optim) </b></td>
    <td> legacy code that has been ported over from torch for backward compatibility reasons </td>
</tr>
</table>

PyTorch的使用情境:

- a replacement for NumPy to use the power of GPUs.
- a deep learning research platform that provides maximum flexibility and speed


[Welcome to PyTorch Tutorials](http://pytorch.org/tutorials/index.html)

[官方範例](https://github.com/pytorch/examples)

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


2.1.2學習環境配置

- 使用iPython
- 使用jupiter notebook

[2.2 PyTorch入門第一步](https://github.com/MyDearGreatTeacher/AI_and_security/blob/master/20180422/chapter2:%20PyTorch%E5%BF%AB%E9%80%9F%E5%85%A5%E9%96%80.ipynb)


2.2.1 Tensor  2.2.2 Autograd ：自動微分 2.2.3神經網絡 2.2.4小試牛刀：CIFAR-10分類

## 3 Tensor和autograd 

[3.1 Tensor](https://github.com/MyDearGreatTeacher/AI_and_security/blob/master/20180422/3_1_Tensor.ipynb)
```
3.1.1基礎操作 3.1.2 Tensor和Numpy  3.1.3內部結構 3.1.4其他有關Tensor的話題 3.1.5小試牛刀：線性回歸
```
[3.2 autograd](https://github.com/MyDearGreatTeacher/AI_and_security/blob/master/20180422/3_2_autograd.ipynb)
```
3.2.1 Variable  3.2.2計算圖 3.2.3擴展autograd  3.2.4小試牛刀：用Variable實現線性回歸
```
## 4神經網絡工具箱nn 
```
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
```
## 5 PyTorch中常用的工具
```
5.1數據處理
5.2計算機視覺工具包：torchvision 
5.3可視化工具
5.3.1 Tensorboard 
5.3.2 visdom 
5.4使用GPU加速：cuda 
5.5持久化
```
**實戰部分**（第六到十章）利用PyTorch實現了幾個酷炫有趣的應用，對於這部分的內容，本倉庫給出完整的實現代碼，並提供預訓練好的模型作為demo，供讀者測試。

- 第六章是承上啟下的一章，這一章的目標不是教會讀者新函數，新知識，而是結合Kaggle中一個經典的比賽，實現一個深度學習中比較簡單的圖像二分類問題。在實現過程中，帶領讀者複習前五章的知識，並提出代碼規範以合理的組織程式，代碼，使得程式更加可讀，可維護。第六章還介紹了在PyTorch中如何進行debug。
- 第七章為讀者講解了當前最火爆的生成對抗網路（GAN），帶領讀者從頭實現一個動漫頭像生成器，能夠利用GAN生成風格多變的動漫頭像。
- 第八章為讀者講解了風格遷移的相關知識，並帶領讀者實現風格遷移網路，將自己的照片變成高大上的名畫。
- 第九章為讀者講解了一些自然語言處理的基礎知識，並講解了CharRNN的原理。而後利用收集了幾萬首唐詩，訓練出了一個可以自動寫詩歌的小程式。這個小程式可以控制生成詩歌的**格式**，**意境**，還能生成**藏頭詩**。
- 第十章為讀者介紹了圖像描述任務，並以最新的AI Challenger比賽的資料為例，帶領讀者實現了一個可以進行簡單圖像描述的的小程式。
- 第十一章（**新增，實驗性**） 由[Diamondfan](https://github.com/Diamondfan) 編寫的語音辨識。完善了本專案（本專案已囊括圖像，文本，語音三大領域的例子）。
```
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
