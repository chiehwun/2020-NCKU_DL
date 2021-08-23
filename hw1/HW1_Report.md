# 2020-NCKU_DL_HW1
---
## 基本資料

| 學號      | 姓名   | 系級     |
| --------- | ------ | -------- |
| E14066282 | 溫梓傑 | 機械 110 |
---

## Environment

**Platform:** Windows-10-10.0.18362-SP0
**Python:** 3.8.5
**Tensorflow:** 2.2.0
**Keras:** 2.4.3

``` console
$ pip install tensorflow==2.2.0
$ pip install PyQt5
$ pip install pyqt5-tools
$ pip install PyQt5-stubs
$ pip pyclean
```
---
## Before Running `hw1_5\main.py`...
* Download [Trained Model](https://drive.google.com/file/d/162IgGehmniBrRxIirVgDbeYP0MCRoPce/view?usp=sharing) to `\2020-NCKU_DL\hw1_5\vgg16_cifar10_11-12_06_42.h5`

## Run

``` console
$ python main.py
```

## Problem 5.2

Print out training hyperparameters (batch size, learning rate, optimizer).

``` console
hyperparameters:
batch size: 32      
learning rate: 0.001
optimizer: Adam 
```
---

## Problem 5.3
Construct and show your model structure.
``` console
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 32, 32, 64)        1792      
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 32, 32, 64)        36928
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 16, 16, 64)        0
_________________________________________________________________
batch_normalization (BatchNo (None, 16, 16, 64)        256
_________________________________________________________________
dropout (Dropout)            (None, 16, 16, 64)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 16, 16, 128)       73856
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 16, 16, 128)       147584
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 8, 8, 128)         0
_________________________________________________________________
batch_normalization_1 (Batch (None, 8, 8, 128)         512
_________________________________________________________________
dropout_1 (Dropout)          (None, 8, 8, 128)         0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 8, 8, 256)         295168
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 8, 8, 256)         590080
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 8, 8, 256)         590080
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 4, 4, 256)         0
_________________________________________________________________
batch_normalization_2 (Batch (None, 4, 4, 256)         1024      
_________________________________________________________________
dropout_2 (Dropout)          (None, 4, 4, 256)         0
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 4, 4, 512)         1180160   
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 4, 4, 512)         2359808
_________________________________________________________________
conv2d_9 (Conv2D)            (None, 4, 4, 512)         2359808
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 2, 2, 512)         0
_________________________________________________________________
batch_normalization_3 (Batch (None, 2, 2, 512)         2048
_________________________________________________________________
dropout_3 (Dropout)          (None, 2, 2, 512)         0
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 2, 2, 512)         2359808
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 2, 2, 512)         2359808
_________________________________________________________________
conv2d_12 (Conv2D)           (None, 2, 2, 512)         2359808
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 1, 1, 512)         0
_________________________________________________________________
flatten (Flatten)            (None, 512)               0
_________________________________________________________________
dense (Dense)                (None, 4096)              2101248
_________________________________________________________________
dense_1 (Dense)              (None, 4096)              16781312
_________________________________________________________________
dense_2 (Dense)              (None, 10)                40970
=================================================================
Total params: 33,642,058
Trainable params: 33,640,138
Non-trainable params: 1,920
_________________________________________________________________
None
```

---

## Problem 5.4

take a screenshot of your training loss and accuracy

![title](11-12_06_42.png)

---

## Outline
1. (20%) Image Processing
    1.1 (5%) Load Image File
    1.2 (5%) Color Separation  
    1.3 (5%) Image Flipping
    1.4 (5%) Blending
2. (20%) Image Smoothing
    2.1 (7%) Median filter
    2.2 (7%) Gaussian blur
    2.3 (6%) Bilateral filter
3. (20%) Edge Detection
    3.1 (5%) Gaussian Blur
    3.2 (5%) Sobel X
    3.3 (5%) Sobel Y
    3.4 (5%) Magnitude
4. (20%) Transforms:
    4.1 (7%) Rotation
    4.2 (7%) Scaling
    4.3 (6%) Translation
5. (20%) Training Cifar-10 Classifier Using VGG16
---
## Contact

### :email: e14066282@gs.ncku.edu.tw