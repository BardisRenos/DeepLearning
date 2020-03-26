# Deep Learning  example 
This repo has two example of deep neural network. In the first model uses ready to use data set [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html). The second model uses a data set that is not splitted and we build the data set to feed our system.

## Installation
First we need to install the libraries. You can use the package manager [pip](https://pip.pypa.io/en/stable/) or [conda](https://docs.conda.io/en/latest/) to install the Deep Learning libraries. I use Keras and Tensorflow as back end. 

```bach 
pip install keras
pip install tensorflow
```

## Description of the model

The first part of the code, I import all the necessary elements in order to build the model. 

```python
from __future__ import print_function
from keras.constraints import maxnorm
from keras.layers import Convolution2D, Dropout, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.datasets import cifar10
```

## Data set
The data set is splitted into Train and Test set. The percentage is 90/10. Hence, the train data is 90% of the total dataset and the test is 10%. 
```python
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
```
## The input part

The main body of deep neural network. I create a sequential model as the below picture depicts. A neural network which has more than one hidden layer is called deep neural network. The sequential model is constracted by input layer, hidden layers and the output layer.
The first layer is a Convolutional layer which is also an input layer layer with 32 neurons with 3*3 filter and input image shape 32 by 32 pixles and by 3 in depth (RGB) if it is collored image and by 1 if it is grayscale.   

<p align="center"> 
<img src="https://github.com/BardisRenos/DeepLearning/blob/master/Sequential.jpeg" width="450" height="250" style=centerme>
</p>

```python
 model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(32, 32, 3), activation='relu', border_mode='same'))
    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))
    model.add(Dropout(0.5))
```

## Input image
The scructure of the input image. An example of an input images is shown below. 
```python
   model.add(Convolution2D(32, 3, 3, input_shape=(32, 32, 3), activation='relu', border_mode='same'))
```

The structure of an input image. 
<img src="https://github.com/BardisRenos/DeepLearning/blob/master/cnn_diagram_notation.png" width="350" height="250">

## Convolutional layer
A convolutional neural network has a convolutional layer which is frame size that works as a filter with arbitrary values. The output of this layer is a new array with new values. 

```python
    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
```
<p align="center"> 
<img src="https://github.com/BardisRenos/DeepLearning/blob/master/ConvLayer.png" width="350" height="250">

 
## Avtivation function
Next layer is an activation function. In our model the ReLU is used. Below the first image shows the mathimatical function. Whereas, the second picture shows how the relu works after the convolution layer. 

```python
   activation='relu'
```

<p align="center"> 
<img src="https://github.com/BardisRenos/DeepLearning/blob/master/Relu.png" width="450" height="250">


The ReLU layer create a new image layer with values that are only positive and all the negative one replace them with 0 

<p align="center"> 
<img src="https://github.com/BardisRenos/DeepLearning/blob/master/ReluNumber.jpg" width="450" height="250">


## MaxPooling layer
After the activation function, maxpoooling layer follows. This layer reduce the size of the image. That way the output image keeps only the most important features. 

```python
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))
```

<p align="center"> 
<img src="https://github.com/BardisRenos/DeepLearning/blob/master/max_pooling.png" width="450" height="250">


## Dropout layer
The dropout layer works in order the model not to avoid overfitting of the model.

```python
    model.add(Dropout(0.5))
```

<p align="center"> 
<img src="https://github.com/BardisRenos/DeepLearning/blob/master/overfitting.png" width="350" height="200">





