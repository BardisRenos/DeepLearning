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
## The main body of deep neural network. I create a sequential model as the 

```python
 model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(32, 32, 3), activation='relu', border_mode='same'))
    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))
    model.add(Dropout(0.5))
```

