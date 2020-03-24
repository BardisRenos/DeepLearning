from __future__ import print_function
from keras.constraints import maxnorm
from keras.layers import Convolution2D, Dropout, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.datasets import cifar10


def training_data():
    # Loading the dataset from
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.0
    X_test /= 255.0

    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    return X_train, X_test, y_train, y_test


def train_model():
    X_train, X_test, y_train, y_test = training_data()
    # Create the model
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(32, 32, 3), activation='relu', border_mode='same'))
    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(y_test.shape[1], activation='softmax'))
    # Compile model
    epochs = 25
    lrate = 0.01
    decay = lrate / epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=epochs, batch_size=64)

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))


if __name__ == '__main__':
    train_model()
