import numpy as np
import os
import cv2
import random
import pickle
from keras.layers import Convolution2D, Dropout, MaxPooling2D, Flatten, Dense, Activation
from keras.models import Sequential
from keras.optimizers import SGD

# Here we set the path of the dataset.
DATALOCATION = 'C:\\path\\of\\folder'

# The categories of the data set. Are two only Cats and dogs. Hence, Dog is category 0 and the Cat is 1
CATEGORIES = ["Dog", "Cat"]
# The siza that we want to reforamt all the images in the same dimentions
IMG_SIZE = 250


class DNN:
    @staticmethod
    def create_training_data() -> list:
        training_data = []
        for categories in CATEGORIES:
            path = os.path.join(DATALOCATION, categories)
            class_num = CATEGORIES.index(categories)
            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    training_data.append([img_array, class_num])
                except Exception as e:
                    pass

        return training_data

    @staticmethod
    def creating_features_labels():
        # Data and labels
        X = []
        y = []
        data = DNN.create_training_data()
        # We shuffle the data to be random
        random.shuffle(data)
        for features, label in data:
            X.append(features)
            y.append(label)
        # We reshape the data in order to fit the DNN model
        X = np.asarray(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        return X, y

    @staticmethod
    def save_the_data():
        X, y = DNN.creating_features_labels()
        # save_the_data()
        pickle_out = open("X.pickle", "wb")
        pickle.dump(X, pickle_out)
        pickle_out.close()

        pickle_out = open("y.pickle", "wb")
        pickle.dump(y, pickle_out)
        pickle_out.close()

    @staticmethod
    def Artificial_Neural_Network():
        # Restore the data
        DNN.save_the_data()
        X = pickle.load(open("X.pickle", "rb"))
        y = pickle.load(open("y.pickle", "rb"))
        X = X / 255.0

        # Create the DNN model
        model = Sequential()
        model.add(Convolution2D(32, 3, 3, input_shape=X.shape[1:], activation='relu', border_mode='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(64, activation='relu'))

        model.add(Dense(1, activation='softmax'))

        # Compile model
        epochs = 2
        lrate = 0.01
        sgd = SGD(lr=lrate, momentum=0.9, decay=lrate/epochs, nesterov=False)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
        print(model.summary())
        model.fit(X, y, validation_split=0.1, nb_epoch=epochs, batch_size=64)

        # Final evaluation of the model
        scores = model.evaluate(X, y)
        print("Accuracy: %.2f%%" % (scores[1] * 100))


if __name__ == '__main__':
    DNN.Artificial_Neural_Network()
