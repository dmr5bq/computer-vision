from __future__ import print_function

from sklearn.metrics import confusion_matrix
import numpy as np

'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop


batch_size = 128
num_classes = 10
epochs = 20
num_trials = 60000


# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(num_trials, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
# model.add(Dropout(0.2))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss = 'categorical_crossentropy',
              optimizer = RMSprop(),
              metrics = ['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

y_calc = model.predict(x_test)


def get_mnist_confusion_matrix(y_test, y_calc):
    digs_truth = []

    for i in range(y_test.shape[0]):
        max = 0
        max_ind = 0
        for j in range(y_test.shape[1]):

            if y_test[i, j] > max:
                max = y_test[i, j]
                max_ind = j

        digs_truth.append(max_ind)


    digs_calc = []

    for i in range(y_calc.shape[0]):

        max = 0
        max_ind = 0

        for j in range(y_calc.shape[1]):

            if y_calc[i, j] > max:
                max = y_calc[i, j]
                max_ind = j

        digs_calc.append(max_ind)

    return confusion_matrix(digs_truth, digs_calc)


def row_normalize_confusion_matrix(c_mat):

    output = np.zeros((c_mat.shape), dtype=np.float32)
    for i in range(c_mat.shape[0]):

        sum = 0

        for j in range(c_mat.shape[1]):
            sum += c_mat[i, j]

        output[i] = np.multiply(c_mat[i].astype("float32"), 1./float(sum))

    return output


print(get_mnist_confusion_matrix(y_test, y_calc))
print(row_normalize_confusion_matrix(get_mnist_confusion_matrix(y_test, y_calc)))