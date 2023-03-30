import os
import numpy as np
import struct
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

# %% Load MNIST dataset and prepare the train/validation/test set.
def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' %kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' %kind)

    with open(labels_path, 'rb') as lbpath:
        # > : Big endian
        # I : unsigned integer
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

train_x, train_y = load_mnist('/Users/mingyucheon/Desktop/dataset', kind='train')
test_x, test_y = load_mnist('/Users/mingyucheon/Desktop/dataset', kind='t10k')

train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.1, shuffle=False, random_state=1)

train_x = train_x / 255
test_x = test_x / 255
valid_x = valid_x / 255

train_y = to_categorical(train_y)
valid_y = to_categorical(valid_y)
test_y = to_categorical(test_y)

# %% Make MLP model
model = Sequential([
    keras.Input(shape=(784)), 
    Dense(1024, activation = 'relu', use_bias=True), 
    Dense(1024, activation = 'relu', use_bias=True), 
    Dense(1024, activation = 'relu', use_bias=True), 
    Dense(1024, activation = 'relu', use_bias=True), 
    Dense(10, activation='softmax', use_bias=True)
])

model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

start_time = time.time()

hist = model.fit(train_x, train_y, epochs = 10, batch_size = 100, validation_data=(valid_x, valid_y))

consumed_time = time.time() - start_time

# %% Evaluate the model
results = model.evaluate(test_x, test_y)
print('Test loss, Test Accuracy', results)
print('Consumed time : ', consumed_time)

loss_ax = plt.subplot()
acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()
