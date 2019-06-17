from __future__ import print_function

import os
from glob import glob

import keras
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.models import load_model

import zkeract
from data import get_mnist_data, num_classes, input_shape

# 3 x (conv, max) , flat, dense_s  ___no dropout___
s = 1
l ='r'

kn = 3
nn = 16

l += str(s)
m = './' + str( l )  + '/'
f = './tb_log/'
f += l

v = './' + str(l) + '/' + str(s) + '.hdf5'

bs = 64
epk = 15


if __name__ == '__main__':
    checkpoint_dir = l

    checkpoints = glob(os.path.join(checkpoint_dir, '*.h5'))

    if len(checkpoints) > 0:

        checkpoints = sorted(checkpoints)
        assert len(checkpoints) != 0, 'No checkpoints found.'
        checkpoint_file = checkpoints[-1]
        print('Loading [{}]'.format(checkpoint_file))
        model = load_model(checkpoint_file)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        print(model.summary())

