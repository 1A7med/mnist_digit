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

s = 1
l ='kt'

kn = 2 * s + 1
nn = 2 ** ( s +3 )
dn = 30

l += str(s)
m = './' + l
f = './tb_log/mnist/'
f += l

bs = 64
epk = 15
tb = TensorBoard(log_dir=f, histogram_freq=0, batch_size=bs, write_graph=True, embeddings_freq=0, update_freq='epoch')


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

        x_train, y_train, x_test, y_test = get_mnist_data()

        
        activ = zkeract.get_activations(model, x_test[0:200]) 
        grad_tw = zkeract.get_gradients_of_trainable_weights(model, x_train[0:10], y_train[0:10])
        grad_activ = zkeract.get_gradients_of_activations(model, x_train[0:10], y_train[0:10])
		
        zkeract.display_gradients_of_trainable_weights(grad_tw, save=False)
        zkeract.display_gradients_of_trainable_weights(grad_activ, save=False)

        a = zkeract.get_activations(model, x_test[9:10])  
        zkeract.display_activations(a, m)

        
    else:
        x_train, y_train, x_test, y_test = get_mnist_data()

        model = Sequential()
        model.add(Conv2D(nn, (kn, kn), activation='relu', input_shape=input_shape, padding='same'))
        model.add(Conv2D(nn, (kn, kn), activation='relu', padding='same'))
        model.add(Conv2D(nn, (kn, kn), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        if( s > 3 ) :
        	model.add(Conv2D(nn, (kn, kn), activation='relu', padding='same'))
        	model.add(Conv2D(nn, (kn, kn), activation='relu', padding='same'))
        	model.add(MaxPooling2D(pool_size=(2, 2)))
        	model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(dn, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

        import shutil

        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
        os.makedirs(checkpoint_dir)

        checkpoint = ModelCheckpoint(monitor='val_acc', save_best_only=True, filepath=os.path.join(checkpoint_dir, 'model_{epoch:02d}_{val_acc:.3f}.h5'))

        model.fit(x_train, y_train, batch_size=bs, epochs=3, verbose=1, validation_data=(x_test, y_test), callbacks=[checkpoint, tb] , shuffle=False)

        score = model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
