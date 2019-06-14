#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 09:50:55 2019

@author: ahmed
"""

from data import get_mnist_data, num_classes, input_shape

from keras.models import load_model, Sequential
from keras.layers import Input, Dense, Flatten, MaxPooling2D, Dropout
import numpy as np

a_model = load_model('ka1/model_15_0.991.h5')
x_train, y_train, x_test, y_test = get_mnist_data()


l0 = a_model.layers[0] #.get_weights() of first layer
l1 = a_model.layers[1]

# kernel weights
weights_0 = l0.get_weights()[0]
weights_1 = l1.get_weights()[0]

# bias weights
bias_0  = l0.get_weights()[1]
bias_1  = l1.get_weights()[1]

# weights averaging
t = 0
w_threshold = 0.05
weights_7 = np.zeros( (3, 3, 1, 16), dtype= 'f' )

for i in range( 3 ) :
	for j in range( 3 ):
		for k in range( 16 ) :
			w_diff = weights_0[i][j][0][k] - weights_1[i][j][0][k]
			if( w_diff < w_threshold and w_diff != 0 ) :
				t = t +1
				weights_7[i][j][0][k] = ( weights_0[i][j][0][k] + weights_1[i][j][0][k] ) / 2

# bias averaging
u = 0
b_threshold = 0.1
bias_7 = np.zeros(16)
				
for i in range( 16 ):
	b_diff = bias_0[i] - bias_1[i]
	if( b_diff < b_threshold and b_diff != 0 ) :
		u = u +1
		bias_7 = ( bias_0 + bias_1) / 2
		
ln = np.array( [ weights_7 , bias_7 ] )	
		
# reconstruct the layer and inject it back 

l0.set_weights( ln )


y_model = Sequential()
y_model.add( l0 )
y_model.add(MaxPooling2D(pool_size=(2, 2)))
y_model.add(Dropout(0.25))
y_model.add( Flatten())
y_model.add( Dense(10, activation='softmax') )

y_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
y_model.fit(x_train, y_train, batch_size=64, epochs=3, verbose=1, validation_data=(x_test, y_test) )


score = a_model.predict(x_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

y_model.save('7mdELLEH.h5')

print(t)
print( w_threshold )


