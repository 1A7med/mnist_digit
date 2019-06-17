from keras.models import load_model

import zkeract
import utils
from data import get_mnist_data, num_classes, input_shape
import numpy as np
import math
import matplotlib.pyplot as plt
import keract

x_train, _, _, _ = get_mnist_data()

a_model = load_model('s2/model_18_0.988.h5')
a = zkeract.get_activations(a_model, x_train[2:3])  # with just one sample.

al =  list( a.values() )

a_layer = al[0]
b_layer = al[1]

t = 0
hold = 0.01
c_layer = np.zeros( (1, 28, 28, 16) )

for i in range( 28 ) :
	for j in range( 28 ):
		for k in range( 16 ) :
			d = a_layer[0][i][j][k] - b_layer[0][i][j][k]
			if( d < hold and d != 0 ) :
				t = t +1
				c_layer[0][i][j][k] = ( a_layer[0][i][j][k] + b_layer[0][i][j][k] ) / 2


keract.display_activations( a )

for layer_name, acts in a.items():
    print(layer_name, acts.shape, end=' ')

    if acts.shape[0] != 1:
            print('-> Skipped. First dimension is not 1.')
            continue
    if len(acts.shape) <= 2:
            print('-> Skipped. 2D Activations.')
            continue

    nrows = int(math.sqrt(acts.shape[-1]) - 0.001) + 1  # best square fit for the given number
    ncols = int(math.ceil(acts.shape[-1] / nrows))
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, figsize=(12, 12))
    fig.suptitle(layer_name)

    for i in range(nrows * ncols):
        if i < acts.shape[-1] :
                #img = acts[0, :, :, i]
                img = c_layer[0, :, :, i]
                hmap = axes.flat[i].imshow(img, cmap='gray')
        axes.flat[i].axis('off')
    fig.subplots_adjust(right=0.8)
    cbar = fig.add_axes([0.85, 0.15, 0.03, 0.7])
    fig.colorbar(hmap, cax=cbar)

    plt.show()
