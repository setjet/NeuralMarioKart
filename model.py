import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D 
from keras.layers.core import Activation, Dense, Flatten
from keras.optimizers import Adam
from PIL import Image
import time
import config as cfg
from util import viz

from keras import backend as K
from scipy.misc import imsave

def get_data():
  y = np.genfromtxt('data/y.npy', delimiter=',')
  X = np.genfromtxt('data/X.npy', delimiter=',').reshape(len(y), cfg.INPUT_HEIGTH, cfg.INPUT_WIDTH, cfg.COLOR_DIM).astype('float32') / 255
  return X, y

def create_model(X, y):
  model = Sequential([
    Convolution2D(32, 5, 5, input_shape=(cfg.INPUT_HEIGTH, cfg.INPUT_WIDTH, cfg.COLOR_DIM), name='conv1'),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Convolution2D(64, 3, 3, name='conv2'),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Convolution2D(128, 3, 3, name='conv3'),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
      
    Flatten(),
    Dense(128),
    Activation('tanh'),
    Dense(2),
    Activation('tanh'),
  ])

  model.compile(loss='mse', optimizer='adam')
  history = model.fit(X, y, nb_epoch=1, batch_size=32, verbose=1)
  model.save(cfg.MODEL)
  return model

X, y = get_data()
model = create_model(X, y)


viz(model, X, 'conv1')
viz(model, X, 'conv1', noise=True)



