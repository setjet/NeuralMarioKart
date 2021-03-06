import numpy as np
import setup.config as cfg
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D 
from keras.layers.core import Activation, Dense, Flatten, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt

def _get_data():
  size = len(np.fromfile('data/y.npy', dtype=np.float32, count=-1, sep=' '))
  y = np.fromfile('data/y.npy', count=-1, dtype=np.float32, sep=' ').reshape((size/2, 2))
  X = np.fromfile('data/X.npy', count=-1, dtype=np.float32, sep=' ').reshape(size/2, cfg.INPUT_HEIGTH, cfg.INPUT_WIDTH, cfg.COLOR_DIM) / 255
  return X, y

def _visualize_loss(history):
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  fig = plt.gcf()
  fig.savefig('model/loss.png')
  plt.show()

def create_model():
  X, y = _get_data()

  model = Sequential([
    Convolution2D(32, 5, 5, input_shape=(cfg.INPUT_HEIGTH, cfg.INPUT_WIDTH, cfg.COLOR_DIM), name='conv1'),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Convolution2D(64, 3, 3, name='conv2'),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Convolution2D(128, 3, 3, name='conv3'),
    Activation('relu'),
    Dropout(0.5),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(64),
    Activation('tanh'),
    Dense(2),
    Activation('tanh'),
  ])

  model.compile(loss='mse', optimizer='adam')
  history = model.fit(X, y, validation_split=0.33, nb_epoch=1, batch_size=32, verbose=1)
  model.save(cfg.MODEL)
  _visualize_loss(history)

create_model()
