import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D 
from keras.layers.core import Activation, Dense, Flatten
from keras.optimizers import Adam
from PIL import Image
import time
import config as cfg

from keras import backend as K
from scipy.misc import imsave

size = len(np.fromfile('data/y.npy', dtype=float, count=-1, sep=' '))


print "loading"
y = np.fromfile('data/y.npy', dtype=float, count=-1, sep=' ').reshape((size/2, 2))
print "loaded y"
X = np.fromfile('data/X.npy', dtype=float, count=-1, sep=' ').reshape((size/2, cfg.COLOR_DIM*cfg.INPUT_WIDTH*cfg.INPUT_HEIGTH))


X_train = X.reshape(X.shape[0], cfg.INPUT_HEIGTH, cfg.INPUT_WIDTH, cfg.COLOR_DIM).astype('float32') / 255


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
    # jees this layer costs much
    Dense(256),
    Activation('tanh'),
    Dense(2),
    Activation('tanh'),
])


adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mse', optimizer=adam)

history = model.fit(X_train, y, nb_epoch=10, batch_size=50, verbose=1)



model.save(cfg.MODEL)

### Vizzing

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x



# this is the placeholder for the input images
input_img = model.input

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers[0:]])


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def viz(layer_name, noise=False):
  print "vizzing layer"
  kept_filters = []
  for filter_index in range(0, 24):
      # we only scan through the first 32 filters,
      # but there are actually varying amount of them

      # we build a loss function that maximizes the activation
      # of the nth filter of the layer considered
      layer_output = layer_dict[layer_name].output
      if K.image_dim_ordering() == 'th':
          loss = K.mean(layer_output[:, filter_index, :, :])
      else:
          loss = K.mean(layer_output[:, :, :, filter_index])

      # we compute the gradient of the input picture wrt this loss
      grads = K.gradients(loss, input_img)[0]

      # normalization trick: we normalize the gradient
      grads = normalize(grads)

      # this function returns the loss and grads given the input picture
      iterate = K.function([input_img], [loss, grads])

      # step size for gradient ascent
      step = 1.

      # we start from a gray image with some random noise
      if K.image_dim_ordering() == 'th':
          input_img_data = np.random.random((1, cfg.INPUT_WIDTH, cfg.INPUT_HEIGTH, 1))
      else:
          if noise:
              input_img_data = np.zeros((1, cfg.INPUT_WIDTH, cfg.INPUT_HEIGTH, 1))
          else:
              input_img_data = np.array([X_train[0]]) 
      input_img_data = (input_img_data - 0.5) * 20 + 128

      # we run gradient ascent for 20 steps
      for i in range(20):
          loss_value, grads_value = iterate([input_img_data])
          input_img_data += grads_value * step

          #print('Current loss value:', loss_value)
          if loss_value <= 0.:
              # some filters get stuck to 0, we can skip them
              break

      # decode the resulting input image
      #if loss_value > 0: why should loss value be smaller than 0
      img = deprocess_image(input_img_data[0])
      kept_filters.append((img, loss_value))

  # we will stich the best 16 filters on a 4 x 4 grid.
  n = 4

  # the filters that have the highest loss are assumed to be better-looking.
  # we will only keep the top 16 filters.
  kept_filters.sort(key=lambda x: x[1], reverse=True)
  kept_filters = kept_filters[:n * n]

  # build a black picture with enough space for
  # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
  margin = 5
  width = n * cfg.INPUT_WIDTH + (n - 1) * margin
  height = n * cfg.INPUT_HEIGTH + (n - 1) * margin
  stitched_filters = np.zeros((width, height, 3))

  # fill the picture with our saved filters
  for i in range(n):
      for j in range(n):
          img, loss = kept_filters[i * n + j]
          stitched_filters[(cfg.INPUT_WIDTH + margin) * i: (cfg.INPUT_WIDTH + margin) * i + cfg.INPUT_WIDTH,
                           (cfg.INPUT_HEIGTH + margin) * j: (cfg.INPUT_HEIGTH + margin) * j + cfg.INPUT_HEIGTH, :] = img

  # save the result to disk 
  #imsave(layer_name, stitched_filters)
  imsave(layer_name+str(noise)+'.png', stitched_filters)

#viz('conv1')
#viz('conv2')
#viz('conv3')
#viz('conv4')
#viz('conv5')
#viz('conv1', noise=True)
#viz('conv2', noise=True)
#viz('conv3', noise=True)
#viz('conv4', noise=True)
#viz('conv5', noise=True)


