import numpy as np
import config as cfg
from keras import backend as K
from scipy.misc import imsave

def _normalize(x):
  # utility function to normalize a tensor by its L2 norm
  return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

# decode the resulting input image
#if loss_value > 0: why should loss value be smaller than 0
def _deprocess_image(x):
  # normalize tensor: center on 0., ensure std is 0.1
  x -= x.mean()
  x /= (x.std() + 1e-5)
  x *= 0.1

  # clip to [0, 1]
  x += 0.5
  x = np.clip(x, 0, 1)

  # convert to RGB array
  x *= 255
  x = np.clip(x, 0, 255).astype('uint8')
  return x

def _get_filters(filter_index, model, layer_name, X):
  layer_dict = dict([(layer.name, layer) for layer in model.layers[0:]])
  # we only scan through the first 32 filters,
  # but there are actually varying amount of them

  # we build a loss function that maximizes the activation
  # of the nth filter of the layer considered
  layer_output = layer_dict[layer_name].output
  loss = K.mean(layer_output[:, :, :, filter_index])

  # we compute the gradient of the input picture wrt this loss
  grads = K.gradients(loss, model.input)[0]

  # normalization trick: we normalize the gradient
  grads = _normalize(grads)

  # this function returns the loss and grads given the input picture
  iterate = K.function([model.input], [loss, grads])

  # we start from a gray image with some random noise
  input_img_data = np.zeros((1, cfg.INPUT_HEIGTH, cfg.INPUT_WIDTH, cfg.COLOR_DIM))
  input_img_data = (input_img_data - 0.5) * 20 + 128

  # we run gradient ascent for 20 steps
  for i in range(20):
    loss_value, grads_value = iterate([input_img_data])
    input_img_data += grads_value

  img = _deprocess_image(input_img_data[0])
  return (img, loss_value)

def visualize(model, X, layer_name):
  print "Visualizing layer " + str(layer_name)

  filters = []
  for filter_index in range(0, 24):
    filters.append(_get_filters(filter_index, model, layer_name, X))

  # we will stich the best 16 filters on a 4 x 4 grid.
  n = 4

  # the filters that have the highest loss are assumed to be better-looking.
  # we will only keep the top 16 filters.
  filters.sort(key = lambda x: x[1], reverse=True)
  filters = filters[:n * n]

  # build a black picture with enough space for
  # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
  margin = 5
  width = n * cfg.INPUT_WIDTH + (n - 1) * margin
  height = n * cfg.INPUT_HEIGTH + (n - 1) * margin
  stitched_filters = np.zeros((height, width, cfg.COLOR_DIM))

  # fill the picture with our saved filters
  for i in range(n):
    for j in range(n):
      img, loss = filters[i * n + j]
      stitched_filters[(cfg.INPUT_HEIGTH + margin) * i: (cfg.INPUT_HEIGTH + margin) * i + cfg.INPUT_HEIGTH,
                       (cfg.INPUT_WIDTH + margin) * j: (cfg.INPUT_WIDTH + margin) * j + cfg.INPUT_WIDTH, :] = img

  # save the result to disk 
  imsave(layer_name+'.png', stitched_filters)
