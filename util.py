import random
import pygame
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from keras.models import load_model
import gtk.gdk
import Image 
import numpy as np
import time
import config as cfg
from keras import backend as K
from scipy.misc import imsave

def viz(model, X, layer_name, noise=False):
  print "vizzing layer"
  input_img = model.input
  kept_filters = []
  layer_dict = dict([(layer.name, layer) for layer in model.layers[0:]])
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

      def normalize(x):
        # utility function to normalize a tensor by its L2 norm
        return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

      # normalization trick: we normalize the gradient
      grads = normalize(grads)

      # this function returns the loss and grads given the input picture
      iterate = K.function([input_img], [loss, grads])

      # step size for gradient ascent
      step = 1.

      # we start from a gray image with some random noise
      if K.image_dim_ordering() == 'th':
          input_img_data = np.random.random((1, cfg.INPUT_HEIGTH, cfg.INPUT_WIDTH, cfg.COLOR_DIM))
      else:
          if noise:
              input_img_data = np.zeros((1, cfg.INPUT_HEIGTH, cfg.INPUT_WIDTH, cfg.COLOR_DIM))
          else:
              input_img_data = np.array([X[0]]) 
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
  stitched_filters = np.zeros((height, width, cfg.COLOR_DIM))

  # fill the picture with our saved filters
  for i in range(n):
      for j in range(n):
          img, loss = kept_filters[i * n + j]
          stitched_filters[(cfg.INPUT_HEIGTH + margin) * i: (cfg.INPUT_HEIGTH + margin) * i + cfg.INPUT_HEIGTH,
                           (cfg.INPUT_WIDTH + margin) * j: (cfg.INPUT_WIDTH + margin) * j + cfg.INPUT_WIDTH, :] = img

  # save the result to disk 
  #imsave(layer_name, stitched_filters)
  imsave(layer_name+str(noise)+'.png', stitched_filters)

def _get_pixel_buffer():
  w = gtk.gdk.get_default_root_window()
  pb = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, False, 8, cfg.CAPTURE_WIDTH, cfg.CAPTURE_HEIGTH)
  return pb, w

def _capture_image(pb, w):
  pixel_buffer = pb.get_from_drawable(w, w.get_colormap(), cfg.CAPTURE_X, cfg.CAPTURE_Y, 0, 0, cfg.CAPTURE_WIDTH, cfg.CAPTURE_HEIGTH)
  image = Image.frombytes("RGB", (cfg.CAPTURE_WIDTH, cfg.CAPTURE_HEIGTH), pixel_buffer.get_pixels())
  if (cfg.GREYSCALE):
    image = image.convert('L')
  image.thumbnail([cfg.INPUT_WIDTH, cfg.INPUT_HEIGTH], Image.ANTIALIAS)
  return image


class xboxController(object):
  def __init__(self):
    try:
      pygame.init()
      self.joystick = pygame.joystick.Joystick(0)
      self.joystick.init()
    except:
      print 'Unable to connect to Xbox Controller'

  def read(self):
    pygame.event.pump()
    x = int(self.joystick.get_axis(0) * cfg.JOYSTICK_NORMALIZER)
    y = int(self.joystick.get_axis(1) * cfg.JOYSTICK_NORMALIZER)
    a = int(round(self.joystick.get_button(0)))
    b = int(round(self.joystick.get_button(2)))
    rb = int(round(self.joystick.get_button(5)))
    return [x, y, a, b, rb]

  def manual_override(self):
    pygame.event.pump()
    return self.joystick.get_button(4) == 1


class neuralNetwork(object):
  def __init__(self): 
    print "neural network"
    self.model = load_model(cfg.MODEL)
    self.pb, self.w = _get_pixel_buffer()
    self.real_controller = xboxController()
    self.last = [0, 0, 0, 0, 0]
    self.flip = 0

    try:
      pygame.init()
      self.joystick = pygame.joystick.Joystick(0)
      self.joystick.init()
    except:
      print 'unable to connect to Xbox Controller'

  def read(self):
    pygame.event.pump()

    manual_override = self.real_controller.manual_override()
    if (manual_override):
      return self.real_controller.read()
    elif self.flip == 5: # too slow otherwise...
      image = _capture_image(self.pb, self.w)
      vector = np.asarray(image).reshape(1, cfg.INPUT_HEIGTH, cfg.INPUT_WIDTH, cfg.COLOR_DIM).astype('float32') / 255

      pred = self.model.predict(vector)[0]
      self.last = [pred[0]*cfg.JOYSTICK_NORMALIZER, 0, round(pred[1]), 0, 0] # only do x and a
      self.flip = 0
      return self.last
    else:
      self.flip = self.flip + 1
      return self.last


def run_server(input_device, port = 8082):
  class httpHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
      pass

    def do_GET(self):
      self.send_response(200)
      self.send_header("Content-type", "text/plain")
      self.end_headers()

      self.wfile.write(input_device.read())
      return

  try:
    server = HTTPServer(('', port), httpHandler)
    print 'Started httpserver on port ', port
    server.serve_forever()

  except KeyboardInterrupt:
    print '^C received, shutting down the web server'
    server.socket.close()
