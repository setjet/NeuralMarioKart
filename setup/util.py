import pygame
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from keras.models import load_model
import gtk.gdk  
import Image 
import numpy as np
import setup.config as cfg


def get_pixel_buffer():
  w = gtk.gdk.get_default_root_window()
  pb = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, False, 8, cfg.CAPTURE_WIDTH, cfg.CAPTURE_HEIGTH)
  return pb, w

def capture_image(pb, w):
  pixel_buffer = pb.get_from_drawable(w, w.get_colormap(), cfg.CAPTURE_X, cfg.CAPTURE_Y, 0, 0, cfg.CAPTURE_WIDTH, cfg.CAPTURE_HEIGTH)
  image = Image.frombytes("RGB", (cfg.CAPTURE_WIDTH, cfg.CAPTURE_HEIGTH), pixel_buffer.get_pixels())
  if (cfg.GREYSCALE):
    image = image.convert('L')
  image.thumbnail([cfg.INPUT_WIDTH, cfg.INPUT_HEIGTH], Image.ANTIALIAS)
  return image


class XboxController(object):
  def __init__(self):
    try:
      print "Starting controller"
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


class NeuralNetwork(object):
  def __init__(self):
    self.model = load_model(cfg.MODEL)
    self.pb, self.w = get_pixel_buffer()
    self.real_controller = XboxController()
    self.last_action = [0, 0, 0, 0, 0]

    try:
      print "Starting neuralnet"
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
    else:
      image = capture_image(self.pb, self.w)
      vector = np.asarray(image).reshape(1, cfg.INPUT_HEIGTH, cfg.INPUT_WIDTH, cfg.COLOR_DIM).astype('float32') / 255
      pred = self.model.predict(vector)[0]
      self.last_action = [pred[0]*cfg.JOYSTICK_NORMALIZER, 0, round(pred[1]), 0, 0] # only do x and a
      self.flip = 0
      return self.last_action


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
