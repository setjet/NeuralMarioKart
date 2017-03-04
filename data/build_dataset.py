import time
import numpy as np
from util import XboxController, get_pixel_buffer, capture_image
import config as cfg


def _capture_controller(controller):
  c_in = controller.read()
  return [c_in[0]/cfg.JOYSTICK_NORMALIZER, c_in[2]] # only do x-axis and a

def _resolve_incomplete_entries(X, y):
  del X[-1]
  del y[-1]
  if len(y) > len(X):
    del y[-1]
  return X, y

def _save_data(data, name, fmt):
  handle = open(name + '.npy', 'a')
  np.savetxt(handle, data, fmt=fmt)
  handle.close()

def capture(capture_rate):
  X = []
  y = []

  try:
    print 'Capturing screen...'
    controller = XboxController()
    pb, w = get_pixel_buffer()
    while True:
      time.sleep(capture_rate)
      image = capture_image(pb, w)
      action = _capture_controller(controller)

      X.append(list(np.asarray(image).flatten()))
      y.append(action)

  except KeyboardInterrupt:
    print '^C received, writing to file'
    X, y = _resolve_incomplete_entries(X, y)
    _save_data(X, 'X', '%d')
    _save_data(y, 'y', '%.3f')

capture(0.2)
