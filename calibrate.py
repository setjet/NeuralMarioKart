import gtk.gdk
import Image 
import numpy as np
import config as cfg

w = gtk.gdk.get_default_root_window()

pb = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, False, 8, cfg.CAPTURE_WIDTH, cfg.CAPTURE_HEIGTH)
pb = pb.get_from_drawable(w, w.get_colormap(), cfg.CAPTURE_X, cfg.CAPTURE_Y, 0, 0, cfg.CAPTURE_WIDTH, cfg.CAPTURE_HEIGTH)

image = Image.frombytes("RGB", (cfg.CAPTURE_WIDTH, cfg.CAPTURE_HEIGTH), pb.get_pixels())
image.thumbnail([cfg.INPUT_WIDTH, cfg.INPUT_HEIGTH], Image.ANTIALIAS)
image.show()
