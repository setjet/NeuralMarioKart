import gtk.gdk
import Image 
import numpy as np

w = gtk.gdk.get_default_root_window()

width = 640
heigth = 480
x_pos = 640
y_pos = 325

pb = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, False, 8, width, heigth)
pb = pb.get_from_drawable(w, w.get_colormap(), x_pos, y_pos, 0, 0, width, heigth)
image = Image.frombytes("RGB", (width,heigth), pb.get_pixels())
image.show()
