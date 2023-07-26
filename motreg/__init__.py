import glob
from ctypes import CDLL
import os
from .libs.ext import (g2o, utils, details, MotionModel)

library_path = glob.glob(os.path.join(os.path.dirname(__file__), 'libs/libg2o*.so'))
library_path += glob.glob(os.path.join(os.path.dirname(__file__), 'libs/libg2o*.dll'))
clib = [CDLL(_) for _ in library_path]
