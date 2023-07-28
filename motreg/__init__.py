import os
if os.name == 'nt':
    from .bins.Release.ext import (g2o, utils, details, MotionModel)
else:
    from .libs.ext import (g2o, utils, details, MotionModel)
