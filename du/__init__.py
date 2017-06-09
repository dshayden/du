__version__ = 0.1

def GetImgPaths(path, **kwargs):
  """Return sorted list of absolute-pathed images in a directory.
  
  Args:
    path (str): path to directory containing images.

  Keyword Args:
    types (str): regexp of permissible extensions (ext1|ext2|...)

  Example:
    >>> GetImgPaths('relative/path/to/images', types='(jpg|jpeg)')
      ['/abs/path/to/img1.jpg', '/abs/path/to/img2.jpg', ...]

    >>> GetImgPaths('path/to/images')
      ['/abs/path/to/img1.jpg', '/abs/path/to/img2.bmp', ...]
  """
  import re, os, glob
  types = kwargs.get('types', '(jpg|jpeg|png|bmp|tif|tiff|ppm|pgm)')
  exp = re.compile('%s$' % types, re.IGNORECASE)
  files = glob.glob('%s%s*' % (path, os.path.sep))
  imgsRel = [x for x in files if exp.findall(x)]
  imgsAbs = [os.path.abspath(x) for x in imgsRel]
  return sorted(imgsAbs)

def fileparts(p):
  """Returns (base, name, ext) of file path p (whether or not it exists).
  
  Args:
    p (str): path of file

  Example:
    >>> fileparts('/a/b/c.txt')
      ('/a/b', 'c', '.txt')
  """
  import os
  base, fullname = os.path.split(p)
  ext = fullname.rfind('.')
  name, ext = (fullname[:ext], fullname[ext:])
  return base, name, ext

def Parfor(fcn, items):
  """Runs supplied function on each item in items in parallel via joblib.
  
  Args:
    fcn (function): function that takes one argument
    items (list): each element is passed to fcn on some thread

  Example:
    >>> def f(x): return x**2
    >>> items = [x for x in range(1000)]
    >>> squares = Parfor(f, items)
  """
  from joblib import Parallel, delayed
  return Parallel(n_jobs=-1)(delayed(fcn)(i) for i in items)

def rgb2lab(rgb):
  """Convert RGB to CIELAB colorspace.

  Args:
    rgb (array): Nx3 array of colors in either 0..1 or 0..255 range.

  Example:
    >>> rgb2lab([0,0,1])
      array([[  32.29566956,   79.18698883, -107.86175537]], dtype=float32)

    >>> rgb2lab([[0,0,255], [255,0,0]])
      array([[  32.29566956,   79.18698883, -107.86175537],
       [  53.24058533,   80.09418488,   67.20153809]], dtype=float32)
  """
  import cv2, numpy as np
  rgb = np.array(rgb)
  if rgb.ndim==1: rgb = np.expand_dims(rgb, axis=0)
  assert rgb.ndim == 2 and rgb.shape[1]==3, 'lab must be Nx3'

  nEl = rgb.size // 3
  rgbIm = rgb.reshape((nEl, 1, 3)).astype('float32')
  labIm = cv2.cvtColor(rgbIm, cv2.COLOR_RGB2LAB)
  return labIm.reshape((nEl, 3))

def lab2rgb(lab):
  """Convert CIELAB to RGB colorspace, return in 0..1 range.

  Args:
    lab (array): Nx3 array of lab colors as float32.

  Example:
    >>> lab = [[32.29566956, 79.18698883, -107.86175537],
               [53.24058533, 80.09418488, 67.20153809]]
      array([[9.63251296e-06, 0.00000000e+00, 1.00000000e+00],
             [9.99999762e-01, 0.00000000e+00, 2.72118496e-06]], dtype=float32)
  """
  import cv2, numpy as np
  lab = np.array(lab)
  if lab.ndim==1: lab = np.expand_dims(lab, axis=0)
  assert lab.ndim == 2 and lab.shape[1]==3, 'lab must be Nx3'

  nEl = lab.size // 3
  labIm = lab.reshape((nEl, 1, 3)).astype('float32')
  rgbIm = cv2.cvtColor(labIm, cv2.COLOR_LAB2RGB)
  return rgbIm.reshape((nEl, 3))

def diffcolors(nCols, bgCols=[1,1,1]):
  """Return Nx3 array of perceptually-different colors in [0..1] rgb range.

  Args:
    nCols (int): number of desired colors.
    bgCols (array): Nx3 list of colors to not be close to, values in [0..1].
  """
  import scipy.spatial.distance as distance, numpy as np

  nEl = 50
  assert nCols <= nEl**3, 'No more than %d distinct colors' % nEl**3
  x = np.linspace(0,1,nEl)
  r,g,b = np.meshgrid(x,x,x)
  rgb = np.stack((r.flatten(), g.flatten(), b.flatten()),
    axis=1).astype('float32')

  lab = rgb2lab(rgb)
  bgLab = rgb2lab(bgCols)

  minDist = 1e8 * np.ones((lab.shape[0]))
  minDist = np.minimum(np.min(distance.cdist(bgLab, lab), axis=0), minDist)

  colors = np.zeros((nCols, 3), dtype='float32')
  lastLab = np.expand_dims(bgLab[-1,:], axis=0)
  for i in range(nCols):
    minDist = np.minimum(distance.cdist(lastLab, lab), minDist)
    idx = np.argmax(minDist)
    colors[i,:] = rgb[idx,:]
    lastLab = np.expand_dims(lab[idx,:], axis=0)
  return colors

def DrawOnImage(img, coords, color):
  """Blends img with color at coords.

  Args:
    img (numpy.ndarray): M x N x 3 image, assumed to be RGB
    coords (sequence-like): length-2 sequence of row, column coordinates
    color (sequence-like): length-3 or 4 sequence of RGB/RGBA colors in [0,1]

  Example:
    >>> import matplotlib.image as mpimg, skimage.draw as draw
    >>> im = mpimg.imread('...')
    >>> coords = draw.circle(img.shape[0]/2, img.shape[1]/2, 40, shape=im.shape)
    >>> color = [1, 0, 0, 0.5]
    >>> imWithTransparentRedCircle = DrawOnImage(im, coords, color)
  """
  import numpy as np

  # get transparency
  if len(color)==3: alpha = 1
  elif len(color)==4: alpha = color[3]
  color = np.asarray(color[0:3])
  if img.dtype==np.uint8: color = (color*255).astype(np.uint8)
  else: color = color.astype(img.dtype)

  # make solid color image
  colorIm = np.ones_like(img)
  for i in range(3): colorIm[:,:,i] *= color[i]
  
  # blend images
  im = img.copy()
  im[coords[0],coords[1],:] = (1-alpha)*img[coords[0],coords[1],:] + \
    alpha*colorIm[coords[0],coords[1],:]
  return im

def GetScreenResolution():
  """Return (width, height) tuple of current screen's resolution
  """
  import sys
  if sys.platform == 'darwin':
    # screeninfo isn't supported on OSX so we have to run an external cmd
    import subprocess
    cmd = """osascript -e 'tell application "Finder" to get bounds of """ \
      """window of desktop'"""
    res = str(subprocess.Popen([cmd],stdout=subprocess.PIPE,
      shell=True).communicate()[0])
    toks = res.split(', ') # returns ['x', 'y', 'w', 'h']
    width, height = (int(toks[2]), int(toks[3]))
    return (width, height)
  else:
    import screeninfo
    sz = screeninfo.get_monitors()[0]
    return (sz.width, sz.height)

def figure(num=None, x=-1, y=-1, w=-1, h=-1):
  """Create/modify size/position of a matplotlib figure. Assumes Qt5 backend.

  Args:
    num (int): figure index; None if a new figure is desired.
    x,y,w,h:   Desired location and dimensions on current monitor, can be any of:
                 absolute (#pixels): must be greater than 1
                 relative (0..1): fraction of current monitor's geometry
                 unchanged (-1): don't change from current value

  Returns:
    hFig:      Reference to targeted/newly-created figure
  """
  import matplotlib, matplotlib.pyplot as plt
  if matplotlib.get_backend() != 'Qt5Agg':
    raise NotImplementedError('Only supports qt5 backend.')

  screenWidth, screenHeight = GetScreenResolution()

  winManager = plt.get_current_fig_manager()
  g = winManager.window.geometry()
  wx, wy, ww, wh = (g.x(), g.y(), g.width(), g.height())

  if x>=0 and x<=1: x*=screenWidth
  elif x>1:         None
  elif x==-1:       x = wx
  else:             raise ValueError('x is invalid')

  if y>=0 and y<=1: y*=screenHeight
  elif y>1:         None
  elif y==-1:       y = wy
  else:             raise ValueError('y is invalid')

  if w>=0 and w<=1: w*=screenWidth
  elif w>1:         None
  elif w==-1:       w = ww
  else:             raise ValueError('w is invalid')

  if h>=0 and h<=1: h*=screenHeight
  elif h>1:         None
  elif h==-1:       h = wh
  else:             raise ValueError('h is invalid')

  x = round(x); y = round(y); w = round(w); h = round(h)
  winManager.window.setGeometry(x, y, w, h)

  if num==None: num = winManager.num
  hFig = plt.figure(num)
  return hFig

def tic():
  """Store current time in as high a resolution as the system will give.

  Use with toc(), which returns the difference (in seconds) from the last tic().
  Nested calls are not supported.
  """
  import time
  tic.secs = time.time()

def toc():
  """Return elapsed seconds as float since tic() was last called.

  Use with toc(), which returns the difference (in seconds) from the last tic()
  Nested calls are not supported.
  Returns 0 if tic() has not been called.
  """
  import time
  if hasattr(tic, 'secs'): return time.time() - tic.secs
  else: return float(0)
