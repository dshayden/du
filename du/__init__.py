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

  nEl = rgb.size/3
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

  nEl = lab.size/3
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
