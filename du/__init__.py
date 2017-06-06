__version__ = 0.1

def GetImgPaths(path, **kwargs):
  """Return sorted list of absolute-pathed images in a directory.
  
  Args:
    path (str): path to directory containing images.

  Keyword Args:
    types (str): regexp of permissible extensions (ext1|ext2|...)

  Example:
    >>> du.GetImgPaths('relative/path/to/images', types='(jpg|jpeg)')
      ['/abs/path/to/img1.jpg', '/abs/path/to/img2.jpg', ...]

    >>> du.GetImgPaths('path/to/images')
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
    >>> du.fileparts('/a/b/c.txt')
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
    >>> squares = du.Parfor(f, items)
  """
  from joblib import Parallel, delayed
  return Parallel(n_jobs=-1)(delayed(fcn)(i) for i in items)
