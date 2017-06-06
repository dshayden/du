__version__ = 0.1

def GetImgPaths(path, **kwargs):
  """Return sorted list of absolute-pathed images in a directory.
  
  Args:
    path (str): path to directory containing images.

  Keyword Args:
    types (str): regexp of desired extensions (ext1|ext2|...)
  """
  import re, os, glob
  types = kwargs.get('types', '(jpg|jpeg|png|bmp|tif|tiff|ppm|pgm)')
  exp = re.compile('%s$' % types, re.IGNORECASE)
  files = glob.glob('%s%s*' % (path, os.path.sep))
  imgsRel = [x for x in files if exp.findall(x)]
  imgsAbs = [os.path.abspath(x) for x in imgsRel]
  return sorted(imgsAbs)

def fileparts(p):
  import os
  base, fullname = os.path.split(p)
  ext = fullname.rfind('.')
  name, ext = (fullname[:ext], fullname[ext:])
  return base, name, ext

def Parfor(fcn, inds):
  from joblib import Parallel, delayed
  return Parallel(n_jobs=-1)(delayed(fcn)(i) for i in inds)
