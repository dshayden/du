__version__ = '0.2'

def GetFilePaths(path, types):
  """Return sorted list of absolute-pathed files in a directory.
  
  Args:
    path (str): path to directory containing files.
    types (str): regexp of valid file extensions (not including dot).

  Example:
    >>> GetFilePaths('relative/path/to/images', types='(jpg|jpeg)')
      ['/abs/path/to/img1.jpg', '/abs/path/to/img2.jpg', ...]

    >>> GetFilePaths('path/to/zips', types='zip')
      ['/abs/path/to/file1.zip', '/abs/path/to/file2.zip', ...]
  """
  import re, os, glob
  exp = re.compile('%s$' % types, re.IGNORECASE)
  files = glob.glob('%s%s*' % (path, os.path.sep))
  filesRel = [x for x in files if exp.findall(x)]
  filesAbs = [os.path.abspath(x) for x in filesRel]
  return sorted(filesAbs)

def GetImgPaths(path):
  """Return sorted list of absolute-pathed images in a directory.

  Will collect any files with the following extensions:
    jpg, jpeg, png, bmp, tif, tiff, ppm, pgm, pbm

  This is a convenience wrapper on GetFilePaths.
  
  Args:
    path (str): path to directory containing files.
  """
  types = '(jpg|jpeg|png|bmp|tif|tiff|ppm|pgm|pbm)'
  return GetFilePaths(path, types)

def fileparts(p):
  """Returns (base, name, ext) of file path p (whether or not it exists).
  
  Args:
    p (str): path of file

  Example:
    >>> fileparts('/a/b/c.txt')
      ('/a/b', 'c', '.txt')
    >>> fileparts('/a/b')
      ('/a', 'b', '')
  """
  import os
  base, fullname = os.path.split(p)
  ext = fullname.rfind('.')
  if ext==-1: return base, fullname, ''
  else:
    name, ext = (fullname[:ext], fullname[ext:])
    return base, name, ext

def Parfor(fcn, items):
  """Runs supplied function for each item in items in parallel via joblib.
  
  Args:
    fcn (function): function to be called.
    items (list): each element is passed to fcn on some thread.
                  if type(items[0]) == list or type(items[0]) == tuple:
                    it is assumed that each inner item is a list and will be
                    unpacked when passed to fcn (i.e. fcn(*items[0]))
                  else:
                    each item is passed directly to fcn (i.e. fcn(items[0]))

  Returns:
    List containing one entry for each function call, full of Nones if fcn does
    not return anything.

  Example:
    >>> def f(x): return x**2
    >>> items = [x for x in range(1000)]
    >>> squares = Parfor(f, items)
  """
  from joblib import Parallel, delayed
  if type(items[0]) == list or type(items[0]) == tuple:
    # unpack each inner list for function
    return Parallel(n_jobs=-1)(delayed(fcn)(*i) for i in items)
  else:
    # list of single items, pack each directly
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
    >>> import skimage.draw as draw
    >>> im = imread(...)
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

def imread(fname):
  """Read image into numpy array.

  Depth images are returned as uint16, color images are RGB uint8.

  Args:
    fname (str): path of image to read

  Returns:
    img (numpy.ndarray): desired image (either 1- or 3-channel).
  """
  import cv2
  im = cv2.imread(fname, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
  if im.ndim==3: im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
  return im

def rgbs2mp4(imgs, outFname, showOutput=False, crf=23, imExt='.jpg'):
  """pack RGB images into an mp4 using external call to ffmpeg.

  Args:
    imgs (list):       list of images, either filenames or numpy.ndarrays.
    outFname (str):    output filename, should have no ending other than mp4
    showOutput (bool): show ffmpeg video output.
    crf (int):         ffmpeg video quality factor between 0..51
                         0: lossless (not recommended)
                         18: visually lossless
                         23: normal
    imExt (str):       desired extension for temporarily writing images in
                       memory; only used if imgs is a list of numpy.ndarrays.
  """
  import numpy as np, subprocess

  if type(imgs[0]) == str:
    # files already on disk, run directly on them
    inDir, name, inExt = fileparts(imgs[0])

  elif type(imgs[0]) == np.ndarray:
    # write images to temp directory
    import tempfile, cv2
    inDir = tempfile.mkdtemp()
    inExt = '.jpg'
    nItems = len(imgs)
    fnames = ['%s/img-%08d%s' % (inDir, i, inExt) for i in range(nItems)]
    args = [(fnames[i], imgs[i]) for i in range(nItems)]
    Parfor(cv2.imwrite, args)

  # ensure output filename ends with mp4
  outDir, outName, outExt = fileparts(outFname)
  if len(outExt)==0: outFname = outFname + '.mp4'

  # format command
  cmd = "ffmpeg -y -pattern_type glob -i '%s/*%s' -c:v libx264 -crf %d -pix_fmt yuv420p %s"
  cmdStr = cmd % (inDir, inExt, crf, outFname)

  # run command
  stderrPipe = None if showOutput else subprocess.PIPE
  res = str(subprocess.Popen([cmdStr],stdout=subprocess.PIPE,
    stderr=stderrPipe, shell=True).communicate()[0])

def mat2im(x, vmin=None, vmax=None, cmap=None, lowColor=None, keepAlpha=False):
  """Convert values in array to colors according to a colormap.

  Typically used to colorize 2D arrays for better visualization, in which case,
  does the ~same preprocessing as Matlab's imagesc(...).

  Args:
    x (numpy.ndarray): array with arbitrary values.
    vmin (numeric): minimum value for colormap interpolation
    vmax (numeric): maximum value for colormap interpolation
    cmap (matplotlib.colors.Colormap): colormap to be used, will most often be
                                       a LinearSegmentedColormap.
    lowColor (array-like): RGB value to swap out with the smallest color in
                           cmap (i.e. for making 0 values black).
    keepAlpha (bool): return with alpha information.

  Returns:
    im: array s.t. im.ndim==x.ndim+1 and
          im.shape[-1]==3 if keepAlpha is false (the default)
          im.shape[-1]==4 otherwise.
  """
  import matplotlib, numpy as np

  # use default colormap
  if cmap is None:
    cmap = getattr(matplotlib.cm, matplotlib.rcParams['image.cmap'])

  if lowColor is not None:
    # make custom colormap from 256 colors, swap lowest out
    newColors = cmap(np.linspace(0,1,256))
    if len(lowColor)==3:
      lowColor = np.array([lowColor[0], lowColor[1], lowColor[2], 1])
    newColors[0,:] = lowColor
    cmap = matplotlib.colors.ListedColormap(newColors)
  
  # apply normalization and
  normalizer = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
  im = cmap(normalizer(x))
  if not keepAlpha:
    # don't know ndim in advance, but we know last dimension is size-4;
    # swap it to the first, remove the alpha channel, swap it back.
    im = np.moveaxis(im, source=-1, destination=0)
    im = im[0:3, :]
    im = np.moveaxis(im, source=0, destination=-1)
  return im

def ViewManyImages(imgs, titles=None):
  """Display set of images on a grid

  Args:
    imgs (sequence-like): sequence of images (filenames or numpy.ndarray)
    titles (str or sequence-like): if str, apply as suptitle, else if
                                   sequence-like, apply ith item to ith plot

  Returns:
    axs (list of matplotlib.axes.Axes): list of each subplot's axes.
  """
  import matplotlib.pyplot as plt, math
  axs = []
  nImgs = len(imgs)
  n = math.ceil(math.sqrt(nImgs))
  
  for i in range(nImgs):
    ax = plt.subplot(n, n, i+1)
    if type(imgs[i])==str: img = imread(imgs[i])
    else: img = imgs[i]
    plt.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    axs.append(ax)
    if titles is not None and type(titles) != str:
      ax.set_title(titles[i])

  if titles is not None and type(titles)==str:
    plt.suptitle(titles)

  return axs

def ViewPlots(idx, fcn, fig=None):
  """Create figure with keyboard callbacks to iterate over custom display code

  Args:
    idx (range/sequence-like): Valid indices to pass to fcn
    fcn (function): Display code for each index, must accept an integer argument
                    and can optionally accept keywords containing additional
                    information:
                      'idxMod': current left/right increment
                      'figure': reference to figure used for display.
    fig (matplotlib.figure.Figure): Figure to use, will create one if None

  Returns:
    (fig, cid): fig is matplotlib.figure.Figure used for display
                cid is an integer connection ID for use with
                  fig.canvas.mpl_disconnect for removing the key callback.

  INTERFACE:
    <left arrow>: iterate backwards by idxMod
    <right arrow>: iterate forwards by idxMod
    <up arrow>: double idxMod (no maximum)
    <down arrow>: half idxMod (minimum 1)
      idxMod is initialized to 1

  NOTE: Matplotlib is ~slow at plotting and key events get buffered so rapid,
        repeated taps could cause significant lag.
  """
  import matplotlib.pyplot as plt
  if fig is None: fig = plt.figure()

  curIdx = idx[0]
  idxMod = 1
  passKwargs = True

  def onKey(event):
    nonlocal idx; nonlocal curIdx; nonlocal idxMod; nonlocal fig;
    nonlocal passKwargs

    def getI(i): return max(idx[0], min(idx[-1], i))
    if event.key.find('left') != -1: curIdx = getI(curIdx - idxMod)
    elif event.key.find('right') != -1: curIdx = getI(curIdx + idxMod)
    elif event.key.find('up') != -1: idxMod = int(max(1, idxMod * 2))
    elif event.key.find('down') != -1: idxMod = int(max(1, idxMod / 2))
    kwargs = {'idxMod': idxMod, 'figure': fig}
    if passKwargs: fcn(curIdx, **kwargs)
    else: fcn(curIdx)
    fig.canvas.draw()

  cid = fig.canvas.mpl_connect('key_press_event', onKey)

  kwargs = {'idxMod': idxMod, 'figure': fig}
  try:
    fcn(curIdx, **kwargs)
  except TypeError:
    fcn(curIdx)
    passKwargs = False

  return (fig, cid)

def ViewImgs(imgs):
  """Interactively view collection of images using ViewPlots.

  Args:
    imgs (sequence): sequence of image filenames or numpy.ndarrays
  """
  import matplotlib.pyplot as plt
  nImgs = len(imgs)
  def ViewImg(i, **kwargs):
    nonlocal imgs; nonlocal nImgs
    img = imread(imgs[i]) if type(imgs[i])==str else imgs[i]
    plt.imshow(img)
    plt.title('%d of %d, idxMod: %d' % (i+1, nImgs, kwargs.get('idxMod', 0)))
  ViewPlots(range(len(imgs)), ViewImg)
  plt.show()

def splabel(img, nSegments, outFname, sigma=1, maxTargets=10, figure=None):
  """Interactively segment an image using superpixel refinements.

  Args:
    img (numpy.ndarray): image to segment, must be RGB
    nSegments (sequence-like): seq. of number of slic components to precompute
    outFname (str): output uint16 png image of segmentation labels
    sigma (float): SLIC sigma parameter
    maxTargets (int): maximum number of separate layers
    figure (matplotlib.figure.Figure): existing figure to use.

  INTERFACE:
    Click within a superpixel to add all pixels within boundary to the active
    segmentation layer. Change active segmentation layer / superpixel resolution
    with arrow keys; press 'w' to save results.

    Hotkeys
      <left arrow>: Change active segmentation layer down by 1
      <right arrow>: Change active segmentation layer up by 1
      <up arrow>: Recompute superpixels with +100 segments.
      <down arrow>: Recompute superpixels with -100 segments.
      h: hide / show superpixel boundaries
      w: write results to outFname
    
    NOTE: Active segmentation layer begins at 1.
    NOTE: Changing active segmentation layer to 0 allows for removal of
          segmentation masks on click.
  """
  import matplotlib.pyplot as plt, numpy as np
  import skimage.segmentation as seg
  import cv2

  base, name, ext = fileparts(outFname)
  if ext.lower() != '.png': raise ValueError('outFname must end in .png')

  colors = np.concatenate((diffcolors(maxTargets),
    0.6*np.ones((maxTargets,1))), axis=1)

  if figure is None: fig = figure(w=img.shape[1], h=img.shape[0])
  else: fig = figure

  if type(nSegments)==int: nSegments = (nSegments, )

  ax = fig.add_subplot(111)

  labelImg = np.zeros(img.shape[0:2], dtype=np.uint16)
  
  items = [(img, x, 10, 50, sigma) for x in nSegments]
  slicLabels = Parfor(seg.slic, items)

  slicIdx = 0
  curIdx = 1
  showSlic = True

  def redraw():
    nonlocal curIdx; nonlocal fig; nonlocal labelImg; nonlocal slicLabels
    nonlocal nSegments; nonlocal slicIdx
    spStr = 'SP Res: %d of %d' % (slicIdx, len(nSegments))

    if curIdx > 0: title = 'Click to label target %d; %s' % (curIdx, spStr)
    else: title = 'Click to remove target label; %s' % spStr
    plt.title(title)

    im = img.copy()
    uniqLabels = np.unique(labelImg)
    for i in uniqLabels[uniqLabels>0]:
      im = DrawOnImage(im, np.nonzero(labelImg==i), colors[i-1,:])

    if showSlic: ax.imshow(seg.mark_boundaries(im, slicLabels[slicIdx]))
    else: ax.imshow(im)
    fig.canvas.draw()

  def onKey(event):
    nonlocal curIdx; nonlocal slicLabels; nonlocal img; nonlocal nSegments
    nonlocal sigma; nonlocal showSlic; nonlocal labelImg; nonlocal outFname
    nonlocal slicIdx

    if event.key.find('left') != -1:
      curIdx = max(0, min(curIdx-1, maxTargets))
    if event.key.find('right') != -1:
      curIdx = max(0, min(curIdx+1, maxTargets))
    if event.key.find('up') != -1:
      slicIdx = min(len(nSegments)-1, slicIdx+1)
    if event.key.find('down') != -1:
      slicIdx = max(0, slicIdx-1)
    if event.key.lower() == 'h':
      showSlic = True if showSlic==False else False
    if event.key.lower() == 'w':
      cv2.imwrite(outFname, labelImg)
      print('Saved results to %s.' % outFname)
    redraw()

  def onClick(event):
    nonlocal curIdx; nonlocal slicLabels; nonlocal labelImg
    nonlocal slicIdx
    y,x = (int(event.ydata), int(event.xdata))
    labelImg[slicLabels[slicIdx]==slicLabels[slicIdx][y,x]] = curIdx
    redraw()

  ax = fig.add_subplot(111)
  # ax.imshow(seg.mark_boundaries(img, slicLabels, mode='thick'))
  ax.imshow(seg.mark_boundaries(img, slicLabels[slicIdx], mode='thick'))
  plt.axis('off')
  fig.canvas.mpl_connect('key_press_event', onKey)
  fig.canvas.mpl_connect('button_press_event', onClick)
  redraw()
  plt.show()

def imresize(img, size, resample='nearest'):
  """Resize image with given resampling method

  Args:
    img (numpy.ndarray): grayscale or color uint8 or uin16 image.
    size (numeric or tuple): desired size: numeric is fractional, tuple is (y,x)
                             in pixels.
    resample (string): one of 'nearest', 'bilinear', 'bicubic', 'lanczos'
                       NOTE: only 'nearest' works on uint16 data

  Returns:
    imgR (numpy.ndarray): resized image with the same datatype
  """
  import PIL.Image, numpy as np
  if type(size) in [list, tuple]: sz = (size[1], size[0])
  else: sz = (int(img.shape[1]*size), int(img.shape[0]*size))

  if resample=='nearest': method = PIL.Image.NEAREST
  elif resample=='bilinear': method = PIL.Image.BILINEAR
  elif resample=='bicubic': method = PIL.Image.BICUBIC
  elif resample=='lanczos': method = PIL.Image.LANCZOS
  else: raise TypeError("resample must be one of 'nearest', " \
    "'bilinear', 'bicubic', 'lanczos'")

  pilImg = PIL.Image.fromarray(img).resize(sz, method)
  return np.array(pilImg, dtype=img.dtype)
