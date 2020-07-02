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

def rgb2lab(rgb, norm0255=False):
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
  if norm0255:
    labIm[:,:,0] *= (255.0 / 100.)
    labIm[:,:,1:3] += 128.0
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

def diffcolors(nCols, bgCols=[1,1,1], alpha=None, asFloat=True):
  """Return Nx3 or Nx4 array of perceptually-different colors in [0..1] rgb.

  Args:
    nCols (int): number of desired colors.
    bgCols (array): Nx3 list of colors to not be close to, values in [0..1].
    alpha (numeric): value between 0..1 to append to each color.
                     if None: return Nx3
                     else: return Nx4
    asFloat (bool): if True, return as float 0..1, else return uint8 as 0..255
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

  if alpha is not None:
    colors = np.concatenate((colors, alpha*np.ones((nCols,1))), axis=1)
  if not asFloat:
    colors = (255*colors).astype(np.uint8)

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
  cords = []
  cords.append(np.maximum(0, np.minimum(im.shape[0]-1, np.array(coords[0]))))
  cords.append(np.maximum(0, np.minimum(im.shape[1]-1, np.array(coords[1]))))
  # coords[0] = np.maximum(0, np.minimum(im.shape[0], np.array(coords[0])))
  # coords[1] = np.maximum(0, np.minimum(im.shape[1], np.array(coords[1])))

  im[cords[0],cords[1],:] = (1-alpha)*img[cords[0],cords[1],:] + \
    alpha*colorIm[cords[0],cords[1],:]

  # im[coords[0],coords[1],:] = (1-alpha)*img[coords[0],coords[1],:] + \
  #   alpha*colorIm[coords[0],coords[1],:]
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
  # if im.ndim==3: im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
  if im.ndim==3: im = im[:,:,[2, 1, 0]]
  return im

def rgbs2mp4(imgs, outFname, showOutput=False, crf=23, imExt='.jpg', fps=25):
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
    args = [(imgs[i], fnames[i]) for i in range(nItems)]
    ParforT(imwrite, args)

  # ensure output filename ends with mp4
  outDir, outName, outExt = fileparts(outFname)
  if len(outExt)==0: outFname = outFname + '.mp4'

  # format command
  # cmd = "ffmpeg -y -pattern_type glob -i '%s/*%s' -c:v libx264 -crf %d -pix_fmt yuv420p %s"
  cmd = "ffmpeg -framerate %d -y -pattern_type glob -i '%s/*%s' -c:v libx264 -crf %d -pix_fmt yuv420p %s"
  # cmdStr = cmd % (inDir, inExt, crf, outFname)
  cmdStr = cmd % (fps, inDir, inExt, crf, outFname)

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

  return (im*255).astype(np.uint8)

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

def ViewPlots(idx, fcn, figure=None):
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
  if figure is None: fig = plt.figure()
  else: fig = figure

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
    fig.clf()
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
  if type(size) in [list, tuple, np.ndarray]: sz = (size[1], size[0])
  else: sz = (int(img.shape[1]*size), int(img.shape[0]*size))

  if resample=='nearest': method = PIL.Image.NEAREST
  elif resample=='bilinear': method = PIL.Image.BILINEAR
  elif resample=='bicubic': method = PIL.Image.BICUBIC
  elif resample=='lanczos': method = PIL.Image.LANCZOS
  else: raise TypeError("resample must be one of 'nearest', " \
    "'bilinear', 'bicubic', 'lanczos'")

  pilImg = PIL.Image.fromarray(img).resize(sz, method)
  return np.array(pilImg, dtype=img.dtype)

def imwrite(img, fname):
  """Write RGB/grayscale/high-depth image to file

  Args:
    img (numpy.ndarray): grayscale (uint8/16) or RGB uint8 image.
    fname (str): output filename with desired extension
  """
  import numpy as np, cv2
  if img.dtype==np.uint8 and img.ndim==3 and img.shape[2] == 3:
    img = np.stack((img[:,:,2], img[:,:,1], img[:,:,0]),axis=2)
  cv2.imwrite(fname, img)

def getrect(ax=None):
  """ Get rectangle from the user

  Args:
    ax (matplotlib.axes): Handle to axes, will use gca() if None

  Returns:
    r (list): Rectangle with x, y, w, h coordinates

  NOTE: This function won't return until the user closes the plot (i.e. after
        dragging out their rectangle). Hacky, but works for now.
  """
  import matplotlib.pyplot as plt
  import matplotlib.widgets
  if ax is None: ax = plt.gca()

  r = []; rs = []
  def onKey(evt): None #holds reference to rs
  def onEvt(eClick, eRelease):
    nonlocal r; nonlocal rs;
    x1, y1 = eClick.xdata, eClick.ydata
    x2, y2 = eRelease.xdata, eRelease.ydata

    x = min(x1, x2); y = min(y1, y2)
    w = max(x1, x2) - x; h = max(y1, y2) - y
    r = (x,y,w,h)
    rs.set_active(False)
    print('Close figure to get rectangle')
  onKey.rs = matplotlib.widgets.RectangleSelector(ax, onEvt, drawtype='box')
  rs = onKey.rs
  plt.connect('key_press_event', onKey)
  plt.show()
  return r

def rect2mask(rect, shape):
  """ Convert xywh-rectangle to mask with given (y,x) shape.

  Args:
    rect (sequence-like): Rectangle with x,y,w,h elements.
    shape (sequence-like): Desired (nRows, nCols) shape.

  Returns:
    mask (numpy.ndarray): Boolean mask of rect over shape.
  """
  import numpy as np

  mask = np.zeros(shape[0:2], dtype=np.bool)
  r = np.arange(shape[0]); c = np.arange(shape[1])

  rIdx = np.bitwise_and(r>=rect[1], r<=(rect[1]+rect[3]))
  cIdx = np.bitwise_and(c>=rect[0], c<=(rect[0]+rect[2]))

  idx = np.ix_(rIdx, cIdx)
  mask[idx] = True

  return mask

def changem(Z, oldCode, newCode):
  """

  INPUT
    Z (ndarray): Array to replace values in.
    oldCode (sequence-like): values to be replaced.
    newCode (sequence-like): values to replace with. Same size as oldCode.

  OUPTUT
    newZ (ndarray): Array with replaced values.
  """
  import numpy as np
  Z = np.asarray(Z)
  newZ = Z.copy()
  for old, new in zip(oldCode, newCode): newZ[Z==old] = new
  return newZ

# def changem(Z, oldCode, newCode):
#   """ Replace values in array Z, in-place.
#
#   Args:
#     Z (ndarray): Array to replace values in.
#     oldCode (sequence-like): values to be replaced.
#     newCode (sequence-like): values to replace with. Same size as oldCode.
#   """
#   for old, new in zip(oldCode, newCode): Z[Z==old] = new

def load(fname, msgpack=False):
  """ Load a gzipped, pickled, object.

  Args:
    fname (str): file to load from, will append .gz if there is no ext.
    msgpack (bool): if True, decode with msgpack, not pickle
  """
  import gzip
  base, name, ext = fileparts(fname)
  if ext != '.picklez' and ext != '.gz': fname += '.gz'
  f = gzip.open(fname, 'rb')
  if msgpack:
    import msgpack, msgpack_numpy as m
    m.patch()
    packedBytes = f.read()
    obj = m.unpackb(packedBytes)
  else:
    import pickle
    obj = pickle.load(f)
  f.close()
  return obj

def save(fname, obj, level=6, msgpack=False):
  """ Save an object as a pickle, then parallel-gzip with pigz.

  WARNING: You must have pigz installed and callable from the command-line.

  Args:
    fname (str): File to save to then overwrite, appending .gz.
    obj (object): Python object that can be pickled.
    level (int): 0..9, 9 is higher compression, 0 is lower.
    msgpack (bool): if True, use msgpack, else use pickle.
  """
  import subprocess, os
  level = int(max(1, min(9, level)))
  f = open(fname, 'wb')
  if msgpack:
    import msgpack, msgpack_numpy as m
    m.patch()
    packedBytes = m.packb(obj)
    f.write(packedBytes)
  else:
    import pickle
    pickle.dump(obj, f, protocol=4)
  f.close()
  parts = fileparts(fname)

  if parts[2].lower() == '.gz':
    os.rename(fname, ''.join(parts[:2]))
    fname = ''.join(parts[:2])
  try:
    subprocess.call(('pigz', '-f', '-%d' % level, fname))
  except:
    print('Error in compressing saved file; Make sure pigz is installed (https://zlib.net/pigz/ or https://blog.kowalczyk.info/software/pigz-for-windows.html)')

def ParforT(fcn, items, nWorkers=None, unpack=None, returnExc=False, **kwargs):
  """Runs supplied function for each item in items in parallel via concurrent
  
  Args:
    fcn (function): function to be called.
    items (list): each element is passed to fcn on some thread.
                  if type(items[0]) == list or type(items[0]) == tuple:
                    it is assumed that each inner item is a list and will be
                    unpacked when passed to fcn (i.e. fcn(*items[0]))
                  else:
                    each item is passed directly to fcn (i.e. fcn(items[0]))
    nWorkers (int): Number of threads to use.
    unpack (bool): Force unpacking / not unpacking arguments.

  Returns:
    List containing one entry for each function call, full of Nones if fcn does
    not return anything. Items can be exceptions if problems were encountered.

  Note:
    This can be more convenient to use than Parfor because no serialization is
    required, hence inner functions (and closures) can be used, whereas
    with Parfor, functions must be serialized for different processes to run
    them. The drawback of ParforT is that threads are subject to the GIL.

  Example:
    >>> def f(x): return x**2
    >>> items = [x for x in range(1000)]
    >>> squares = ParforT(f, items)
  """
  import concurrent.futures, os
  from tqdm import tqdm

  futures = []
  res = []
  with concurrent.futures.ThreadPoolExecutor(max_workers=nWorkers) as pool:
    if unpack is None:
      if type(items[0]) == list or type(items[0]) == tuple: unpack = True
      else: unpack = False

    if unpack:
      for i in range(len(items)): futures.append(pool.submit(fcn, *items[i]))
    else:
      for i in range(len(items)): futures.append(pool.submit(fcn, items[i]))

    # progress bar
    if kwargs.get('showProgress', True):
      pbKwargs = {'total': len(futures), 'unit': 'it', 'unit_scale': True,
        'leave': True}
      for f in tqdm(concurrent.futures.as_completed(futures), **pbKwargs): pass

    excIdx = []
    for i in range(len(items)):
      try:
        data = futures[i].result()
        res.append(data)
      except Exception as exc:
        res.append(exc)
        excIdx.append(i)

  if len(excIdx) > 0:
    print('WARNING: Exception in one or more workers!')
    raise res[excIdx[0]]

  if returnExc: return res, excIdx
  else: return res

def For(fcn, items, nWorkers=None, unpack=None, **kwargs):
  """Runs supplied function for each item in items in parallel via concurrent
  
  Args:
    fcn (function): function to be called.
    items (list): each element is passed to fcn on some thread.
                  if type(items[0]) == list or type(items[0]) == tuple:
                    it is assumed that each inner item is a list and will be
                    unpacked when passed to fcn (i.e. fcn(*items[0]))
                  else:
                    each item is passed directly to fcn (i.e. fcn(items[0]))
    nWorkers (int): Dummy parameter for compatibility with Parfor, ParforT
    unpack (bool): Force unpacking / not unpacking arguments.

  Returns:
    List containing one entry for each function call, full of Nones if fcn does
    not return anything. Items can be exceptions if problems were encountered.

  Note:
    This is most useful for diagnosing problems with Parfor / ParforT .

  Example:
    >>> def f(x): return x**2
    >>> items = [x for x in range(1000)]
    >>> squares = For(f, items)
  """
  import progressbar
  if unpack is None:
    if type(items[0]) == list or type(items[0]) == tuple: unpack = True
    else: unpack = False
  if nWorkers is None: nWorkers = -1
  if kwargs.get('showProgress', False):
    res = []
    bar = progressbar.ProgressBar()
    if unpack:
      for i in bar(range(len(items))):
        res.append(fcn(*items[i]))
    else:
      for i in bar(range(len(items))):
        res.append(fcn(items[i]))
    return res
  else:
    if unpack: return [fcn(*i) for i in items]
    else: return [fcn(i) for i in items]

def TextOnImage(img, text, loc=(10,10), fontsize=24, color=(0,0,0,1), bg=None):
  """Draw text on an rgb image with basic multiline support.

  Args:
    img (numpy.ndarray): grayscale or color uint8 or uin16 image.
    text (str): text to rasterize.
    loc (tuple): y,x location of text
    fontsize (int): font size in points
    color (tuple): rgb[a] in range 0..1
    bg (tuple): None or rgb[a] in range 0..1

  Returns:
    imgT (ndarray): image with text on it
    mask (ndarray): image mask where text / background is drawn
  """
  import PIL, PIL.ImageFont, PIL.ImageDraw, numpy as np
  if len(color)==3: color = (color[0], color[1], color[2], 1)

  fontpath = '%s/monaco.ttf' % fileparts(__file__)[0]
  font = PIL.ImageFont.truetype(fontpath, fontsize)
  
  pilImg = PIL.Image.fromarray(img)
  ctx = PIL.ImageDraw.Draw(pilImg)
  color = [int(x*255) for x in color]
  ctx.text((loc[1], loc[0]), text, (color[0], color[1], color[2],
    color[3]), font=font)
  del ctx

  # basic handling of newlines, though best to just not use
  numNewlines = text.count('\n')
  if numNewlines>0:
    from functools import reduce
    lines = text.split('\n')
    longest = reduce(lambda x, y: max(x,y), map(len, lines))
    for i in range(len(lines)):
      if len(lines[i])==longest:
        fontsz = font.getsize(lines[i])
        break
  else:
    fontsz = font.getsize(text)

  mask = np.zeros(img.shape[0:2], dtype=np.bool)
  yrng = np.arange(loc[0], (loc[0]+fontsz[1])*(numNewlines+1))
  xrng = np.arange(loc[1], loc[1]+fontsz[0])

  try:
    mask[np.ix_(yrng, xrng)] = True
  except Exception as exc:
    print('Font will not fit, returning original image and 0 mask')
    return img, mask

  if loc[0]+fontsz[1] > img.shape[0]:
    print('Warning: Font exceeds image height')
  if loc[1]+fontsz[0] > img.shape[1]:
    print('Warning: Font exceeds image width')

  imgT = np.array(pilImg, dtype=img.dtype)
  if bg is not None: imgT = DrawOnImage(imgT, mask.nonzero(), bg)

  return imgT, mask

def savepcd(pts, shape, fname):
  """Save array of points into .pcd format.

  Args:
    pts (ndarray): Nx3 (xyz) or Nx6 (xyzrgb) array, rgb \in [0, 255]
    shape (tuple): Original image shape in pixels
    fname (str): name of file to save out to, should end in .pcd
  """
  from io import StringIO, BytesIO
  import numpy as np

  def packRgb(rgb):
    # pack RGB color to float
    rgb = rgb.astype(np.uint32)
    rgb = np.array((rgb[:, 0] << 16) | (rgb[:, 1] << 8) | (rgb[:, 2] << 0),
      dtype=np.uint32)
    rgb.dtype = np.float32
    return rgb

  if pts.shape[1] == 3:
    fields = 'x y z'
    size = '4 4 4'
    count = '1 1 1'
    types = 'F F F'
  elif pts.shape[1] == 6:
    pts = pts.astype(np.float32)
    pts = np.concatenate((pts[:,0:3], packRgb(pts[:,3:6])[:,np.newaxis]),
      axis=1)
    fields = 'x y z rgb'
    size = '4 4 4 4'
    count = '1 1 1 1'
    types = 'F F F F'

  with open(fname, 'w') as f:
    # header
    print('VERSION .7', file=f)
    print('FIELDS %s' % fields, file=f)
    print('SIZE %s' % size, file=f)
    print('TYPE %s' % types, file=f)
    print('COUNT %s' % count, file=f)
    print('WIDTH %d' % shape[1], file=f)
    print('HEIGHT %d' % shape[0], file=f)
    print('VIEWPOINT 0 0 0 1 0 0 0', file=f)
    print('POINTS %d' % pts.shape[0], file=f)
    print('DATA ascii', file=f)

    import tempfile
    fd, name = tempfile.mkstemp(suffix='.txt', text=True)
    np.savetxt(name, pts)
    fd = open(name, 'r')
    print(fd.read(), file=f, end='')
    fd.close()

def mrdivide(B, A):
  """Solve xA = B for x. Similar to Matlab's B/A"""
  import numpy as np
  return np.linalg.solve(A.T, B.T).T

def RectCoords(R, fill=False, shape=None):
  """ Get rectangle row, column coordinates for drawing in an image.

  Args:
    R: 4-vector of xywh.
    fill: boolean, true for inside of rect, false for perimeter.
    shape: image shape to be confined to.

  Returns:
    (rr, cc): row, column coordinates

  Related:
    DrawOnImage
  """
  import skimage.draw as draw
  r = ( R[1], R[1]+R[3], R[1]+R[3], R[1] )
  c = ( R[0], R[0], R[0]+R[2], R[0]+R[2] )
  if fill: return draw.polygon(r,c,shape)
  else: return draw.polygon_perimeter(r,c,shape)

def asShape(x, shape):
  """ Return copy of array x with given shape.

  INPUTS
    x (ndarray): array of arbitrary shape
    shape (tuple): new shape
  
  OUTPUTS
    y (ndarray): copy of x with shape shape.
  """
  y = x.copy()
  y.shape = shape
  return y

def imgs(x, shape=None, plotShape=None):
  """ Show each slice of array as an image in its own subplot, assumes axis 0. """
  import matplotlib.pyplot as plt, numpy as np

  nImgs = x.shape[0]
  if plotShape is None:
    nR = np.ceil(np.sqrt(nImgs))
    nC = nR
  else:
    nR, nC = plotShape

  for i in range(nImgs):
    plt.subplot(nR, nC, i+1)
    im = x[i] if shape is None else asShape(x[i], shape)
    plt.imshow(im)
    plt.xticks([])
    plt.yticks([])

def rect2slice(rect, shape):
  """ Get bounds-checked y,x slices from an xywh-rectangle.

  Note: Slices will generate views rather than copies of ndarrays.

  INPUTS
    rect (tuple): (4,) xywh rectangle, arbitrary type.
    shape (tuple): (2,) (rows, columns) of the desired bounds.

  OUTPUTS
    rr (slice): slice for rows
    cc (slice): slice for columns
  """
  import numpy as np
  r = (np.max((0, int(rect[1]))), np.min((shape[0], int(rect[1]+rect[3]))))
  c = (np.max((0, int(rect[0]))), np.min((shape[1], int(rect[0]+rect[2]))))
  rr = slice(r[0], r[1], 1)
  cc = slice(c[0], c[1], 1)
  return (rr,cc)

def roc(yhat, y, show=False):
  import numpy as np
  import matplotlib.pyplot as plt

  # tprs = [0.]
  # fprs = [0.]
  tprs = []; fprs = []

  for tau in np.arange(0,1.05,.05):
    tp = np.sum(np.logical_and(yhat>=tau, y))
    fp = np.sum(np.logical_and(yhat>=tau, ~y))
    fn = np.sum(np.logical_and(yhat<tau, y))
    tn = np.sum(np.logical_and(yhat<tau, ~y))

    tprs.append(tp / (tp+fn))
    fprs.append(fp / (fp+tn))

  # tprs.append(1.)
  # fprs.append(1.)
  
  print(np.trapz(tprs, fprs))

  plt.plot(fprs, tprs)
  plt.show()

def savepts(pts, fname):
  """Save array of points into .txt format.

  Args:
    pts (ndarray): Nx7 (xyzrgba) array, rgb \in [0, 1] 
    fname (str): name of file to save out to, should end in .pcd
  """
  import tempfile, numpy as np
  with open(fname, 'w') as f:
    fd, name = tempfile.mkstemp(suffix='.txt', text=True)
    np.savetxt(name, pts)
    fd = open(name, 'r')
    print('%d' % pts.shape[0], file=f)
    print(fd.read(), file=f, end='')
    fd.close()

def imreadStack(paths, parallel=True):
  # read first image to get shape and type
  import numpy as np
  im = imread(paths[0])
  imgs = np.zeros((len(paths),) + im.shape, dtype=im.dtype)
  def worker(idx): imgs[idx] = imread(paths[idx])
  _, excIdx = ParforT(worker, range(len(imgs)), returnExc=True)
  return imgs, excIdx

def Parfor(fcn, items, nWorkers=None, unpack=None, returnExc=False, **kwargs):
  """Runs supplied function for each item in items in parallel (over processes)
  
  Args:
    fcn (function): function to be called.
    items (list): each element is passed to fcn on some process.
                  if type(items[0]) == list or type(items[0]) == tuple:
                    it is assumed that each inner item is a list and will be
                    unpacked when passed to fcn (i.e. fcn(*items[0]))
                  else:
                    each item is passed directly to fcn (i.e. fcn(items[0]))
    nWorkers (int): Number of threads to use.
    unpack (bool): Force unpacking / not unpacking arguments.
    returnExc (bool): Return tuple of (results, exceptions)
  
  Keyword Args:
    showProgress (bool): Print a progress bar to console. Default True.

  Returns:
    List containing one entry for each function call, full of Nones if fcn does
    not return anything. Items can be exceptions if problems were encountered.

  Example:
    >>> def f(x): return x**2
    >>> items = [x for x in range(1000)]
    >>> squares = Parfor2(f, items)
  """
  import concurrent.futures, os
  from tqdm import tqdm

  futures = []
  res = []
  with concurrent.futures.ProcessPoolExecutor(max_workers=nWorkers) as pool:
    if unpack is None:
      if type(items[0]) == list or type(items[0]) == tuple: unpack = True
      else: unpack = False

    if unpack:
      for i in range(len(items)): futures.append(pool.submit(fcn, *items[i]))
    else:
      # futures = [pool.submit(fcn, items[i]) for i in range(len(items))]
      for i in range(len(items)): futures.append(pool.submit(fcn, items[i]))
    
    # progress bar
    if kwargs.get('showProgress', True):
      pbKwargs = {'total': len(futures), 'unit': 'it', 'unit_scale': True,
        'leave': True, 'miniters': 1}
      for f in tqdm(concurrent.futures.as_completed(futures), **pbKwargs): pass

    # exceptions
    excIdx = []
    for i in range(len(items)):
      try:
        data = futures[i].result()
        res.append(data)
      except Exception as exc:
        res.append(exc)
        excIdx.append(i)

  for i in range(len(items)):
    if type(res[i]) is Exception:
      print('Warning: exceptions detected in ParforT results')
      break

  if returnExc: return res, excIdx
  else: return res

def rgb2labImg(rgb, norm0255=True):
  """ Convert RGB image to 0..255 normalized CIELAB colorspace.

  Args:
    rgb (array): MxNx3 array of colors in 0..255 range.
  """
  import skimage.color as sc, numpy as np
  lab = sc.rgb2lab(rgb)
  if norm0255:
    lab[:,:,0] *= 255/100.
    lab[:,:,1:] += 128
  lab = lab.astype(np.uint8)
  return lab

def lab2rgbImg(lab, norm0255=True):
  """ Convert Lab image to uint8 0..255 RGB.

  Args:
    lab (array): MxNx3 array of lab colors. Treated as 0..255 if dtype if uint8.
    norm0255 (bool): if True, output is made uint8 0..255, else output is 0..1

  Returns:
    rgb (array): MxNx3 array of rgb colors.
  """
  import skimage.color as sc, numpy as np
  labf = lab.copy()
  if labf.dtype == np.uint8:
    labf = labf.astype(np.double)
    labf[:,:,0] /= 255/100.
    labf[:,:,1:] -= 128
  rgb = sc.lab2rgb(labf)
  # if lab.dtype == np.uint8:
  #   labf = lab.astype(np.double)
  #   labf[:,:,0] /= 255/100.
  #   labf[:,:,1:] -= 128
  # else:
  #   labf = lab
  # rgb = sc.lab2rgb(labf)
  return rgb

def ParforD(fcn, items, nWorkers=None, unpack=None, **kwargs):
  """Runs supplied function for each item in items in parallel via distex 
  
  Args:
    fcn (function): function to be called.
    items (list): each element is passed to fcn on some thread.
                  if type(items[0]) == list or type(items[0]) == tuple:
                    it is assumed that each inner item is a list and will be
                    unpacked when passed to fcn (i.e. fcn(*items[0]))
                  else:
                    each item is passed directly to fcn (i.e. fcn(items[0]))
    nWorkers (int): Number of threads to use. None for default number of CPUs.
    unpack (bool): Force unpacking / not unpacking arguments.

  Returns:
    List containing one entry for each function call, full of Nones if fcn does
    not return anything. Items can be exceptions if problems were encountered.

  Note:
    This runs separate processes; fcn must have all imports within it and cannot
    be a lambda.

  Example:
    >>> def f(x): return x**2
    >>> items = [x for x in range(1000)]
    >>> squares = ParforD(f, items)
  """
  import distex, os
  from tqdm import tqdm

  if unpack is None:
    if type(items[0]) == list or type(items[0]) == tuple: unpack = True
    else: unpack = False

  pool = distex.Pool(num_workers=nWorkers, data_pickle=distex.PickleType.dill)
  if kwargs.get('showProgress', True):
    res = list(tqdm(pool.map(fcn, items, star=unpack), total=len(items)))
  else:
    res = list(pool.map(fcn, items, star=unpack))

  pool.shutdown()
  return res

def eigh_proper_all(sigma):
  """ Get all 2^{D-1} eigendecompositions UDU^T s.t. det(U) = 1

  INPUT
    sigma (ndarray, [D,D]): positive definite matrix

  OUTPUT
    sigD (ndarray, [D,]): eigenvalues
    sigU (ndarray, [2^{D-1}, D, D]): possible proper rotations
  """
  import numpy as np, itertools
  
  sigD, sigU = np.linalg.eigh(sigma)
  D = sigma.shape[0]

  # combinations
  combs = []
  for t in range(D+1):
    combs = combs + list(itertools.combinations(range(D), t))

  possible = np.zeros((2**(D-1), D, D))
  cnt = 0
  for c in combs:
    sU = sigU.copy()
    sU[:,c] = -sU[:,c]
    if np.linalg.det(sU) < 0: continue
    UDU = sU.dot(np.diag(sigD)).dot(sU.T)
    norm = np.linalg.norm( UDU - sigma )
    assert norm < 1e-8, 'bad norm'
    possible[cnt] = sU
    cnt += 1
  sigU = possible
  return sigD, sigU 

def mu_sig(ys):
  """ Efficiently compute mu, sigma of ys with shape (T, N, D) or (N, D)

  if ndim(ys)==3 then efficiently compute mean and covariance along first axis
  else return efficient mean and covariance along first axis

  INPUT
    ys (ndarray, [N,D] or [T,N,D]): N obs, D dimension, (optional) T timestep

  OUTPUT
    mu (ndarray, [D,] or [T,D]): mu at each time
    sig (ndarray, [D,D] or [T,D,D]): sig at each time
  """
  import numpy as np
  if ys.ndim==2: return np.mean(ys, axis=0), np.cov(ys.T)
  assert ys.ndim==3, 'ys must be dimension 2 or 3'
  N = ys.shape[1]
  mu = np.mean(ys, axis=1)
  yctr = ys - mu[:,np.newaxis,:]
  sig = np.einsum('tij,tik->tjk', yctr, yctr) / (N-1)
  return mu, sig

def scatter_matrix(ys, center=True):
  """ compute centered or uncentered data scatter matrix.

  INPUT
    ys (ndarray, [N x D]): N obs, D dimension
    center (bool): center the data

  OUTPUT
    sc (ndarray, [D,D]): \sum_n z[n] z[n].T where z[n] =
                           ys[n] - np.mean(ys,axis=0) if center=True
                           ys[n] otherwise
  """               
  import numpy as np
  if ys.ndim == 1: return np.outer(ys, ys)
  y = ys - np.mean(ys, axis=0) if center else ys
  return np.einsum('ij,ik->jk', y, y)

def SavePlots(idx, fcn, names):
  import matplotlib.pyplot as plt
  if type(names) != list and type(names) != tuple: names = [names, ]

  def save(i):
    fcn(i)
    plt.savefig(names[i], dpi=300, bbox_inches='tight')
    plt.close()
  For(save, idx)

def GetDataLims(y, **kwargs):
  # get lower, upper limits of data along each of last axis
  import numpy as np
  K = y.shape[-1]
  extra = kwargs.get('extra', 0.1)

  axReduce = tuple(np.arange(y.ndim-1).tolist())
  mins = np.min(y, axis=axReduce)
  maxs = np.max(y, axis=axReduce)
  diff = (maxs - mins) * extra
  return np.array((mins - diff, maxs - diff)).T
