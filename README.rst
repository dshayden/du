du
--------

Data handling utilities; originally made to ease the transition from Matlab to
Python, but has since grown to graeter functionality. Example use::
    >>> import du 
    >>> imgs = du.GetImgPaths('...')
    >>> def f(x): return x**2
    >>> squares = du.Parfor(f, range(10))

Some particularly useful functions include:
* save, load : save/load pickles parallel-compressed with pigz for small binaries (like Matlab)
* Parfor, ParforD, ParforT : Process- or Thread-level parallelism with a function and a list of inputs.
* tic, toc : timing functions, as in Matlab
* GetFilePaths, GetImgPaths : list of absolute paths of files with pattern.
* DrawOnIm : draw on image, including with alpha channel.
* diffcolors : get N different perceptually colors w/ ability to specify BG
* imread, imwrite, imresize : matlab-style image read/write to numpy arrays
* rgbs2mp4 : list of images to an mp4 file (requires ffmpeg to be installed)
* ViewPlots, ViewImgs : dynamically draw plots with hotkeys to cycle through
