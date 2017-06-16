import numpy as np

def logmvnpdfdiag(x, mu, sigma):
  """ Fast multivariate normal pdf for diagonal covariance.
  
  Args:
    x (ndarray): NxD array, N observations, D dimensions
    mu (ndarray): NxD or D-vector of means
    sigma (ndarray): NxD or D-vector of diagonal covariances
  
  Returns
    y (ndarray): N-vector of log-likelihoods
  """
  N,D = x.shape

  # normalization terms
  Dlog2pi = D*1.8379
  logdetSig = np.log(np.prod(sigma))

  # exponent terms
  xc = x - mu
  xcs = xc * 1/sigma
  logExp = np.sum(xcs*xc, axis=1)

  return -0.5*(Dlog2pi + logdetSig + logExp)

def Guass2DPoints(mu, Sigma, deviations):
  import numpy as np

  U, D, V = np.linalg.svd(Sigma)
  ori = np.arctan2(U[1,0], U[0,0])
  rot2d = np.array([[np.cos(ori), -np.sin(ori)],[np.sin(ori), np.cos(ori)]])

  a = deviations * np.sqrt(D[0])
  b = deviations * np.sqrt(D[1])

  t = np.linspace(0, 2*np.pi, 100)
  v = np.concatenate((a*np.cos(t)[np.newaxis,:], b*np.sin(t)[np.newaxis,:]))

  pts = rot2d.dot(v).T + mu
  x = pts[:,0]
  y = pts[:,1]

  return (x, y)
