import numpy as np
import scipy.interpolate

def catrnd(p):
  """ Sample categorical random variables.

  Args:
    p (ndarray): NxD array, N separate trials each with D outcomes. Assumes that
                 the columns of p all sum to 1.
  """
  cs = np.cumsum(p, axis=1)
  assert np.allclose(cs[:,-1], 1.0), 'du.stats.catrnd :: non-unit probabilities'
  rv = np.random.rand(p.shape[0])[:,np.newaxis]
  return np.argmax(rv <= cs, axis=1)

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
  if type(sigma)==float or len(sigma.shape)==1:
    logdetSig = np.log(np.prod(sigma))
  else:
    logdetSig = np.log(np.prod(sigma,axis=1))

  # exponent terms
  xc = x - mu
  xcs = xc * 1/sigma
  logExp = np.sum(xcs*xc, axis=1)

  return -0.5*(Dlog2pi + logdetSig + logExp)

def Gauss2DPoints(mu, Sigma, deviations=3):
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

def logUnifpdf(x, a, b):
  """ Log uniform pdf
  
  Args:
    x (ndarray): [N,D], N observations, D dimensions
    a (ndarray): [D,], per-dimension minimums
    b (ndarray): [D,], per-dimension maximums
  
  Returns
    y (ndarray): N-vector of log-likelihoods, is np.finfo(np.double).min for
                 probability-0 observations.
  """
  y = np.log(1 / (b-a)) * np.ones_like(x, dtype=np.double)
  validMask = np.logical_and(a <= x, x <= b)
  y[~validMask] = np.finfo(np.double).min
  return y

def logmvtpdf(X, mu, Sigma, v):
  """ Log multivariate-t pdf.

  INPUT
    X (ndarray, [N, D]): observations, each row is one observation.
    mu (ndarray, [D,]): mean
    Sigma (ndarray, [D, D]): shape
    v (float): degrees of freedom
  """
  from scipy.special import gammaln
  from scipy.stats._multivariate import _PSD
  from scipy.stats import multivariate_normal as mvn
  import numpy as np

  # Clean, efficient parameter checking/shaping and logdet/mahal
  D, mu, Sigma = mvn._process_parameters(None, mu, Sigma)
  X = mvn._process_quantiles(X, D)
  psd = _PSD(Sigma, allow_singular=False)

  # Constant term
  logZ_numerator = gammaln(0.5*(v+D))
  logZ_denom1 = gammaln(0.5 * v)
  logZ_denom2 = 0.5 * D * v
  logZ_denom3 = 0.5 * D * np.log(np.pi)
  logZ_denom4 = 0.5 * psd.log_pdet
  logZ_denom = logZ_denom1 + logZ_denom2 + logZ_denom3 + logZ_denom4
  logZ = logZ_numerator - logZ_denom
  
  # Data term
  xCtr = X - mu
  mahal = np.sum(np.square(np.dot(xCtr, psd.U)), axis=1)
  term1 = -(0.5 * (v+D)) * (1 + (1/v)*mahal)
  
  return logZ + term1

def mvtrnd(m, S, v, size=1):
  """ Sample multivariate-t RV.

  INPUT
    m (ndarray, [D,]): mean
    S (ndarray, [D, D]): covariance
    v (int): degrees of freedom
    size (int): number of samples
  """
  D = m.shape[0]
  g = np.tile(np.random.gamma(v/2., 2./v, size), (D,1)).T
  Z = np.random.multivariate_normal(np.zeros(D), S, size)
  return m + Z/np.sqrt(g)

def MH(x0, logp, q, logq=None, **kwargs):
  # Run Metropolis-Hastings with initial point x0
  # log posterior logp
  # transition q(x | xPrev)
  # log transition probability, None if symmetric
  nS = kwargs.get('nS', 1000)
  x0 = np.asarray(x0)
  
  ll = np.zeros(nS+1)
  accept = np.zeros(nS+1)
  xs = np.tile(x0, np.concatenate(([nS+1,], np.ones(x0.ndim, dtype=np.int))))
  ll[0] = logp(x0)

  us = np.random.rand(nS+1) # sample all random in advance
  for s in range(1, nS+1):
    # sample xp
    xp = q(xs[s-1])

    # evaluate log posterior under xp   
    log_xp = logp(xp)
    
    # evaluate log posterior under previous sample
    log_x = ll[s-1]

    # evaluate hastings ratio if logq is not None
    if logq is None:
      logq_xp = 1
      logq_x = 1
    else:
      logq_xp = logq(xp, xs[s-1])
      logq_x = logq(xs[s-1], xp)

    # sample uniform, accept/reject
    ratio = (log_xp + logq_x) - (log_x + logq_xp)
    if ratio >= 0: expRatio = 1.0
    else: expRatio = np.exp(ratio)
    accept[s] = us[s] < min(1.0, expRatio)
    if accept[s]:
      xs[s] = xp
      ll[s] = log_xp
    else:
      xs[s] = xs[s-1]
      ll[s] = log_x

  return xs[1:], ll[1:], accept[1:]
