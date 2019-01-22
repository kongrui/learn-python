import numpy as np

from scipy.signal import welch
from scipy.signal.windows import boxcar, hann

import matplotlib
matplotlib.use('agg')
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt

def hanningSegment():
  # does not work
  y = np.loadtxt('data.512.csv', delimiter=",", unpack=True)
  datos = y
  print(datos.size)
  N = 512
  fs = N * 1.0 / (3 - 0)
  nblock = N
  win = hann(nblock, sym=False)
  f, Pxxf = welch(datos, fs, window=win, nfft=nblock, return_onesided=True, detrend=False)
  print(Pxxf.max())
  print(Pxxf.size)
  idxs = Pxxf.argsort()[-5:]
  print(idxs)
  print(Pxxf[idxs] * 4)
  #plt.scatter(f, Pxxf)
  #plt.grid()
  #plt.show()
  """
512
0.9941412859746589
257
[36 60 13 11 12]
[5.24180895e-06 5.38116270e-06 9.93953149e-01 9.94206460e-01
 3.97656514e+00]
  """

def multiplSegments():
  y = np.loadtxt('data.512.csv', delimiter=",", unpack=True)
  datos = y
  print(datos.size)
  N = 512
  fs = N * 1.0 / (3 - 0)
  nblock = N / 2
  win = boxcar(nblock)
  f, Pxxf = welch(datos, fs, window=win, nfft=nblock, return_onesided=True, detrend=False)
  print(Pxxf.max())
  print(Pxxf.size)
  plt.scatter(f, Pxxf)
  plt.grid()
  plt.show()
  # 0.7456053206019085 match

if __name__ == "__main__":
  hanningSegment()
