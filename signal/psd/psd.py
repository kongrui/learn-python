import os
import numpy as np
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from scipy.signal import welch, hanning, resample
from scipy.signal.windows import boxcar
from scipy import interpolate


def brutalDFT(x):
  """
  Compute the discrete Fourier Transform of the 1D array x
  :param x: (array)

  https://www.ritchievink.com/blog/2017/04/23/understanding-the-fourier-transform-by-example/
  N = number of samples
  n = current sample
  xn = value of the sinal at time n
  k = current frequency (0 Hz to N-1 Hz)

  https://stackoverflow.com/questions/39269804/fft-normalization-with-numpy
  normalization

  https://docs.scipy.org/doc/numpy/reference/routines.fft.html#implementation-details
  Normalization
  The default normalization has the direct transforms unscaled and the inverse transforms are scaled by 1/n.
  It is possible to obtain unitary transforms by setting the keyword argument norm to "ortho" (default is None)
  so that both direct and inverse transforms will be scaled by 1/\sqrt{n}.

  """

  N = x.size
  n = np.arange(N)
  k = n.reshape((N, 1))
  e = np.exp(-2j * np.pi * k * n / N)
  return np.dot(e, x)

def dfft():

  y = np.loadtxt('data.512.csv', delimiter=",", unpack=True)
  res = np.fft.rfft(y)
  np.savetxt("rfft.csv", res, delimiter=",")
  mag = np.absolute(res)
  print ("mag.max.rui = %s" % mag.max())
  np.savetxt("rfft.mag.csv", mag, delimiter=",")

  fs = 256
  magYYYY = np.absolute(res) / fs
  print ("mag.max.yyyy - %s" % magYYYY.max())
  np.savetxt("rfft.mag.yyyy.csv", magYYYY, delimiter=",")

  # scale has be done manually, ortho is a special scaling,
  res = np.fft.rfft(y, norm="ortho") # output *= 1 / sqrt(n)
  mag = np.absolute(res)
  print ("mag.max.ortho - %s" % mag.max())
  np.savetxt("rfft.ortho.csv", mag, delimiter=",")
  # 255.248808144
  # 306.195

def dfftWhyNorm():

  # https://stackoverflow.com/questions/19975030/amplitude-of-numpys-fft-results-is-to-be-multiplied-by-sampling-period

  # https://dsp.stackexchange.com/questions/11376/why-are-magnitudes-normalised-during-synthesis-idft-not-analysis-dft
  # In most examples and FFT code that I've seen, the output (frequency magnitudes) of the forward DFT operation
  # is scaled by N -- i.e. instead of giving you the magnitude of each frequency bin, it gives you N times the magnitude.
  #
  # fft_mag = fft_mag * 2 / n

  #
  # Normalize the amplitude by number of bins and multiply by 2
  # because we removed second half of spectrum above the Nyqist frequency
  # and energy must be preserved
  # fft_mag = fft_mag * 2 / n
  #
  #
  xo, yo = np.loadtxt('data.csv', delimiter=",", unpack=True)
  f = interpolate.interp1d(xo, yo)

  fs = 64
  x = np.linspace(0, 3, num=fs*2, endpoint=False)
  y = f(x)
  res = np.fft.rfft(y)
  mag = np.absolute(res) / fs
  print ("mag.max.64=%s" % mag.max())

  fs = 128
  x = np.linspace(0, 3, num=fs*2, endpoint=False)
  y = f(x)
  res = np.fft.rfft(y) / fs
  mag = np.absolute(res)
  print ("mag.max.128=%s" % mag.max())

  # in your code, check if there is a norm parameter to turn off 1/N
  # since scale is off by default, later on, we need to scale M and half

def dfftWithSampling():
  fs = 64
  xo, yo = np.loadtxt('data.csv', delimiter=",", unpack=True)
  f = interpolate.interp1d(xo, yo)
  x = np.linspace(0, 3, num=fs*2, endpoint=False)
  y = f(x)
  np.savetxt("sampling.256.csv", zip(x,y), delimiter=",")
  print(y.size)
  res = np.fft.rfft(y)
  np.savetxt("rfft.256.csv", res, delimiter=",")
  # 255.248808144
  print (res.size)
  mag = np.absolute(res)
  print ("mag.max.rui=%s" % mag.max())
  np.savetxt("rfft.mag.256.csv", mag, delimiter=",")
  conjugate = np.square(mag)
  np.savetxt("rfft.conjugate.256.csv", conjugate, delimiter=",")
  magYYYY = np.absolute(res) / fs
  np.savetxt("rfft.mag.yyyy.256.csv", magYYYY, delimiter=",")
  print ("mag.max.yyyy=%s" % magYYYY.max())

def singleSegment():
  y = np.loadtxt('data.512.csv', delimiter=",", unpack=True)
  datos = y
  print(datos.size)
  N = 512
  fs = N * 1.0 / (3 - 0)
  nblock = N
  win = boxcar(nblock)
  f, Pxxf = welch(datos, fs, window=win, nfft=nblock, noverlap=0, return_onesided=True, detrend=False)
  print(Pxxf.max())
  print(Pxxf.size)
  plt.scatter(f, Pxxf)
  plt.grid()
  plt.show()
  # 1.49 same
  # 1.4912098859928722
  # scale = 1.0 / (fs * (win*win).sum())  (t2-t1)/N^2  3/262144

def doubleSegment():
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

def hanningSegment():
  # does not work
  y = np.loadtxt('data.512.csv', delimiter=",", unpack=True)
  datos = y
  print(datos.size)
  N = 512
  fs = N * 1.0 / (3 - 0)
  nblock = N
  win = hanning(nblock)
  f, Pxxf = welch(datos, fs, window=win, nfft=nblock, return_onesided=True, detrend=False)
  print(Pxxf.max())
  print(Pxxf.size)
  plt.scatter(f, Pxxf)
  plt.grid()
  plt.show()

def runPSD2():
  x = np.linspace(0, 10, 100001)
  dt = x[1] - x[0]
  fs = 1 / dt
  a1 = 1
  f1 = 500
  a2 = 10
  f2 = 2000
  y = a1 * np.sin(2 * np.pi * f1 * x) + a2 * np.sin(2 * np.pi * f2 * x)
  datos = y
  nblock = 2048
  overlap = 128
  win = hanning(nblock, True)
  print win.size
  f, Pxxf = welch(datos, fs, window=win, noverlap=overlap, nfft=nblock, return_onesided=True, detrend=False)
  print(Pxxf.max())
  plt.semilogy(f, Pxxf, '-o')
  plt.grid()
  plt.show()
  # 1024 3.2384
  # 2048 5.54013431789

def runPSD():
  WORK_DIR = os.path.dirname(os.path.realpath(__file__))
  print(WORK_DIR)
  xo, yo = np.loadtxt(WORK_DIR + '/data.csv', delimiter=",", unpack=True)
  x = np.linspace(0, 3, 512)
  dt = x[1] - x[0]
  fs = 1 / dt
  print fs
  fs = 1
  datos = yo
  print(datos.size)
  nblock = 256
  #overlap = 128
  #win = hanning(nblock, True)
  win = boxcar(nblock)
  f, Pxxf = welch(datos, fs, window=win, nfft=nblock, return_onesided=True, detrend=False)
  print(Pxxf.max())
  print(Pxxf.size)
  #plt.semilogy(f, Pxxf, '-o')
  plt.scatter(f, Pxxf)
  plt.grid()
  plt.show()
  # win=256 0.472064843756
  # win=128 0.254662254953

def runWin():
  window = boxcar(51)
  plt.plot(window)
  plt.title("Boxcar window")
  plt.ylabel("Amplitude")
  plt.xlabel("Sample")
  plt.show()

def windowProduct():
  # scale = 1.0 / (fs * (win*win).sum())
  win1 = np.array([1,1,1,1])
  win2 = np.array([1,1,1,1])
  scale = (win1 * win2).sum()
  print(scale)

def main():
  print("Hello World!")

if __name__ == "__main__":
  #runPSD()
  #dfft()
  dfftWhyNorm()
  #doubleSegment()
  #hanningSegment()
  #singleSegment()
  #doubleSegment()
  #dfftWithSampling()
  #windowProduct()
