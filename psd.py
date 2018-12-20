import os
import numpy as np
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from scipy.signal import welch, hanning, resample
from scipy.signal.windows import boxcar
from scipy import interpolate

def dfft():
  fs = 256
  #y = np.loadtxt('data.512.csv', delimiter=",", unpack=True)
  xo, yo = np.loadtxt('data.csv', delimiter=",", unpack=True)
  #win = boxcar(yo.size)
  f = interpolate.interp1d(xo, yo)
  x = np.linspace(1, 3, num=fs*2, endpoint=False)
  y = f(x)
  np.savetxt("sampling.csv", zip(x,y), delimiter=",")
  y = np.loadtxt('data.512.csv', delimiter=",", unpack=True)
  print(y.size)
  res = np.fft.rfft(y)
  np.savetxt("rfft.csv", res, delimiter=",")
  # 255.248808144
  print (res.size)
  mag = np.absolute(res)
  print (mag.max())
  np.savetxt("rfft.mag.csv", mag, delimiter=",")
  conjugate = np.square(mag)
  np.savetxt("rfft.conjugate.csv", conjugate, delimiter=",")
  magYuan = np.absolute(res) / fs
  np.savetxt("rfft.mag.yuan.csv", magYuan, delimiter=",")
  print (magYuan.max())
  # 255.248808144
  # 306.195

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

def main():
  print("Hello World!")


if __name__ == "__main__":
  #runPSD()
  #dfft()
  #doubleSegment()
  #hanningSegment()
  #singleSegment()
  doubleSegment()
