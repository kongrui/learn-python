import os
import numpy as np
import matplotlib
matplotlib.use('agg')
matplotlib.use('Qt5Agg')
#matplotlib.use("tkagg")
#matplotlib.use('TkAgg')
import math

from scipy.signal import hilbert, chirp

import matplotlib.pyplot as plt
plt.style.use('ggplot')
from scipy.signal import welch, hanning, resample
from scipy.signal.windows import boxcar
from scipy import interpolate

from numpy import fft

def fourierExtrapolation(x, n_predict, n_harm=10):
  n = x.size
  # number of harmonics in model
  t = np.arange(0, n)
  print t
  p = np.polyfit(t, x, 1)  # find linear trend in x
  print p
  x_notrend = x - p[0] * t  # detrended x
  x_freqdom = fft.fft(x_notrend)  # detrended x in frequency domain
  f = fft.fftfreq(n)  # frequencies
  indexes = range(n)
  # sort indexes by frequency, lower -> higher
  indexes.sort(key=lambda i: np.absolute(f[i]))

  t = np.arange(0, n + n_predict)
  restored_sig = np.zeros(t.size)
  for i in indexes[:1 + n_harm * 2]:
    ampli = np.absolute(x_freqdom[i]) / n  # amplitude
    phase = np.angle(x_freqdom[i])  # phase
    restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
  return restored_sig + p[0] * t

def main():
  x = np.array(
    [669, 592, 664, 1005, 699, 401, 646, 472, 598, 681, 1126, 1260, 562, 491, 714, 530, 521, 687, 776, 802, 499, 536,
     871, 801, 965, 768, 381, 497, 458, 699, 549, 427, 358, 219, 635, 756, 775, 969, 598, 630, 649, 722, 835, 812, 724,
     966, 778, 584, 697, 737, 777, 1059, 1218, 848, 713, 884, 879, 1056, 1273, 1848, 780, 1206, 1404, 1444, 1412, 1493,
     1576, 1178, 836, 1087, 1101, 1082, 775, 698, 620, 651, 731, 906, 958, 1039, 1105, 620, 576, 707, 888, 1052, 1072,
     1357, 768, 986, 816, 889, 973, 983, 1351, 1266, 1053, 1879, 2085, 2419, 1880, 2045, 2212, 1491, 1378, 1524, 1231,
     1577, 2459, 1848, 1506, 1589, 1386, 1111, 1180, 1075, 1595, 1309, 2092, 1846, 2321, 2036, 3587, 1637, 1416, 1432,
     1110, 1135, 1233, 1439, 894, 628, 967, 1176, 1069, 1193, 1771, 1199, 888, 1155, 1254, 1403, 1502, 1692, 1187, 1110,
     1382, 1808, 2039, 1810, 1819, 1408, 803, 1568, 1227, 1270, 1268, 1535, 873, 1006, 1328, 1733, 1352, 1906, 2029,
     1734, 1314, 1810, 1540, 1958, 1420, 1530, 1126, 721, 771, 874, 997, 1186, 1415, 973, 1146, 1147, 1079, 3854, 3407,
     2257, 1200, 734, 1051, 1030, 1370, 2422, 1531, 1062, 530, 1030, 1061, 1249, 2080, 2251, 1190, 756, 1161, 1053,
     1063, 932, 1604, 1130, 744, 930, 948, 1107, 1161, 1194, 1366, 1155, 785, 602, 903, 1142, 1410, 1256, 742, 985,
     1037, 1067, 1196, 1412, 1127, 779, 911, 989, 946, 888, 1349, 1124, 761, 994, 1068, 971, 1157, 1558, 1223, 782,
     2790, 1835, 1444, 1098, 1399, 1255, 950, 1110, 1345, 1224, 1092, 1446, 1210, 1122, 1259, 1181, 1035, 1325, 1481,
     1278, 769, 911, 876, 877, 950, 1383, 980, 705, 888, 877, 638, 1065, 1142, 1090, 1316, 1270, 1048, 1256, 1009, 1175,
     1176, 870, 856, 860])
  n_predict = 100
  extrapolation = fourierExtrapolation(x, n_predict)
  plt.plot(np.arange(0, extrapolation.size), extrapolation, 'r', label='extrapolation')
  plt.plot(np.arange(0, x.size), x, 'b', label='x', linewidth=3)
  plt.legend()
  plt.show()

def run4Eval():
  f = np.loadtxt('data.in.csv', delimiter="|", usecols=(0))
  x = np.loadtxt("data.in.csv", delimiter="|", usecols=(1), converters={1: lambda s: eval(s)}).view(complex).reshape(-1)
  t1, v1 =  np.loadtxt('data.out.wo.baseband.csv', usecols=(0, 1), delimiter=',', unpack=True)
  t2, v2 =  np.loadtxt('data.out.baseband.csv', usecols=(0, 1), delimiter=',', unpack=True)
  plt.legend()
  plt.show()
  print x.size
  t = np.fft.ifft(x)
  #f = interpolate.interp1d(t, yo)
  #x = np.linspace(0, 3, num=fs*2, endpoint=False)
  #y = f(x)
  extrapolation = fourierExtrapolation(x, 10, 2)
  plt.plot(np.arange(0, extrapolation.size), extrapolation, 'r', label='extrapolation')
  plt.plot(np.arange(0, x.size), x, 'b', label='x', linewidth=3)
  plt.plot(t1, v1, label='1', linewidth=5)
  plt.plot(t2, v2, label='2', linewidth=1)
  plt.legend()
  plt.show()

def toTimeDomainSlow():
  v = np.loadtxt("data.in.csv", delimiter="|", usecols=(1)
                 , converters={1: lambda s: eval(s)}).view(complex).reshape(-1)
  f = np.loadtxt('data.in.csv', delimiter="|", usecols=(0))
  for t in [0,2,4,6,8,10]:
    res = 0.0
    for fv, vv in zip(f, v):
      r = np.real(vv)
      i = np.imag(vv)
      theta1 = 2 * np.pi * t * fv
      vvv1 = r * math.cos(theta1) - i * math.sin(theta1)
      theta2 = - 2 * np.pi * t * fv
      vvv2 = r * math.cos(theta2) + i * math.sin(theta2)
      #vvv = math.fabs(r * math.cos(theta) - i * math.sin(theta))
      vvvv = math.fabs(vvv1)
      res += vvvv
      #print("%f %f %f %f %f" % (fv, theta, vvv, r, i))
    print(t)
    print(res)

#0.000,2.907
#2.000,-25.46E-3
#4.000,-65.70E-3
#6.000,34.86E-3
#8.000,-8.087E-3
#10.00,-13.43E-3
def toTimeDomainSlow02():
  v = np.loadtxt("data.in.csv", delimiter="|", usecols=(1)
                 , converters={1: lambda s: eval(s)}).view(complex).reshape(-1)
  def sf(x):
    return complex(x.real - x.imag, x.imag - x.real)
  sffunc = np.vectorize(sf)
  f = np.loadtxt('data.in.csv', delimiter="|", usecols=(0))
  for t in [0,2,4,6,8,10]:
    res = 0.0
    for fv, vv in zip(f, v):
      r = np.real(vv)
      i = np.imag(vv)
      theta1 = 2 * np.pi * t * fv
      vvvv = r * math.cos(theta1) - i * math.sin(theta1)
      res += vvvv
      #print("%f %f %f %f %f" % (fv, theta, vvv, r, i))
    #print(res.real)
    res = sf(res)
    print(res.real)
    #print(np.abs(res))


# toTimeDomain(yvals, freqs, from, to, step)
# idx = 0
# for t in from, to with step:
#   ts[idx] = 0
#   for (i = 0; i < size; i++)
#     theta = 2 * 3.1415926 * t * freqs;
#     ts[idx] = ts[idx] +  yval * cos(theta) - yival * sin(theta);
#   idx++
#   M * N = N
def toTimeDomain():
  v = np.loadtxt("data.in.csv", delimiter="|", usecols=(1)
                 , converters={1: lambda s: eval(s)}).view(complex).reshape(-1)
  f = np.loadtxt('data.in.csv', delimiter="|", usecols=(0))
  t = np.linspace(0, 10, 6, endpoint=True)
  tT = t.reshape(6, 1)
  tTf = tT * f
  print tTf.shape
  M = np.exp(2j * np.pi * tTf)
  #print(M)
  res = np.dot(M, v)
  print(res.real)


def toTimeDomainShift():
  v = np.loadtxt("data.in.csv", delimiter="|", usecols=(1)
                 , converters={1: lambda s: eval(s)}).view(complex).reshape(-1)
  f = np.loadtxt('data.in.csv', delimiter="|", usecols=(0))
  f = f - f[f.size/2]
  t = np.linspace(0, 10, 6, endpoint=True)
  tT = t.reshape(6, 1)
  tTf = tT * f
  print tTf.shape
  M = np.exp(2j * np.pi * tTf)
  #print(M)
  res = np.dot(M, v)
  print(res.real)
  print(np.abs(res))

def toTimeDomainHalf():
  v = np.loadtxt("data.in.csv", delimiter="|", usecols=(1)
                 , converters={1: lambda s: eval(s)}).view(complex).reshape(-1)
  f = np.loadtxt('data.in.csv', delimiter="|", usecols=(0))
  v = v[0:v.size/2]
  f = f[0:f.size/2]
  t = np.linspace(0, 10, 6, endpoint=True)
  tT = t.reshape(6, 1)
  tTf = tT * f
  print tTf.shape
  M = np.exp(2j * np.pi * tTf)
  #print(M)
  res = np.dot(M, v)
  print(res.real)
  print(np.abs(res))

def toTimeDomainBB():
  v = np.loadtxt("data.in.csv", delimiter="|", usecols=(1)
                 , converters={1: lambda s: eval(s)}).view(complex).reshape(-1)
  f = np.loadtxt('data.in.csv', delimiter="|", usecols=(0))
  t = np.linspace(0, 10, 6, endpoint=True)
  tT = t.reshape(6, 1)
  tTf = tT * f
  print tTf.shape
  M = np.exp(2j * np.pi * tTf)
  #print(M)
  res = np.dot(M, v)
  print(np.abs(res))

def toTimeDomainDouble():
  v = np.loadtxt("data.in.csv", delimiter="|", usecols=(1)
                 , converters={1: lambda s: eval(s)}).view(complex).reshape(-1)
  f = np.loadtxt('data.in.csv', delimiter="|", usecols=(0))
  t = np.linspace(0, 10, 6*2, endpoint=True)
  tT = t.reshape(6*2, 1)
  tTf = tT * f
  print tTf.shape
  M = np.exp(2j * np.pi * tTf)
  #print(M)
  res = np.dot(M, v)
  print(np.real)
  print(np.absolute(res))

def toTimeDomainSym():
  v = np.loadtxt("data.in.csv", delimiter="|", usecols=(1)
                 , converters={1: lambda s: eval(s)}).view(complex).reshape(-1)
  #nv = np.conjugate(v)
  #v = np.concatenate([v,nv])
  f = np.loadtxt('data.in.csv', delimiter="|", usecols=(0))
  #nf = f * -1
  #f = np.concatenate([f,nf])
  t = np.linspace(-5, 5, 6, endpoint=True)
  tT = t.reshape(6, 1)
  tTf = tT * f
  print tTf.shape
  M = np.exp(2j * np.pi * tTf)
  #print(M)
  res = np.dot(M, v)
  print(res.real)
  print(np.absolute(res))
  print(np.absolute(res) / 2)


def toFeqDomain():
  x = np.loadtxt("data.out.baseband.csv", delimiter=",", usecols=(1), converters={1: lambda s: eval(s)}, unpack=True)
  N = x.size
  n = np.arange(N)
  k = n.reshape((N, 1))
  e = np.exp(-2j * np.pi * k * n / N)
  res = np.dot(e, x)
  print res

def DFTSlow():
  x = np.random.random(1000)
  x = np.asarray(x, dtype=float)
  N = x.shape[0]
  n = np.arange(N)
  k = n.reshape((N, 1))[0:9]
  print(k)
  M = np.exp(-2j * np.pi * k * n / N)
  print(M)
  res = np.dot(M, x)
  print res

def runHilbert():
  v = np.loadtxt("data.in.csv", delimiter="|", usecols=(1)
                 , converters={1: lambda s: eval(s)}).view(complex).reshape(-1)
  f = np.loadtxt('data.in.csv', delimiter="|", usecols=(0))
  analytic_signal = hilbert(v.real)
  amplitude_envelope = np.abs(analytic_signal)
  print amplitude_envelope

if __name__ == "__main__":
  #run4Eval()
  #main()
  #toTimeDomain()
  #toFeqDomain()
  #DFTSlow()
  #toTimeDomainHalf()
  #toTimeDomainShift()
  #toTimeDomainBB()
  #toTimeDomainSym()
  #toTimeDomainDouble()
  #toTimeDomainSlow02()
  runHilbert()
