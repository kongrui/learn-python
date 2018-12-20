import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, hanning, boxcar, resample, hamming, get_window

np.random.seed(42)

# https://www.mathworks.com/help/signal/ref/pwelch.html#btulige-1

#
# n = 0:319;
# x = cos(pi/4*n)+randn(size(n));
# The signal segments are multiplied by a Hamming window 132 samples in length. The number of
# overlapped samples is not specified, so it is set to 132/2 = 66. The DFT length is 256
# points,yielding a frequency resolution of  rad/sample. Because the signal is real-valued, the
# PSD estimate is one-sided and there are 256/2+1 = 129 points. Plot the PSD as a function of
# normalized frequency.
def main():
  n = np.arange(319)
  y = np.cos(np.pi / 4 * n) + np.random.rand(n.size)
  f, pxx = welch(y, window="hamming", nperseg=71, return_onesided=True, detrend=False)
  print(pxx.size)
  plt.plot(np.arange(129), 10*np.log(pxx))
  plt.xlim([0, 130])
  plt.grid()
  plt.show()
  pass


#fs = 10;
#n_samp = 32;
#x = (0:n_samp-1) / fs;
#y = sin(2*pi*2*x);
#overlap=0;
#windowsel=hann(n_samp);
#[Pxx,f]=pwelch(y,windowsel,overlap,n_samp,fs,'onesided');
#figure();
#semilogy(f,Pxx, '-o');
def dtrend():
  fs = 10.
  n_samp = 32
  t = np.arange(n_samp) / fs
  y = np.sin(2 * np.pi * 2 * t)
  win = hanning(n_samp, False)
  f, Pxxf = welch(y, fs, window=win, noverlap=0, nfft=n_samp, return_onesided=True, detrend=False)
  plt.semilogy(f, Pxxf, '-o')
  plt.xlim([0, 5])
  plt.ylim([1e-7, 1])
  plt.show()

def dtrend2():
  x, y = np.genfromtxt('data.csv', delimiter=',', unpack=True)
  fs = 256
  res = np.fft.rfft(y, fs*2)
  print
  np.savetxt("rfft.csv", res, delimiter=",")
  window = get_window('boxcar', fs*1)
  print(window)
  print(window.sum())
  ff,yy = welch(y, fs=fs, window = window, noverlap = 0, nfft=fs*2,
                  axis=0, scaling="density", detrend=False)
  np.savetxt("spectrum.csv", yy, delimiter=",")

#MATLAB:
#
#input       = csvread('matlab_input.csv');
#fs          = 128
#win         = hamming(fs);
#[pxx,f]     = pwelch(input ,win,[],[],fs,'psd');
#csvwrite('matlab_spectrum.csv',pxx);

def dfft():
  x, y = np.genfromtxt('data.csv', delimiter=',', unpack=True)
  res = np.fft.rfft(y)
  mag = np.absolute(res)
  print (mag.max())
  np.savetxt("rfft.csv", mag, delimiter=",")

if __name__ == "__main__":
  #main()
  #dtrend()
  dfft()