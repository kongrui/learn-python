import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import numpy as np

f = 10  # Frequency, in cycles per second, or Hertz
f_s = 100  # Sampling rate, or number of measurements per second

t = np.linspace(0, 2, 2 * f_s, endpoint=False)
print(t.size)
print(t)
x = np.sin(f * 2 * np.pi * t)
#print(x)

#fig, ax = plt.subplots()
#ax.plot(t, x)
#ax.set_xlabel('Time [s]')
#ax.set_ylabel('Signal amplitude')

from scipy import fftpack

X = fftpack.fft(x)
freqs = fftpack.fftfreq(len(x)) * f_s
print(freqs)

fig, ax = plt.subplots()

mag = np.abs(X)
print(mag)

ax.stem(freqs, mag)
ax.set_xlabel('Frequency in Hertz [Hz]')
ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
ax.set_xlim(-f_s / 2, f_s / 2)
ax.set_ylim(-5, 110)

plt.show()