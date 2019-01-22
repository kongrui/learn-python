import os
import numpy as np

from scipy.signal import welch, hanning, resample
from scipy.signal.windows import boxcar
from scipy import interpolate

from numpy import min, max
from scipy import linspace
from scipy.signal import lti, step, impulse

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt

def step_info(t, yout, pct):
    #print "OS: %f%s" % ((yout.max() / yout[-1] - 1) * 100, '%')
    #print "Tr: %fs" % (t[next(i for i in range(0, len(yout) - 1) if yout[i] > yout[-1] * .90)] - t[0])
    print "Ts: %fs" % (
                t[next(len(yout) - i for i in range(2, len(yout) - 1) if abs(yout[-i] / yout[-1]) > pct)] - t[0])

def test_step_response():

    # making transfer function
    # example from Ogata Modern Control Engineering
    # 4th edition, International Edition page 307

    # num and den, can be list or numpy array type
    num = [6.3223, 18, 12.811]
    den = [1, 6, 11.3223, 18, 12.811]

    tf = lti(num, den)

    # get t = time, s = unit-step response
    t, s = step(tf)

    # recalculate t and s to get smooth plot
    t, s = step(tf, T=linspace(min(t), t[-1], 500))

    # get i = impulse
    t, i = impulse(tf, T=linspace(min(t), t[-1], 500))
    plt.plot(t, s, t, i)
    plt.title('Transient-Response Analysis')
    plt.xlabel('Time(sec)')
    plt.ylabel('Amplitude')
    plt.hlines(1, min(t), max(t), colors='r')
    plt.hlines(0, min(t), max(t))
    plt.xlim(xmax=max(t))
    plt.legend(('Unit-Step Response', 'Unit-Impulse Response'), loc=0)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    #t, yo = np.loadtxt('data.csv', delimiter=",", unpack=True)
    #step_info(t, yo)
    #test_step_response()

    num = [6.3223, 18, 12.811]
    den = [1, 6, 11.3223, 18, 12.811]

    tf = lti(num, den)

    # get t = time, s = unit-step response
    t, s = step(tf)

    # recalculate t and s to get smooth plot
    t, s = step(tf, T=linspace(min(t), t[-1], 500))
    pct=1.05
    step_info(t, s, pct)
    yout = s
    print(len(yout))
    print ([t[i] for i in range(2, len(yout))])
    print ([i for i in range(2, len(yout) - 1) if abs(yout[-i] / yout[-1]) > pct])
    print ([len(yout) - i for i in range(2, len(yout) - 1) if abs(yout[-i] / yout[-1]) > pct])

    plt.plot(t, s)
    plt.title('Transient-Response Analysis')
    plt.xlabel('Time(sec)')
    plt.ylabel('Amplitude')
    plt.hlines(1, min(t), max(t), colors='r')
    plt.hlines(0, min(t), max(t))
    plt.xlim(xmax=max(t))
    plt.legend(('Unit-Step Response', 'Unit-Impulse Response'), loc=0)
    plt.grid()
    plt.show()
