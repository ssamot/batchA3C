import pylab

from numpy import genfromtxt
import numpy as np
my_data = genfromtxt('Qs.txt', delimiter=' ')

print my_data.shape
data = my_data[:,1]

# def movingaverage(interval, window_size):
#     window= np.ones(int(window_size))/float(window_size)
#     return np.convolve(interval, window, 'same')
#
#
# window = 200
#
# data = movingaverage(data,window)
#
# pylab.plot(range(len(data)-window), data[:-window], label = "V")

pylab.plot(range(len(data)), data[:], label = "V")

pylab.legend()
pylab.title("Title of Plot")
pylab.xlabel("X Axis Label")
pylab.ylabel("Y Axis Label")

pylab.show()