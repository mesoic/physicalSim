# ---------------------------------------------------------------------------------
# 	physicsUtilities -> asyncFactory.py
#	Copyright (C) 2020 Michael Winters
#	github: https://github.com/mesoic
#	email:  mesoic@protonmail.com
# ---------------------------------------------------------------------------------
#	
#	Permission is hereby granted, free of charge, to any person obtaining a copy
#	of this software and associated documentation files (the "Software"), to deal
#	in the Software without restriction, including without limitation the rights
#	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#	copies of the Software, and to permit persons to whom the Software is
#	furnished to do so, subject to the following conditions:
#	
#	The above copyright notice and this permission notice shall be included in all
#	copies or substantial portions of the Software.
#	
#	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#	SOFTWARE.
#

#!/usr/bin/env python
import numpy as np

# For asyncfactory
import multiprocessing as mp

# Generic multiprocess class
class asyncFactory:
	
	# Initialize with function and callback
	def __init__(self):
		
		# Initialize multiprocess pool
		self.pool = mp.Pool()

	# async: call method
	def call(self, func, callback, *args, **kwargs):

		self.pool.apply_async(func, args, kwargs, callback)

	# async: wait method
	def wait(self):

		self.pool.close()
		self.pool.join()


# Scipy cookbook signal processing (smooth)
def smooth(_x, window_len=11, window='hanning'):
  
	# Cast input as an np.array 
	x = np.array(_x)

	# Check dimension
	if x.ndim != 1:
		
		raise ValueError("smooth only accepts 1 dimension arrays.")

	# Check window size
	if x.size < window_len:
		
		raise ValueError("Input vector needs to be bigger than window size.")

	# Check minimum window length
	if window_len < 3:
		
		return x

	# Check if window is valid type
	if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
		
		raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

	# Row wize merging
	s = np.r_[ x[window_len-1 : 0 :-1], x, x[-2: -window_len-1 : -1] ]

	#print(s)
	#rint(len(s))
	if window == 'flat': #moving average

		w = np.ones(window_len,'d')

	# Prepare window
	else:

		w = eval('np.' + window + '(window_len)')

	# Perfrom convolution and return
	y = np.convolve( w / w.sum(), s, mode='same')

	return y[ (window_len - 1): -(window_len - 1) ]


# Method tor return a histogram curve
def histogram_curve(data, bins=20, normed=True):

	yhist, binedges = np.histogram( data, bins=bins, normed=normed )

	bincenters = np.mean(np.vstack( [binedges[0:-1],binedges[1:]] ), axis=0)

	return bincenters, yhist