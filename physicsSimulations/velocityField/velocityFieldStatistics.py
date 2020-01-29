# ---------------------------------------------------------------------------------
# 	velocityField -> velocityFieldStatistics.py
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
import pickle as p

# Import python utils
import collections
import itertools

# Import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# So we can access physicsUtilities directory
import sys
sys.path.insert(1, '..')

# Import physical and material constants
from physicsUtilities.solidstate.materialConstants import GaAs

# Import utilites
from physicsUtilities.utilities.curveUtilities import smooth
from physicsUtilities.utilities.curveUtilities import histogram_curve


# Script to extract statistics from data files
if __name__ == "__main__":

	# Set simulation name
	simulation_name = "GaAs-20kV"

	# List of simulation datafiles to postprocess
	paths = [ "./data/simulation/%s.%s"%(simulation_name, int(_)) for _ in range(25) ]


	# List of fields to generate statistics for
	fields = [1e3, 5e3, 10e3, 20e3]

	# list of keys of plot
	data_keys = ["velocity", "energy"]

	# List of fields we want to examine statistics for
	statistics = collections.OrderedDict( {_f : { _ : [] for _ in data_keys } for _f in fields } )
		
	# Track the total number of simulaton events
	events = 0

	for _path in paths:

		# Processing path
		print("Processing path: %s"%_path)

		data = p.load( open( _path, "rb" ) )

		# Events 
		events += data["config"]["events"]

		# For each field we want to have statistics for
		for _f in statistics.keys():

			for data_key in data_keys:

				# Look for closet field in simulation data
				index = np.argmax( data["config"]["field"]  >= _f )

				# Extract field from configuration
				field = data["config"]["field"][index]	

				# Extract data from simulation (safe concatenate)
				statistics[_f][data_key] = [_ for _ in itertools.chain( statistics[_f][data_key], data["Simulation.result"][field][data_key]) ]


	# Plot the data	(velocity)
	_smooth = 15

	fig = plt.figure()
	ax0 = fig.add_subplot(111)

	hlist = []
	for _f in statistics.keys():

		bins, hist = histogram_curve( statistics[_f]["velocity"] , bins=300, normed=False )

		h, = ax0.plot(bins, smooth(hist, _smooth) )

		hlist.append(h)

	ax0.set_xlabel("Electron Velocity (cm/s)")
	ax0.set_ylabel("Counts")
	ax0.set_title("GaAs Electron Velocity Statistics : n=%.1e events"%events)
	ax0.legend(hlist, ["%s $(kV/cm)$"%(_f/1e3) for _f in fields])

	# Plot the data		
	fig = plt.figure()
	ax0 = fig.add_subplot(111)

	hlist = []
	for _f in statistics.keys():

		bins, hist = histogram_curve( statistics[_f]["energy"] , bins=300, normed=False )

		h, = ax0.plot(bins, smooth(hist, _smooth) )

		hlist.append(h)

	ax0.set_xlabel("Electron Energy (eV)")
	ax0.set_ylabel("Counts")
	ax0.set_title("GaAs Electron Energy Statistics : n=%.1e events"%events)
	ax0.legend(hlist, ["%s $(kV/cm)$"%(_f/1e3) for _f in fields])
	plt.show()