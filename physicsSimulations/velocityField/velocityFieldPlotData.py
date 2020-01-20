# ---------------------------------------------------------------------------------
# 	velocityField -> velocityFieldPlotData.py
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

# Import matplotlib
import matplotlib.pyplot as plt 

# Import physical and material constants
from physicsUtilities.utilities.curveUtilities import smooth

# Routine to plot and analyze postprocessed velocity field data
if __name__ == "__main__":

	# Data path
	postprocess_path = "./data/postprocess/GaAs-20kV.dat"

	# Data to analyze (keys are filenames)
	postprocess_data = p.load( open( postprocess_path, "rb") )

	# Calculate differential mobility on smoothed signal
	mobility = np.gradient( smooth( postprocess_data["composite"]["velocity"] ) , postprocess_data["composite"]["field"] ) 

	# Plot electron velocity vs. electric field
	if True: 

		fig = plt.figure()
		ax0 = fig.add_subplot(111)
		ax1 = ax0.twinx()
		h0, = ax0.plot( postprocess_data["composite"]["field"] / 1e3, postprocess_data["composite"]["velocity"], "tab:blue" ) 
		h1, = ax1.plot( postprocess_data["composite"]["field"] / 1e3, mobility, "tab:orange" ) 
		ax0.set_xlabel("Electric Field $(kV/cm)$")
		ax0.set_ylabel("Electron Velocity $(cm/s)$")
		ax1.set_ylabel("Electron Mobility $(cm/Vs)$")
		ax0.legend([h0,h1],["Velocity $(v)$", "Mobility $(\mu)$"])
		ax0.set_title("GaAs Velocity Field : n=%.1e events"%postprocess_data["composite"]["events"] )
	
	# Plot valley occupancy vs. electric field strength
	if True:

		# extract the total amount of time
		time = postprocess_data["composite"]["valley"]["G"] + postprocess_data["composite"]["valley"]["L"]
		
		fig = plt.figure()
		ax0 = fig.add_subplot(111)
		h0, = ax0.plot( postprocess_data["composite"]["field"] / 1e3, postprocess_data["composite"]["valley"]["G"] / time )
		h1, = ax0.plot( postprocess_data["composite"]["field"] / 1e3, postprocess_data["composite"]["valley"]["L"] / time )
		ax0.set_xlabel("Electric Field $(kV/cm)$")
		ax0.set_ylabel("Valley Occupancy $(\%)$")
		ax0.set_title("GaAs Electron Phonon Scattering: Valley Occupancy")
		ax0.legend([h0,h1], ["$\Gamma$ valley", "L valley"])

	plt.show()
