# ---------------------------------------------------------------------------------
# 	physicsUtilities -> postprocessVelocityField.py
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

# So we can access physicsUtilities directory
import sys
sys.path.insert(1, '..')

# Import matplotlib
import matplotlib.pyplot as plt

# Import physical and material constants
from physicsUtilities.materialConstants import GaAs

# Routines to postprocess velocity field simulation data
if __name__ == "__main__":

	# Read back data object
	path = "./data/GaAs-20kV.dat"
	data = p.load( open( path, "rb" ) )

	print( data.keys() )

	# Calculate mean velocity
	key = "velocity"

	sim_data = []	
	for key in data["Simulation.result"].keys():

		sim_data.append( -1.0 * np.mean( data["Simulation.result"][key]["velocity"] ) ) 

	# Plot results	
	fig = plt.figure()
	ax0 = fig.add_subplot(111)
	#ax1 = ax0.twinx()

	h0, = ax0.plot(data["config"]["field"] / 1e3, sim_data, "tab:blue" )	
	ax0.set_title("GaAs Velocity Field Simulation")
	ax0.set_xlabel("Electric Field $(V/cm)$")
	ax0.set_ylabel("Electron Velocity $(cm/s)$")

	# Show simulation results
	plt.show()