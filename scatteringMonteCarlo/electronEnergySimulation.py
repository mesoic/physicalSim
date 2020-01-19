# ---------------------------------------------------------------------------------
# 	physicsUtilities -> electronEnergySimulation.py
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

# So we can access physicsUtilities directory
import sys
sys.path.insert(1, '..')

# Import physical and material constants
from physicsUtilities.materialConstants import GaAs

# Import Monte Carlo simulation
from scatteringMonteCarlo import scatteringMonteCarlo

# Matplotlib
import matplotlib.pyplot as plt

if __name__ == "__main__":

	# Define energy range to simulate
	energy = np.linspace(0.0, 2.0, 1000)

	# Generate configuration dictionary
	config = {
		"material"	: GaAs(),
		"energy"	: energy,
		"field"		: 20000,
		"events"	: 1000
	}

	# Initialize monte carlo simulation
	Simulation = scatteringMonteCarlo(config)
	Simulation.randomizeInitial()

	# Run simulation
	Simulation.run()

	# Want to visualize the valley occupancy
	Gdata = np.empty( config["events"] )
	Ldata = np.empty( config["events"] )
	Tdata = np.empty( config["events"] )

	# Initialize to Nan
	Gdata[:] = np.nan
	Ldata[:] = np.nan
	Tdata[:] = np.nan

	# Parameter key
	data_key = "energy"

	# loop through valley occupancy
	for i,V in enumerate(Simulation.result["valley"]):

		if V == "G":
			Gdata[i] = Simulation.result[ data_key ][i] 

		if V == "L":
			Ldata[i] = Simulation.result[ data_key ][i] 

		try: 
			if ( ( V == "G" and Simulation.result["valley"][i-1] == "L" ) or 
			 	( V == "L" and Simulation.result["valley"][i-1] == "G" ) ):

			 	Tdata[i-1] = Simulation.result[data_key][i-1]
			 	Tdata[i] = Simulation.result[data_key][i]
		except: 
			pass

	#print(np.mean(Simulation.result["energy"] ))	
	#print(np.mean(k.kz for k in Simulation.result["wavevector"]) )	

	# Plot results
	fig = plt.figure()
	ax0 = fig.add_subplot(111)
	h0, = ax0.plot( Simulation.result["time"], Gdata, ".")
	h1, = ax0.plot( Simulation.result["time"], Ldata, ".")
	ax0.plot( Simulation.result["time"], Tdata, "grey")
	ax0.set_xlabel("Time $(s)$")
	ax0.set_ylabel("Energy $(eV)$")
	ax0.set_title("GaAs Electron Energy Simulation : |E| = 10kV/cm")
	ax0.legend([h0,h1],["$\Gamma$ valley", "L valley"])
	plt.show()
