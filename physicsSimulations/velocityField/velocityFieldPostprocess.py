# ---------------------------------------------------------------------------------
# 	velocityField -> velocityFieldPostprocess.py
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
from physicsUtilities.solidstate.materialConstants import GaAs

# Routines to postprocess velocity field simulation data. This routine 
# averages velocity for each simulation and produces a composite average 
# over all simulations.
if __name__ == "__main__":

	# List of simulation datafiles to postprocess
	if False:

		# Set simulation name		
		simulation_name = "GaAs-20kV"

		# Build simulation paths
		paths = [ "./data/simulation/%s.%s"%(simulation_name, int(_)) for _ in range(3) ]
		
		# Path to output file
		postprocess_path = "./data/postprocess/%s.dat"%simulation_name

	# For working with single files
	else:	

		paths = ["./data/simulation/GaAs-20kV.4"]
		postprocess_path = "./data/postprocess/tmp.dat"

	# Dictionary to hold processed data
	postprocess_data = {}

	# Read back data object from each simulation datafile	
	for _path in paths:
	
		# Processing path
		print("Processing path: %s"%_path)

		data = p.load( open( _path, "rb" ) )

		# Dictionary to store postprocessed data
		postprocess = { 
			"field" 	: [], 
			"energy"	: [], 
			"velocity"	: [], 
			"valley"	: {"G" : [], "L" : []}, 
			"events"	: 0 
		} 

		# Store electric field 
		postprocess["field"] = data["config"]["field"]

		# Store the number of events
		postprocess["events"] = data["config"]["events"]

		# Loop through simulation data
		for field in data["config"]["field"]:

			# Electron energy
			postprocess["energy"].append( -1.0 * np.mean( data["Simulation.result"][field]["velocity"] ) ) 

			# Electron drift velocity
			postprocess["velocity"].append( -1.0 * np.mean( data["Simulation.result"][field]["velocity"] ) ) 

			# Calculate valley occupancy ratio	
			Gsum, Lsum = 0.0, 0.0

			# Loop through all fields
			for _index, _data in enumerate( 

				zip( data["Simulation.result"][field]["time"], data["Simulation.result"][field]["valley"] ) ):

				# Lookback protection
				if _index > 0:

					# Calculate occupancy time
					tau = _data[0] - data["Simulation.result"][field]["time"][_index - 1]

					if data["Simulation.result"][field]["valley"][_index] == "G":

						Gsum += tau 

					if data["Simulation.result"][field]["valley"][_index] == "L":

						Lsum += tau
		
			# Append occupancy times to postprocess data
			postprocess["valley"]["G"].append( Gsum )
			postprocess["valley"]["L"].append( Lsum )

			# Store number of events
			postprocess["events"] = len(data["Simulation.result"][field]["time"])

			# Append data to dictionary
			postprocess_data[_path] = postprocess


	# Produce the composite average of simulations
	# Initialization lambda returns a zero valued list
	composite_init = lambda: np.zeros( len(data["config"]["field"]) )

	# Length of composite
	composite_len = len(postprocess_data.keys())

	# Composite data structure
	composite = {
	
		"field" 	: data["config"]["field"], 
		"energy"	: composite_init(), 		
		"velocity"	: composite_init(), 
		"valley"	: {"G" : composite_init(), "L" : composite_init()}, 
		"events"	: 0 
	}

	for simulation in postprocess_data.keys():

		# Sum velocity
		composite["velocity"] += postprocess_data[simulation]["velocity"]

		# Sum energy
		composite["energy"] += postprocess_data[simulation]["energy"]

		# Sum valley scattering times
		composite["valley"]["G"] += postprocess_data[simulation]["valley"]["G"]
		composite["valley"]["L"] += postprocess_data[simulation]["valley"]["L"]
		
		# Sum events
		composite["events"] += postprocess_data[simulation]["events"]

	# Velocity
	composite["velocity"] = composite["velocity"] / composite_len

	# Energy
	composite["energy"] = composite["energy"] / composite_len

	# Valley occupancy
	composite["valley"]["G"] = composite["valley"]["G"] / composite_len
	composite["valley"]["L"] = composite["valley"]["L"] / composite_len

	# Dump the results		
	p.dump( {"composite" : composite ,"data" : postprocess_data}, open(postprocess_path, "wb") )
