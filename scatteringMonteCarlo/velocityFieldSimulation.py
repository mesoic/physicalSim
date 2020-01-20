# ---------------------------------------------------------------------------------
# 	scatteringMonteCarlo -> velocityFieldSimulation.py
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

# Import ordered dict
# So we can access physicsUtilities directory
import sys
sys.path.insert(1, '..')

# Import physical and material constants
from physicsUtilities.materialConstants import GaAs
from physicsUtilities.genericUtilities import asyncFactory

# Import Monte Carlo simulation
from scatteringMonteCarlo import scatteringMonteCarlo

# Simulate electron velocity vs. electric field
class velocityFieldSimulation:

	def __init__(self, config):

		# Configuration data
		self.config = config

		# Dictionary to store simuilation results
		self.result = {}
	
	# Simulation run method 
	def run(self):

		factory = asyncFactory()
		
		for _f in self.config["field"]: 

			factory.call(self.simulate_field, self.log_result, _f)

		factory.wait()

	def simulate_field(self, field):
	
		# Confirmation
		print("Simulating: %s"%field)

		# Generate configuration dictionary
		config = {
			"material"	: self.config["material"],
			"energy"	: self.config["energy"],
			"events"	: self.config["events"],
			"field"		: field
		}

		# Initialize monte carlo simulation
		Simulation = scatteringMonteCarlo(config)
		Simulation.randomizeInitial()

		# Run simulation
		Simulation.run()

		# Return simulation result
		return Simulation.result

	def log_result(self, sim_result):
		
		self.result[ sim_result["field"] ] = sim_result


if __name__ == "__main__":

	# Generate configuration dictionary for simulation
	config = {
		"material"	: GaAs(),
		"energy"	: np.linspace(0.0, 2.0, 1000),
		"field"		: np.linspace(300, 2e4, 100),
		"events"	: 1000
	}


	# Initialize simulation
	Simulation = velocityFieldSimulation(config)
	Simulation.run()

	# Serialize the simulation results for post processing
	path = "./data/simulation/GaAs-20kV.3"
	p.dump( {"config": config, "Simulation.result" : Simulation.result } , open(path, "wb") )
