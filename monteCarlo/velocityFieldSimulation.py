# ---------------------------------------------------------------------------------
# 	physicsUtilities -> velocityFieldSimulation.py
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
import random

# So we can access physicsUtilities directory
import sys
sys.path.insert(1, '..')

# Import physical and material constants
from physicsUtilities.materialConstants import GaAs
from physicsUtilities.physicalConstants import physicalConstants

# Import scattering rates object
from physicsUtilities.materialScatteringRates import materialScatteringRates

# Import simulation local utilities
from velocityFieldUtilities import solidStateElectron
from velocityFieldUtilities import cylindricalWavevector
from velocityFieldUtilities import scatteringEventProcessor

# Matplotlib
import matplotlib.pyplot as plt

# A class to simulate velocity saturation with intervalley scattering. 
# via Monte Carlo methods.
class velocityFieldSimulation:

	# We want to 
	def __init__(self, config):

		# Random number generator
		self.random = random.SystemRandom()

		# Store sinulation configuration data
		self.material = config["material"]
		self.energy   = config["energy"]
		self.field	  = config["field"]
		self.events   = config["events"] 

		# Calculate scattering rates for phonon processes
		self.rates = materialScatteringRates( self.energy, self.material )

		# Initialize solid state electron object
		self.electron = solidStateElectron( self.material, "G" )

	# This method will randomize the initial state of the electon	
	def randomizeInitial(self, Ef = 0.1):
		
		# Initialize scattering event processor
		Processor = scatteringEventProcessor()
		Processor.isotropicScatteringEvent(self.electron, Ef, "G")

		# Dictionary to store results
		self.result = {
			"wavevector": [self.electron.K],
			"energy"	: [self.electron.E],
			"time" 		: [0.0]
		}

	# Method to simulate the time between scattering events. 
	def generateFlightTime(self):

		# Throw a random number on interval [0, 1]
		r = self.random.random()

		# Get total scattering rate for current valley
		if self.electron.valley in ["G", "Gamma"]:

			G = self.rates.getScatteringRate("Gsum")

		if self.electron.valley in ["L"]:
			
			G = self.rates.getScatteringRate("Lsum")


		# Simulate new time interval
		tau  = ( -1.0/ np.max(G) ) * np.log(r)

		# Update elapsed time vector (recursively)
		time = tau + self.result["time"][-1]

		# Return free flight time
		return tau, time

	# Apply electric field to electron for simulated flight time (tau)
	def applyElectricField(self, Processor, tau):
		
		# Calculate the change in components wavevector 
		# due to acceleration in an electric field
		dKz = ( -self.field * tau) / self.material.hbar
		dKr = 0.0

		# Initialize wavevector
		dK  = cylindricalWavevector(dKz, dKr)

		# Update electron state
		Processor.accelerationEvent(self.electron, dK)

	# Run the simulation
	def runVelocityField(self):

		# Initialize scattering event processor
		Processor = scatteringEventProcessor()

		# Interate over number of scattering events
		for _ in range( int(self.events) ):

			# Generate a flight time
			tau, time  = self.generateFlightTime()

			# Apply electric field to electron
			self.applyElectricField(Processor, tau)

			# Update simulation data
			self.result["time"].append(time)
			self.result["energy"].append(self.electron.E)

# Main program
if __name__ == "__main__":

	# Define energy range to simulate
	energy = np.linspace(0.0, 1.0, 100)

	# Generate configuration dictionary
	config = {
		"material"	: GaAs(),
		"energy"	: energy,
		"field"		: 1000,
		"events"	: 1e3
	}

	# Initialize monte carlo simulation
	Simulation = velocityFieldSimulation(config)
	Simulation.randomizeInitial()
	
	# Run simulation
	Simulation.runVelocityField()

	# Plot results
	plt.plot(Simulation.result["time"], Simulation.result["energy"])
	plt.show()

