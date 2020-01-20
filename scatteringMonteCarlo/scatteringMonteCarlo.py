# ---------------------------------------------------------------------------------
# 	scatteringMonteCarlo -> scatteringMonteCarlo.py
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

# Import scattering rates object
from physicsUtilities.materialScatteringRates import materialScatteringRates

# Import simulation local utilities
from scatteringEventProcessor import solidStateElectron
from scatteringEventProcessor import cylindricalWavevector
from scatteringEventProcessor import scatteringEventProcessor

# A class to simulate velocity saturation with intervalley scattering. 
# via Monte Carlo methods.
class scatteringMonteCarlo:

	# We want to 
	def __init__(self, config):

		# Random number generator
		self.random = random.SystemRandom()

		# Store sinulation configuration data
		self.material = config["material"]
		self.energy   = config["energy"]
		self.field	  = config["field"]
		self.events   = config["events"] 

		# Initialize solid state electron object
		self.electron = solidStateElectron( self.material, "G" )

		# Calculate scattering rates for phonon processes
		self.rates = materialScatteringRates( self.energy, self.material )

		# Build scattering event processor for calculated rates
		self.Processor = scatteringEventProcessor( self.rates )

	# This method will randomize the initial state of the electon	
	def randomizeInitial(self, Emax = 0.05):
		
		r = self.random.random()

		# Initialize scattering event processor 
		self.Processor.isotropicScatteringEvent(self.electron, Emax*r, "G")

		# Dictionary to store results
		self.result = {
			"time" 		: [0.0],
			"valley"	: [self.electron.valley],
			"energy"	: [self.electron.E],
			#"wavevector": [self.electron.K],
			"velocity"	: [self.electron.v],
			"field"		: self.field
		}

	# Apply electric field to electron for simulated flight time (tau)
	def applyElectricField(self):

		# Generate a flight time for our electon 
		tau  = self.Processor.generateFlightTime(self.electron)
		time = self.result["time"][-1] + tau
		self.result["time"].append(time)

		# Calculate the change in components wavevector 
		# due to acceleration in an electric field
		dKz = ( -self.field * tau) / self.material.hbar
		dKr = 0.0

		# Initialize wavevector
		dK  = cylindricalWavevector(dKz, dKr)

		# Update electron state
		self.Processor.accelerationEvent(self.electron, dK)

	# Run the simulation
	def run(self):

		# Interate over number of scattering events
		for _ in range( int(self.events) - 1 ):

			# Apply electric field to electron
			self.applyElectricField()

			# Simulate scattering event
			self.Processor.generateScatteringEvent(self.electron)

			# Update energy vector
			self.result["valley"].append(self.electron.valley)
			self.result["energy"].append(self.electron.E)	
			self.result["velocity"].append(self.electron.v)
			#self.result["wavevector"].append(self.electron.K)
