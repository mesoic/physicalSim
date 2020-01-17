


#!/usr/bin/env python 
import numpy as np
import random

# So we can access physicsUtilities directory
import sys
sys.path.insert(1, '..')

# Import physical and material constants
from physicsUtilities.materialConstants import GaAs
from physicsUtilities.materialScatteringRates import materialScatteringRates


class monteCarlo:

	# We want to 
	def __init__(self, config):

		# Random number generator
		self.random = random.SystemRandom()

		# Store sinulation configuration data
		self.material = config["material"]
		self.energy   = config["energy"]
		self.field	  = config["field"]
		self.maxiter  = config["maxiter"] 

		# Calculate scattering rates for phonon processes
		self.rates = materialScatteringRates( self.energy, self.material )

		# Initialize solid state electron object
		self.electon = solidStateElectron( material, "G")

	# This method will randomize the initial state of the electon	
	def randomizeInitial(self, Ef = 0.1):
		
		Processor = scatteringEventProcessor()
		Processor.isotropicScatteringEvent(self.electon, Ef, "G")


	# Method to simulate the time between scattering events	
	def flightTime(self):

		# Throw a random number on interval [0, 1]
		r = self.random.random()


		G = self.rates.maxScatteringRates( self.electron.valley )

		# Return a free flight time
		 (-1.0/ G )*np.log(r)



if __name__ == "__main__":



	config = {
		"material"	: GaAs(),
		"energy"	: np.linspace(0.0, 1.0, 100),
		"field"		: 1000,
		"maxiter"	: 1e5
	}

	MC = monteCarlo(config)
	MC.randomizeInitial()



	#MC.run()
