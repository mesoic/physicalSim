# ---------------------------------------------------------------------------------
# 	scatteringMonteCarlo -> scatteringEventProcessor.py
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

# Import numpy and random
import numpy as np
import random

# So we can access physicsUtilities directory
import sys
sys.path.insert(1, '..')

# Import physical and material constants
from physicsUtilities.physicalConstants import physicalConstants

# A data class to hold cylindrical wavevectors
class cylindricalWavevector:

	def __init__(self, kz, kr):
	
		self.kz  = kz 
		self.kr  = kr 

		self.mag = np.sqrt(kz**2 + kr**2)

# A data class to hold the state of the electron. 
class solidStateElectron:

	# Initialization method
	def __init__(self, material, valley = "G"):

		# Cache the material propertes object
		self.material = material
		
		# Electron valley occupancy
		self.valley = valley

		# Initialize some perameters
		self.m = self.material.effectiveMass( valley )
		self.K = cylindricalWavevector(0.0, 0.0)
		self.E = 0.0
		self.v = 0.0

	# Update electron state
	def update(self, E, K, valley ):
	
		# Electron valley occupancy
		self.valley = valley

		# Update perameters
		self.m = self.material.effectiveMass( valley )
		self.K = K
		self.E = E if E >= 0 else 0.0
		self.v = ( self.K.kz * self.material.hbar ) / self.m

# The purpouse of this class is to process to prepare wavevectors an electron that has 
# undergone a scattering event into a state with energy Ef. For scattering events, we 
# consider a cylindrical coordinate system in which the z-axis is oriented parallel to 
# the electric field.
class scatteringEventProcessor:

	# Namespace
	def __init__(self, rates):

		# Random number generator
		self.random = random.SystemRandom()

		# Store material scattering rates
		self.rates = rates

		# Buils scattering matrices
		self.buildScatteringMatrices()

	# Return magnitude of wavevector given energy: 
	# 	|k|^2 = 2mE/hbar^2
	def magK(self, mass, Ef): 

		# Initialize physical constants
		const = physicalConstants()

		# Protect negative energy
		Ef = Ef if Ef >= 0 else 0.0

		# Return magnitude of wavevector
		return np.sqrt( (2.0 * mass * Ef ) / (const.hbar**2) )

	# Return energy for a given wavevector
	def magE(self, mass, Kf ):

		const = physicalConstants()

		return ( Kf.mag * const.hbar)**2 / (2.0 * mass)

	# Method to simulate the time between scattering events. 
	def generateFlightTime(self, electron):

		# Throw a random number on interval [0, 1]
		r = self.random.random()

		# Get total scattering rate for current valley
		if electron.valley in ["G", "Gamma"]:

			# Simulate new time interval
			tau  = ( -1.0/ self.Gmax ) * np.log(r)

		if electron.valley in ["L"]:

			# Simulate new time interval
			tau  = ( -1.0/ self.Lmax ) * np.log(r)
			
		# Return free flight time
		return tau

	# A method to increment the k-vector for acceleration under free flight
	def accelerationEvent(self, electron, dK):

		# Valley occupancy is unchanged
		Vf  = electron.valley

		# Updare electron weavevector. For free acceleration in an electric 
		# field, all energy goes into axial component.
		Kzf = electron.K.kz + dK.kz
		Krf = electron.K.kr

		# Initialize cylindrical wavevector
		Kf  = cylindricalWavevector( Kzf, Krf )

		# Calculate total energy afer increment
		Ef  = self.magE( electron.m, Kf )

		# Update electron state
		electron.update(Ef, Kf, Vf)

	# Method to build the scattering matrices
	def buildScatteringMatrices(self):

		# Cache maximum scattering rate (G valley)
		self.Gmax = max( self.rates.getScatteringRate( ["Gsum"] ) )

		Gmat = np.zeros( ( 7 ,len(self.rates.energy) ) )

		for index in range( 5 ):

			Gmat[index + 1, :] = Gmat[index, :] + self.rates.getScatteringRate( [index, "G"] ) / self.Gmax

		Gmat[6, :] = np.ones( len(self.rates.energy) )

		# Cache maximum scattering rate (L valley)
		self.Lmax = max( self.rates.getScatteringRate( ["Lsum"] ) )

		Lmat = np.zeros( ( 9 ,len(self.rates.energy) ) )

		for index in range( 7 ):

			Lmat[index + 1, :] = Lmat[index, :] + self.rates.getScatteringRate( [index, "L"] ) / self.Lmax

		Lmat[8, :] = np.ones( len(self.rates.energy) )		

		# Build an object to hold the scattering matrices
		self.scatteringMatrices = {
			"G" : Gmat,
			"L" : Lmat
		}

	# Method to generate scattering event	
	def generateScatteringEvent(self, electron):

		# Find the column index in scattering matrix for the current electron energy. 
		# This approximates the scattering rates for that energy
		tmp = abs( self.rates.energy - electron.E)
		col = list(tmp).index( min(tmp) )

		# Extract the scattering rates from the scattering matrix
		R = self.scatteringMatrices[electron.valley][:, col]		

		# Throw a random number on interval [0, 1] to determine which event we will 
		r = self.random.random()

		# Find the index of scattering event. This is the index of the lower value 
		# of the values that r is between in R.
		index = np.argmax( R > r ) - 1

		# Get the corresponding scattering event metadata. getScatteringMeta will 
		# return None if we end up in the last slot of the scattering matrix. 
		meta = self.rates.getScatteringMeta( [index, electron.valley] )
		
		# If we have thrown a real scattering event, must update the electron state
		if meta is not None:

			if meta["sym"] == "isotropic": 

				self.isotropicScatteringEvent(electron, meta["dE"], meta["Vf"] )

			if meta["sym"] == "anisotropic": 

				self.anisotropicScatteringEvent(electron, meta["dE"], meta["Vf"] )

	# This method simulates isotropic scattering events by generating a randomly 
	# oriented wavevector for an electron that has scattered into a state with 
	# energy (Ef) in dispersion valley (Vf) in ['Gamma', 'L'] and updates the 
	# electron state accordingly.
	def isotropicScatteringEvent(self, electron, dE, Vf):

		# Throw a random number on interval [0, 1]
		r = self.random.random()

		# Calculate the energy after scattering
		Ef = electron.E + dE

		# Protect negative energy
		Ef = Ef if Ef >= 0 else 0.0
		
		# Extract the effective mass for the final state valley 
		m  = electron.material.effectiveMass(Vf)

		# After an isotropic scattering event, the angle with respect to the 
		# electric field oriented randomly on the interval [0, 2pi]. 
		Kzf = self.magK(m, Ef) * np.cos( 2.0 * np.pi * r ) 

		# Calculate the radial component of the wavevector: |K|^2 = Kz^2 + Kr^2 
		Krf = self.magK(m, Ef) * np.sin( 2.0 * np.pi * r )

		# Initialize cylindrical wavevector
		Kf  = cylindricalWavevector( Kzf, Krf )

		# Update electron state
		electron.update(Ef, Kf, Vf)

	# This method generates a wavevector that is preferentially oriented along 
	# the original wavevector. 
	def anisotropicScatteringEvent(self, electron, dE, Vf):

		# Store initial and final energies
		Ei = electron.E
		Ef = electron.E + dE

		# Extract the effective mass for the final state valley 
		m  = electron.material.effectiveMass(Vf)

		# Throw a random number on interval [0, 1]
		r  = self.random.random()

		if (Ef >= 0):

			# The parameter (xi) governing anisotropic scattering
			xi = 2.0 * np.sqrt( Ei * Ef ) / ( np.sqrt(Ei) - np.sqrt(Ef) )**2

			# Calculate cos(theta) : theta scattering angle in a rotated system
			cos_theta = ( (1 + xi) - np.power( 1.0 + 2.0 * xi, r) ) / xi

		else: 

			cos_theta = 1.0

		sin_theta = np.sqrt(1 - cos_theta**2)
		cos_phi   = np.cos(2.0 * np.pi * r)

		# Components in unrotated coordinate system
		cos_alpha = electron.K.kz / electron.K.mag
		sin_alpha = np.sqrt(1.0 - cos_alpha**2)
	
		# Calculate the new k componenets
		Kzf = self.magK(m, Ef) * ( cos_alpha * cos_theta - sin_alpha * sin_theta *cos_phi )
		Krf = self.magK(m, Ef) * np.sqrt(1 - ( cos_alpha * cos_theta - sin_alpha * sin_theta *cos_phi )**2)

		# Initialize cylindrical wavevector
		Kf  = cylindricalWavevector( Kzf, Krf )

		# Update electron state
		electron.update(Ef, Kf, Vf)
