

# Import numpy
import numpy as np
import random

# ---------------------------------------------------------------------------------
# 	physicsUtilities -> velocityFieldUtilities.py
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

	# Update electron state
	def update(self, E, K, valley ):
	
		# Electron valley occupancy
		self.valley = valley			

		# Update perameters
		self.m = self.material.effectiveMass( valley )
		self.K = K
		self.E = E


# The purpouse of this class is to process to prepare wavevectors an electron that has 
# undergone a scattering event into a state with energy Ef. For scattering events, we 
# consider a cylindrical coordinate system in which the z-axis is oriented parallel to 
# the electric field.
class scatteringEventProcessor:

	# Namespace
	def __init__(self):

		# Random number generator
		self.random = random.SystemRandom()

	# Return magnitude of wavevector given energy: 
	# 	|k|^2 = 2mE/hbar^2
	def magK(self, electron, Ef): 

		# Initialize physical constants
		const = physicalConstants()

		# Return magnitude of wavevector
		return np.sqrt( (2.0 * electron.m * Ef) / (const.hbar**2) )

	# Return energy for a given wavevector
	def magE(self, electron, Kf ):

		const = physicalConstants()

		return ( Kf.mag * const.hbar)**2 / (2.0 * electron.m)


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
		Ef  = self.magE( electron, Kf )

		# Update electron state
		electron.update(Ef, Kf, Vf)


	# This method simulates isotropic scattering events by generating a randomly 
	# oriented wavevector for an electron that has scattered into a state with 
	# energy (Ef) in dispersion valley (Vf) in ['Gamma', 'L'] and updates the 
	# electron state accordingly.
	def isotropicScatteringEvent(self, electron, Ef, Vf):

		# Throw a random number on interval [0, 1]
		r = self.random.random()
		
		# After an isotropic scattering event, the angle with respect to the 
		# electric field oriented randomly on the interval [0, 2pi]. 
		Kzf = self.magK(electron, Ef) * np.cos( 2.0 * np.pi * r ) 

		# Calculate the radial component of the wavevector: |K|^2 = Kz^2 + Kr^2 
		Krf = self.magK(electron, Ef) * np.sin( 2.0 * np.pi * r )

		# Initialize cylindrical wavevector
		Kf  = cylindricalWavevector( Kzf, Krf )

		# Update electron state
		electron.update(Ef, Kf, Vf)


	# This method generates a wavevector that is preferentially oriented along 
	# the original wavevector. 
	def anisotropicWavevectorMC(self, electron, Ef, Ei, Vf):

		# Throw a random number on interval [0, 1]
		r  = self.random.random()

		# The parameter (xi) governing anisotropic scattering
		xi = np.sqrt( Ei * Ef ) / ( np.sqrt(Ei) - np.sqrt(Ef) )

		# Calculate cos(theta) : theta scattering angle in a rotated system
		cos_theta = ( (1 + x) - np.power( 1.0 + 2.0 * x, r) ) / x
		sin_theta = np.sqrt(1 - cos_theta**2)
		cos_phi   = np.cos(2.0 * np.pi * r)

		# Components in unrotated coordinate system
		cos_alpha = electron.K.kz / electron.K.mag
		sin_alpha = np.sqrt(1.0 - cos_alpha**2)
	
		# Calculate the new k componenets
		Kzf = self.magK(Ef) * ( cos_alpha * cos_theta - sin_alpha * sin_theta *cos_phi )
		Krf = self.magK(Ef) * np.sqrt(1 - ( cos_alpha * cos_theta - sin_alpha * sin_theta *cos_phi )**2)

		# Initialize cylindrical wavevector
		Kf  = cylindricalWavevector( Kzf, Krf )

		# Update electron state
		electron.update(Ef, Kf, Vf)
