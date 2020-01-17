


import random
from physicsUtilities.materialConstants import GaAs


# Method to hold cylind
class cylindricalWavevector:

	def __init__(self, kz, kr):
	
		self.kz  = kz 
		self.kr  = kr 

		self.mag = np.sqrt(kz**2 + kr**2)

# This class simulates scattering events on eletron in a solid state material
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
	def update(self, E, kz, kr, valley ):
	
		# Electron valley occupancy
		self.valley = valley			

		# Update perameters
		self.m = self.material.effectiveMass( valley )
		self.K = cylindricalWavevector(0.0, 0.0)
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
	def magK(self, E): 

		return np.sqrt( (2.0 * self.getM() * _E) / (self.material.hbar**2) )

	# A method to increment the k-vector for free flight
	def accelerationWavevector(self, ki, kf):

		return cylindricalWavevector( ki.kz + kf.kz , ki.kr + kf.kr)


	# This method simulates isotropic scattering events by generating a randomly 
	# oriented wavevector for an electron that has scattered into a state with 
	# energy (Ef) in dispersion valley (Vf) in ['Gamma', 'L'] and updates the 
	# electron state accordingly.
	def isotropicScatteringEvent(self, electron, Ef, Vf):

		# Throw a random number on interval [0, 1]
		r = self.random.random()
		
		# After an isotropic scattering event, the angle with respect to the 
		# electric field oriented randomly on the interval [0, 2pi]. 
		Kzf = self.magK(Ef) * np.cos( 2.0 * np.pi * r ) 

		# Calculate the radial component of the wavevector: |K|^2 = Kz^2 + Kr^2 
		Krf = self.magK(Ef) * np.sin( 2.0 * np.pi * r )

		# Update electron state
		electron.update(Ef, Kzf, Krf, Vf)


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

		# Update electron state
		electron.update(Ef, Kzf, Krf, Vf)

# 
if __name__ == "__main__":

	# Define material to simulate
	material = GaAs()

	electron = solidStateElectron( material )