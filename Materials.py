import numpy as np

class material:

	def __init__(self):

		# Vacuum Permitivvity (F/cm)
		self.e0 = 8.85e-14 

		# Boltzman Constant (eV/K)
		self.kb = 8.617e-5

		# Electric Charge (C)
		self.q  = 1.602e-19


# A container for the Silicon parameters.
class Silicon(material):

	def __init__(self, T=300):
	
		# Initialize physical constants
		material.__init__(self)

		# Intrinsic carrier density (#/cm-3)
		self.ni = 1.5e10

		# Relative Permativvity
		self.ep = self.e0 * 11.9

		# Thermal voltage
		self.Vt = self.kb * T

		# Debye Length (intrinsic)
		self.Ld = np.sqrt( (self.ep * self.Vt) / (self.q * self.ni) )
