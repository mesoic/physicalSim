# ---------------------------------------------------------------------------------
# 	physicsUtilities -> materialConstants.py
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

# Import physical constants class
from .physicalConstants import physicalConstants

# A container for the Silicon parameters.
class Silicon(physicalConstants):

	def __init__(self, T=300):
	
		# Initialize physical constants
		physicalConstants.__init__(self)

		# Intrinsic carrier density (#/cm-3)
		self.ni = 1.5e10

		# Relative Permativvity
		self.ep = self.e0 * 11.9

		# Thermal voltage
		self.Vt = self.kb * T

		# Debye Length (intrinsic)
		self.Ld = np.sqrt( (self.ep * self.Vt) / (self.q * self.ni) )


# A container for the GaAs parameters.
class GaAs(physicalConstants):

	def __init__(self, T=300):

		# Material name
		self.name = "GaAs"

		# Initialize physical constants
		physicalConstants.__init__(self)


		##################################
		# MATERIAL CONSTANTS 
		#

		# Density (eV/c^2) / (cm^3)
		self.rho = 3.0123e33/(self.c**2)

		# Speed of Sound (cm/s)
		self.nu = 5.22e5
		
		# High Frequency Dielectric Const
		self.epI = 10.82
		
		# Static Dielectric Const
		self.ep0 = 12.53
		
		# Thermal voltage
		self.Vt = self.kb * T

		# Temperature
		self.T = T

		##################################
		# PHONON FREQUENCIES 
		#

		# Polar optical phonon frequency (rad/s)
		self.wOP = 5.37e13

		# Eq. intervalley phonon frequency (rad/s)
		self.wE = 4.54e13
		
		
		##################################
		# DEFORMATION POTENTIALS
		#

		# Acoustic deformation potential (eV)
		self.Da= 7.0

		# Eq Intervalley deformation potential (eV)
		self.De= 1e9
		

		##################################
		# ELECTRON DISPERSION PARAMETERS
		#

		# GAMMA valley effective mass (eV/c^2)
		self.mG = 0.067*self.me

		# L valley effective mass (eV/c^2)
		self.mL = 0.350*self.me
	
		# L valley degeneracy (#)
		self.gL = 3.0

		# GAMMA -> L valley energy gap (eV)
		self.D = 0.36

	def effectiveMass( self, valley ): 
	
		if valley in ["G", "GAMMA"]:

			return self.mG

		if valley in ["L"]:

			return self.mL