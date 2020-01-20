# ---------------------------------------------------------------------------------
# 	physicsUtilities -> physcialConstants.py
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
class physicalConstants:

	def __init__(self):

		# Vacuum Permitivvity (F/cm)  
		# 	for (eV/V^2) divide by self.q**2
		self.e0 = 8.8548e-14 

		# Boltzman Constant (eV/K)
		self.kb = 8.617e-5

		# Electric Charge (C)
		self.q  = 1.602e-19

		# Pi (unitless)
		self.pi = 3.14159265359

		# Reduced Planck constant h/2pi (eV/s) 
		self.hbar = 6.5821e-16

		# Speed of light (cm/s)
		self.c = 2.99792458e10

		# Electron Mass (eV/c^2)
		self.me = 0.511e6/(self.c**2)
