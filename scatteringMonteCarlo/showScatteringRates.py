# ---------------------------------------------------------------------------------
# 	physicsUtilities -> materialScatteringRates.py
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

# So we can access physicsUtilities directory
import sys
sys.path.insert(1, '..')

# Import physical and material constants
from physicsUtilities.materialConstants import GaAs
from physicsUtilities.materialScatteringRates import materialScatteringRates

# Program to plot phonon scattering rates in GaAs
if __name__ == "__main__":

	# Define material to simulate
	material = GaAs()

	# Define energy range to simulate
	energy = np.linspace(0.0, 1.0, 5) # (eV)

	# Calculate scattering rates for phonon processes
	rates = materialScatteringRates( energy, material )
	rates.showRates()
	