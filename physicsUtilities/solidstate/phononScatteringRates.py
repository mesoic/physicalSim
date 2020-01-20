# ---------------------------------------------------------------------------------
# 	physicsUtilities/solidstate -> phononScatteringRates.py
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

# Import physical constants
from ..utilities.physicalConstants import physicalConstants

# Container class for phonon scattering data
class phononScatteringData:

	def __init__(self, rate, meta):

		# Scattering rate for phonon
		self.rate = rate

		# Scattering metadata for phonon
		#
		#	 meta = {
		#		"type"	= ["acoustic", "optical", "intervalley"]
		#		"sym"	= ["isotropic", "anisotropic"]
		#		"mode"	= ["Absorption, Emission", None] 
		#		"dE" 	= associated change in energy
		#		"Vi" 	= valley before scattring
		#		"Vf"	= valley after scattring
		# 	}
		self.meta = meta

	# Get rate
	def get_rate(self):
	
		return self.rate

	# Get meta
	def get_meta(self):

		return self.meta

	
# Phonon Scattering Rates Class
class phononScatteringRates:

	# Namespace class
	def __init__(self):

		pass

	# Acoustic phonon scattering
	def acousticPhonons(self, Ei, material, valley = "G"):
		
		# Store physical constants
		const = physicalConstants()

		# Store effectve mass for specified valley
		mass = material.effectiveMass( valley )

		# Calculate scattering rates
		a = np.power( 2.0 * mass, 1.5 ) * material.Vt * (material.Da**2)
		b = 4 * const.pi * material.rho * (material.nu**2) * (const.hbar**4)
		
		# Scattering rate
		GAMMA = (a / b) * np.sqrt( Ei )

		# Phonon metadata
		META = {
			"type"	: "acoustic",
			"sym"	: "isotropic",
			"mode"	: None,
			"dE" 	: 0.0,
			"Vi" 	: valley,
			"Vf"	: valley
		}

		return phononScatteringData( np.array(GAMMA), META ) 

	# Optical phonon scattering (absorbtion and emission)
	def opticalPhonons(self, Ei, material, mode, valley = "G" ):

		# Store physical constants
		const = physicalConstants()

		# Store effectve mass for specified valley
		mass = material.effectiveMass( valley )
		
		# Phonon occupation number
		Nop = 1.0 / ( np.exp( material.wOP * const.hbar / material.Vt ) - 1 )
		
		# Phonon absorbtion
		if mode in ["A", "Absorption"]: 
			
			dE 	 = (const.hbar*material.wOP) 
			Nop += 0.0

		# Phonon emission
		if mode in ["E", "Emission"]: 

			dE 	 = -(const.hbar*material.wOP)
			Nop += 1.0

		# Increment energy
		Ef = Ei + dE  
		
		# Calculate prefactor
		a = Nop * (const.q**2) * np.sqrt(mass) * material.wOP

		b = np.sqrt(2.0) * const.hbar * (4.0 * const.pi * const.e0 * const.q)

		c = (1.0 / material.epI) - (1.0/material.ep0)

		F = (a / b) * c
	
		# Prepare empty array for scattering rates
		GAMMA = []

		# Calculate energy dependence
		for _Ei, _Ef in zip(Ei, Ef):

			# Check final energy:   Ef > 0.0 (transition allowed)
			# Check initial energy: Ei > 0.0 (log of negative number) 
			if _Ef > 0.0 and _Ei > 0.0:
	
				delta = np.abs( ( np.sqrt(_Ei) + np.sqrt( _Ef ) ) / ( np.sqrt(_Ei) - np.sqrt(_Ef) ) ) 

				GAMMA.append( F * np.log( delta ) / np.sqrt(_Ei) )

			# Otherwise append 0.0
			else:

				GAMMA.append( 0.0 )

		# Phonon metadata
		META = {
			"type"	: "optical",
			"sym"	: "anisotropic",
			"mode"	: mode,
			"dE"	: dE,
			"Vi" 	: valley,
			"Vf"  	: valley
		}		

		return phononScatteringData( np.array(GAMMA), META ) 

	# Intervalley scatering (Gamma -> L) 
	def intervalleyGtoL(self, Ei, material, mode):
		
		# Store physical constants
		const = physicalConstants()

		# Store effectve mass for specified valley. 
		mass = material.effectiveMass("L")
		
		# Phonon occupation number
		Nop = 1.0 / ( np.exp( material.wE * const.hbar / material.Vt ) - 1 )

		# Phonon absorbtion
		if mode in ["A", "Absorption"]: 

			dE   = (const.hbar*material.wE) - material.D	
			Nop += 0.0

		# Phonon emission
		if mode in ["E", "Emission"]: 

			dE   = -(const.hbar*material.wE) - material.D
			Nop += 1.0

		# Increment energy 
		Ef = Ei + dE

		# Calculate prefactor incuding valley degeneracy
		a = Nop * material.gL * np.power( mass, 1.5 ) * (material.De**2)
		b = np.sqrt(2.0) * const.pi * material.rho * material.wE * (const.hbar**3)

		# Prepare empty array for scattering rates
		GAMMA = []

		# Calculate energy dependence
		for _Ef in Ef: 

			# Check final energy Ef > 0.0 (transition allowed)
			GAMMA.append( (a / b) * np.sqrt(_Ef) if _Ef > 0.0 else 0.0 )

		# Prepare metadata
		META = {
			"type"	: "intervalley",
			"sym"	: "isotropic",
			"mode"	: mode,
			"dE"	: dE,
			"Vi" 	: "G",
			"Vf"  	: "L"
		}		

		return phononScatteringData( np.array(GAMMA), META ) 

	# Intervalley scatering (L -> Gamma) 
	def intervalleyLtoG(self, Ei, material, mode):

		# Store physical constants
		const = physicalConstants()

		# Store effectve mass for specified valley. 
		mass = material.effectiveMass("G")
		
		# Phonon occupation number
		Nop = 1.0 / ( np.exp( material.wE * const.hbar / material.Vt ) - 1 )

		# Phonon absorbtion
		if mode in ["A", "Absorption"]: 
	
			dE   = (const.hbar*material.wE) + material.D
			Nop += 0.0

		# Phonon emission
		if mode in ["E", "Emission"]: 

			dE   = -(const.hbar*material.wE) + material.D
			Nop += 1.0

		# Increment energy 
		Ef   = Ei + dE
			
		# Calculate prefactor
		a = Nop * np.power( mass, 1.5 )* (material.De**2)
		b = np.sqrt(2.0) * const.pi * material.rho * material.wE * (const.hbar**3)

		# Prepare empty array for scattering rates
		GAMMA = []

		# Calculate energy dependence
		for _Ef in Ef: 

			# Check final energy Ef > 0.0 (transition allowed)
			GAMMA.append( (a / b) * np.sqrt(_Ef) if _Ef > 0.0 else 0.0 )

		# Prepare metadata
		META = {
			"type"	: "intervalley",
			"sym"	: "isotropic",
			"mode"	: mode,
			"dE"	: dE,
			"Vi" 	: "L",
			"Vf"  	: "G"
		}		

		return phononScatteringData( np.array(GAMMA), META ) 

	# Intervalley scatering (L -> L) 
	def intervalleyLtoL(self, Ei, material, mode):

		# Store physical constants
		const = physicalConstants()

		# Store effectve mass for specified valley. 
		mass =  material.effectiveMass("L")
		
		# Phonon occupation number
		Nop = 1.0 / ( np.exp( material.wE * const.hbar / material.Vt ) - 1 )

		# Phonon absorbtion
		if mode in ["A", "Absorption"]: 
	
			dE   = (const.hbar*material.wE) 
			Nop += 0.0

		# Phonon emission
		if mode in ["E", "Emission"]: 

			dE   = -(const.hbar*material.wE) 
			Nop += 1.0

		# Increment energy
		Ef = Ei + dE

		# Calculate prefactor incuding valley degeneracy
		a = Nop * ( material.gL - 1 ) * np.power( mass, 1.5 ) * (material.De**2)
		b = np.sqrt(2.0) * const.pi * material.rho * material.wE * (const.hbar**3)

		# Prepare empty array for scattering rates
		GAMMA = []

		# Calculate energy dependence
		for _Ef in Ef: 

			# Check final energy Ef > 0.0 (transition allowed)
			GAMMA.append( (a / b) * np.sqrt(_Ef) if _Ef > 0.0 else 0.0 )

		# Prepare metadata
		META = {
			"type"	: "intervalley",
			"sym"	: "isotropic",
			"mode"	: mode,
			"dE"	: dE,
			"Vi" 	: "L",
			"Vf"  	: "L"
		}		

		return phononScatteringData( np.array(GAMMA), META ) 
