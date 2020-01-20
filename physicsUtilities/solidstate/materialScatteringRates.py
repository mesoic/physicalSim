# ---------------------------------------------------------------------------------
# 	physicsUtilities/solidstate -> materialScatteringRates.py
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

# Import phonon scattering rates
from .phononScatteringRates import phononScatteringRates
from .phononScatteringRates import phononScatteringData

# Plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Container class for material phonons
class materialScatteringRates:

	def __init__(self, energy, material):

		# Phonon scattering rates
		self.PSR = phononScatteringRates()

		# Build scattering rates
		self.buildScatteringRates(energy, material)

		# Store energy vector and material (for plotting)
		self.material = material
		self.energy   = energy 

	# Method to build scattering rates over energy range
	def buildScatteringRates(self, energy, material):
		
		# Scattering rates are to be stored in a tuple keyed dictionary
		self.scatteringRates = {

			# Gamma valley
			# Acoustic phonons
			(0, "G", "Gac", None) 			: self.PSR.acousticPhonons(energy, material, valley="G"),

			# Optical phonons
			(1, "G", "Gop", "Absorption")	: self.PSR.opticalPhonons(energy, material, mode="Absorption", valley="G"),
			(2, "G", "Gop", "Emission")	 	: self.PSR.opticalPhonons(energy, material, mode="Emission"  , valley="G"),
			
			# Intervalley processes			
			(3, "G", "GtoL", "Absorption") 	: self.PSR.intervalleyGtoL(energy, material, mode="Absorption"),
			(4, "G", "GtoL", "Emission")   	: self.PSR.intervalleyGtoL(energy, material, mode="Emission"),
	
			# L valley
			# Acoustic phonons
			(0, "L", "Lac", None) 			: self.PSR.acousticPhonons(energy, material, valley="L"),

			# Optical phonons
			(1, "L", "Lop", "Absorption")   : self.PSR.opticalPhonons(energy, material, mode="Absorption", valley="L"),
			(2, "L", "Lop", "Emission") 	: self.PSR.opticalPhonons(energy, material, mode="Emission"  , valley="L"),
			
			# Intervalley processes			
			(3, "L", "LtoG", "Absorption") 	: self.PSR.intervalleyLtoG(energy, material, mode="Absorption"),
			(4, "L", "LtoG", "Emission")	: self.PSR.intervalleyLtoG(energy, material, mode="Emission"),
			(5, "L", "LtoL", "Absorption") 	: self.PSR.intervalleyLtoL(energy, material, mode="Absorption"),
			(6, "L", "LtoL", "Emission") 	: self.PSR.intervalleyLtoL(energy, material, mode="Emission"),
		}

		# Calculate sum of scattering rates (Gamma)
		Gsum = np.zeros( len(energy) )
		Lsum = np.zeros( len(energy) )
		
		for index in range(5):

			Gsum  += self.getScatteringRate( [index, "G"] ) 
	 	
		for index in range(7):

			Lsum  += self.getScatteringRate( [index, "L"] )

		# Add sums to array as phononScatteringData objects (no meta)
		self.scatteringRates[ (None, None, "Gsum", None) ] = phononScatteringData( rate = Gsum, meta = None)
		self.scatteringRates[ (None, None, "Lsum", None) ] = phononScatteringData( rate = Lsum, meta = None)

	# Method to get scattering rate
	def getScatteringRate(self, keys) :

		for rate_key in self.scatteringRates.keys():

			if all( [k in rate_key for k in list(keys)] ):

				return self.scatteringRates[rate_key].get_rate()

		# If key is not found retrun none
		return None		

	# Method to get scattering metadata
	def getScatteringMeta(self, keys):

		for rate_key in self.scatteringRates.keys():

			if all( [k in rate_key for k in list(keys)] ):

				return self.scatteringRates[rate_key].get_meta()

		# If key is not found retrun none
		return None		


	# Method to transform zero scattering rates into nan
	def zeroAsNan(self, rate):

		return [ np.nan if _ == 0 else _ for _ in rate ] 


	# Method to show all scattering rates
	def showRates(self):

		# This will be plotted using gridspec
		fig = plt.figure(constrained_layout=True, figsize=(16,9))
		gs = gridspec.GridSpec(ncols = 3, nrows = 2, figure=fig)

		# Acoustic Phonons
		ax00 = fig.add_subplot(gs[0, 0])
		h0,  = ax00.semilogy(self.energy, self.getScatteringRate( ["Gac"] ) )
		h1,  = ax00.semilogy(self.energy, self.getScatteringRate( ["Lac"] ) )
		ax00.set_xlabel("Energy $(eV)$")
		ax00.set_ylabel("Acoustic $(s^{-1})$")
		ax00.legend([h0,h1],["$\Gamma$ Valley","L valley"])

		# Optical Phonons (Gamma)
		ax01 = fig.add_subplot(gs[0, 1])
		h0,  = ax01.plot(self.energy, self.zeroAsNan( self.getScatteringRate( ["Gop", "Absorption"] ) ) )
		h1,  = ax01.plot(self.energy, self.zeroAsNan( self.getScatteringRate( ["Gop", "Emission"] ) ) )
		ax01.set_xlabel("Energy $(eV)$")
		ax01.set_ylabel("$\Gamma$ Optical $(s^{-1})$")
		ax01.legend([h0,h1],["Absorption", "Emission"])

		# Optical Phonons (L)
		ax02 = fig.add_subplot(gs[0, 2])
		h0,  = ax02.plot(self.energy, self.zeroAsNan( self.getScatteringRate( ["Lop", "Absorption"] ) ) )
		h1,  = ax02.plot(self.energy, self.zeroAsNan( self.getScatteringRate( ["Lop", "Emission"] ) ) )
		ax02.set_xlabel("Energy $(eV)$")
		ax02.set_ylabel("L Optical $(s^{-1})$")
		ax02.legend([h0,h1],["Absorption", "Emission"])


		# Intervalley scattering (G -> L)
		ax10 = fig.add_subplot(gs[1, 0])
		h0,  = ax10.semilogy(self.energy, self.getScatteringRate( [ "GtoL", "Absorption"] ) )
		h1,  = ax10.semilogy(self.energy, self.getScatteringRate( [ "GtoL", "Emission"] ) )
		ax10.set_xlabel("Energy $(eV)$")
		ax10.set_ylabel("($\Gamma$ $\\rightarrow$ L) Intervalley $(s^{-1})$")
		ax10.legend([h0,h1],["Absorption", "Emission"])

		# Intervalley scattering (L -> G)
		ax11 = fig.add_subplot(gs[1, 1])
		h0,  = ax11.semilogy(self.energy, self.getScatteringRate( ["LtoG", "Absorption"] ) )
		h1,  = ax11.semilogy(self.energy, self.getScatteringRate( ["LtoG", "Emission"] ) )
		ax11.set_xlabel("Energy $(eV)$")
		ax11.set_ylabel("(L $\\rightarrow$ $\Gamma$) Intervalley $(s^{-1})$")
		ax11.legend([h0,h1],["Absorption", "Emission"])

		# Intervalley scattering (L -> L)
		ax12 = fig.add_subplot(gs[1, 2])
		h0,  = ax12.semilogy(self.energy, self.getScatteringRate( ["LtoL", "Absorption"] ) )
		h1,  = ax12.semilogy(self.energy, self.getScatteringRate( ["LtoL", "Emission"] ) )
		ax12.set_xlabel("Energy $(eV)$")
		ax12.set_ylabel("(L $\\rightarrow$ L) Intervalley $(s^{-1})$")
		ax12.legend([h0,h1],["Absorption", "Emission"])

		# Add a tile and show plot
		fig.suptitle("%s electron-phonon scattering rates (T = %sK)"%(self.material.name, self.material.T))

		# Plot sum of scattering rates
		fig = plt.figure()
		ax0 = fig.add_subplot(111)
		h0, = ax0.semilogy( self.energy, self.zeroAsNan( self.getScatteringRate( ["Gsum"] ) ) )
		h1, = ax0.semilogy( self.energy, self.zeroAsNan( self.getScatteringRate( ["Lsum"] ) ) )
		ax0.set_xlabel("Energy $(eV)$")
		ax0.set_ylabel("$\Sigma \Gamma_i$ $(s^{-1})$")
		ax0.set_title("Total Scattering Rate")
		ax0.legend([h0,h1],["$\Gamma$ Valley","L valley"])

		plt.show()
