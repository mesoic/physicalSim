# ---------------------------------------------------------------------------------
# 	diodePotential -> poissonDiode.py
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
import numpy.linalg as la
import matplotlib.pyplot as plt

# For solver
from copy import deepcopy 


# Import Silicon material constants
from physicsUtilities.solidstate.materialConstants import Silicon

# Import discrete differential equation
import physicsUtilities.utilities.discreteDifferentialEquation as DDE

# A class to set up and solve Poisson via the finite difference method
class pnJunction:

	# To start we store the domain and range
	def __init__(self, config):

		# Material proprties
		self.material = Silicon()

		# Simulation configuration
		self.npoints  = config["npoints"]
		self.epsilon  = config["epsilon"]
		self.converge = config["converge"]
		self.maxiter  = config["maxiter"]

		# Thickness of doping (cm)
		self.d 	= 1.0e-4

		# Domain (x) to solve (cm) - remove first and last value ()
		self.x  = np.linspace(0.0, 10.0 * self.d, self.npoints)[1:-1]
		self.dx = self.x[1] - self.x[0]

		# Simulation size
		self.simsize  = self.npoints - 2

		# Base donor and acceptor densities (#/cm-3)
		self.Nd0 = 1.0e18
		self.Na0 = 1.0e15

		# Calculate doping profiles
		self.Nd = self.Nd_profile()
		self.Na = self.Na_profile()

	# Donor doping profile: Nd(x)
	def Nd_profile(self):
	
		return np.array( [ self.Nd0 * np.exp( -1.0 * ( _x / self.d )**2 ) for _x in self.x ] )

	# Acceptor doping profile: Na(x)
	def Na_profile(self):	

		return np.array( [ self.Na0 for _x in self.x ] )

	# Number of carriers: f(x, phi(x))
	def f(self, phi): 

		n =  np.exp(  np.array(phi) ) - ( self.Nd / self.material.ni )
		p = -np.exp( -np.array(phi) ) + ( self.Na / self.material.ni )

		return ( n + p )
	
	# Derivative of carriers: df(x, phi(x))
	def df(self, phi): 

		dn =  np.exp(  np.array(phi) )
		dp =  np.exp( -np.array(phi) )

		return ( dn + dp ) 

	# Solve Poisson Equation: DD(phi'(x')) = f(x', phi'(x')) 
	# 	
	# 	arg : phi = initial guess to solution normalized to Vt
	#
	def solve(self, phi):

		# Pass by value behaviour
		phi_solve = deepcopy(phi)

		# Build operators
		DiscreteOperators = DDE.Operators(self.simsize, self.dx / self.material.Ld) 

		# Calculate boundary conditions
		BoundaryConditions = DDE.BoundaryConditions(self.simsize, self.dx / self.material.Ld)
		BoundaryConditions.addDirchlet(phi[ 0], pos="0")
		BoundaryConditions.addDirchlet(phi[-1], pos="N")

		# Convergence comparison
		self.step = 0

		# If boundary conditions are valid perform iteration
		if BoundaryConditions.areValid():

			while True:

				# Calculate cost function
				eF = np.dot(DiscreteOperators.DD(), phi_solve) + BoundaryConditions.evaluate(phi) - self.f(phi_solve)

				# Calculate convergence criteria 
				delta = ( sum( np.abs(eF) ) / self.simsize )

				# Print convergence condition
				if ( self.step % 1 ) == 0:
		
					print("Conv: %s"%delta)

				# Stop on max_iterations
				if ( self.step ) > self.maxiter:

					print( "Maximum number of iterations %s", self.step)
						
					return phi_solve

				if ( delta ) < self.converge:

					return phi_solve

				else:
			
					J, dF = DiscreteOperators.DD(), self.df(phi_solve) 

					for i in range( self.simsize ):

						J[i][i]	-= dF[i]

					phi_solve -= self.epsilon * np.dot( la.inv(J) , eF )

					self.step += 1

			return phi_solve


if __name__ == "__main__":

	# lambda: convert cm to um 
	def toum(vec):
		return np.array( [ v/1e-4 for v in vec] )

	# pn-Junction Transformation 
	#
	# 	domain 		: { x'   | x/Ld   } (Debye length scaling)		
	# 	potential 	: { phi' | phi/Vt } (Thermal voltage scaling) 
	#
	config = {
		"npoints" : 1000,
		"epsilon" : 1.0,
		"converge": 1e-8,
		"maxiter" : 10
	}

	# Note that "npoints = N" will result in a domain of len = N - 2 
	# due to the inclusion of boundary conditions
	diode = pnJunction(config)

	# Guess function (phi' = phi/Vt scale)
	N0  = 0.5 * ( np.sqrt( ( diode.Na - diode.Nd )**2 + 4*diode.material.ni**2 ) + diode.Nd - diode.Na )
	phi = np.log( N0 / diode.material.ni  ) 

	# Call the solver
	phi_solve = diode.solve( phi )

	# Plot results
	fig = plt.figure()
	ax0 = fig.add_subplot(111)
	ax0.set_title("pn-Junction : $N_D = N^0_De^{-x^2/\\delta^2}$ : $N_A = N_A^0$")
	ax0.set_xlabel("Distance $(\\mu m)$") 
	ax0.set_ylabel("Potential $(\\phi)$") 
	h0, = ax0.plot( toum(diode.x), diode.material.Vt * phi )	
	h1, = ax0.plot( toum(diode.x), diode.material.Vt * phi_solve ) 

	# place a text box in upper left in axes coords
	props = dict(boxstyle='round', facecolor='white', alpha=0.5, linewidth=0.4)
	ax0.text(0.53, 0.97,
		"$\\partial^2_{x'} \\phi' = e^{\\phi'} - e^{-\\phi'} - N_D/n_i + N_A/n_i$   \n $\\phi' \\rightarrow \\phi / V_T$ \n $x' \\rightarrow x / \\lambda^i_D$", 
		transform=ax0.transAxes, 
		verticalalignment='top', 
		bbox=props
    )
	ax0.legend([h0, h1], ["Approximation", "DDE Solution"], loc=(0.68, 0.775))	


	# Carrier density plot
	fig = plt.figure()
	ax0 = fig.add_subplot(111)
	ax0.set_title("pn-Junction : $N_D = N^0_De^{-x^2/\\delta^2}$ : $N_A = N_A^0$")
	ax0.set_xlabel("Distance $(\\mu m)$") 
	ax0.set_ylabel("Carrier Density $(cm^{-3})$") 
	n_carriers = [np.nan if _  > 0 else abs(_) for _ in diode.material.ni * diode.f(phi_solve * diode.material.Vt) ]
	p_carriers = [np.nan if _ <= 0 else abs(_) for _ in diode.material.ni * diode.f(phi_solve * diode.material.Vt) ]
	h0, = ax0.semilogy( toum(diode.x) , n_carriers)
	h1, = ax0.semilogy( toum(diode.x) , p_carriers)
	ax0.legend([h0, h1], ["n-density", "p-density"])	
	plt.show()