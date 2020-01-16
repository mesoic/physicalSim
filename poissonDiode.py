#!/usr/bin/env python
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

import discreteDifferentialEquation as DDE

from Materials import Silicon

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

		# Build operators
		DiscreteOperators = DDE.Operators(self.simsize, self.dx / self.material.Ld) 

		# Calculate boundary conditions
		BoundaryConditions = DDE.BoundaryConditions(self.simsize, self.dx / self.material.Ld)
		BoundaryConditions.addDirchlet(phi[ 0],  pos="0")
		BoundaryConditions.addDirchlet(phi[-1], pos="N")

		# Convergence comparison
		self.step = 0

		# If boundary conditions are valid perform iteration
		if BoundaryConditions.areValid():

			while True:

				# Calculate cost function
				eF = np.dot(DiscreteOperators.DD(), phi) + BoundaryConditions.evaluate(phi) - self.f(phi)

				# Calculate convergence criteria 
				delta = ( sum( np.abs(eF) ) / self.simsize )

				# Print convergence condition
				if ( self.step % 1 ) == 0:
		
					print("Conv: %s"%delta)

				# Stop on max_iterations
				if ( self.step ) > self.maxiter:

					print( "Maximum number of iterations %s", self.step)
						
					return phi

				if ( delta ) < self.converge:

					return phi

				else:
			
					J, dF = DiscreteOperators.DD(), self.df(phi) 

					for i in range( self.simsize ):

						J[i][i]	-= dF[i]

					phi -= self.epsilon * np.dot( la.inv(J) , eF )

					self.step += 1

			return phi		


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
		"npoints" : 100,
		"epsilon" : 1.0,
		"converge": 1e-8,
		"maxiter" : 10
	}

	diode = pnJunction(config)

	# Guess function (phi' = phi/Vt scale)
	N0  = 0.5 * ( np.sqrt( ( diode.Na - diode.Nd )**2 + 4*diode.material.ni**2 ) + diode.Nd - diode.Na )
	phi = np.log( N0 / diode.material.ni  ) 


	plt.plot( toum(diode.x), diode.material.Vt * phi )	
	phi_solve = diode.solve( phi )
	plt.plot( toum(diode.x), diode.material.Vt * phi_solve ) 
	plt.show()

	############################
	# PLOTTING #
	############################
	# if False:

	# 	fig = plt.figure(2)
	# 	ax1 = fig.add_subplot(111)
	# 	ax1.plot(_domain, guess,"r")
	# 	ax1.plot(_domain, guess0,"r", linestyle="--")
	# 	ax2 = ax1.twinx()
	# 	ax2.semilogy(_domain, s.Nd,"k", linestyle="--")
	# 	ax2.semilogy(_domain, s.Na,"k", linestyle="--")
	# 	ax2.set_ylim(1e14,1e18)
	# 	ax1.set_xlabel("Depth $(\mu m)$")
	# 	ax1.set_ylabel("Potential $(V)$")
	# 	ax2.set_ylabel("Carrier Density $(cm^{-3})$")
	# 	ax1.set_ylim(-0.4,0.5)

	# 	fig = plt.figure(3)
	# 	ax1 = plt.subplot(111)
	# 	ax2 = ax1.twinx()
	# 	ax1.semilogy(_domain, Si.ni*s.getN(guess/Si.Vt),"k")
	# 	ax2.semilogy(_domain, Si.ni*s.getP(guess/Si.Vt),"r")
	# 	ax1.set_xlabel("Depth $(\mu m)$")
	# 	ax1.set_ylabel("Electron Density $(cm^{-3})$")
	# 	ax2.set_ylabel("Hole Density $(cm^{-3})$")
	# 	plt.show()