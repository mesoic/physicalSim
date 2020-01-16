
import numpy as np

# Classes to consrtruct and evaluate discrete differential equations

# Class to construct differential operators
class Operators:

	def __init__(self, size, delta = 1.0):

		# Store matrix size
		self.size = size

		# Store the differential scale
		self.delta = delta

		
	# Build first derivative operator
	def D(self):

		# First derivative
		D 	= np.zeros( (self.size, self.size) )

		# Loop through indices 
		for i in range(self.size):

			# Differential factor
			factor = 1.0  / ( 2.0 * self.delta )

			# Tridiagonal elements (+)
			try: 
				D[i][i+1] = 1.0 * factor

			except IndexError:		
				pass	

			# Tridiagonal elements (-)
			try:	
				# Algorithm escapes python's negative indexing
				D[i][i-1] = -1.0 * factor if (i-1 >= 0) else 0.0 

			except IndexError:
				pass

		return D
				
	# Build second derivative operator
	def DD(self):

		DD = np.zeros( (self.size, self.size) )

		# Loop through indices 
		for i in range(self.size):

			factor = 1.0 / ( self.delta**2 )

			# Diagonal elemnts
			DD[i][i] = -2.0 * factor

			# Tridiagonal elements (+)
			try: 
				DD[i][i+1] = 1.0 * factor 
		
			except IndexError:				
				pass

			# Tridiagonal elements (-)
			try:	
				# Algorithm escapes python's negative indexing
				DD[i][i-1] = 1.0 * factor if (i-1 >= 0) else 0.0 

			except IndexError:
				pass

		return DD
		

# Class to construct boundary conditions
class BoundaryConditions:

	def __init__(self, size, delta = 1.0):
		
		# Store matrix size
		self.size = size

		# Store the differential scale
		self.delta = delta

		# Boundary conditions "operator"
		self.B = np.zeros( (self.size, self.size) )

		# Initialze
		self.b0 = self.init()
		self.bN = self.init()

	def init(self):
	
		return {
			"vec"	: np.zeros( self.size ),
			"val" 	: np.nan, 
			"set" 	: False,
			"type" 	: None 
		}

	# A Dirchlet boundary condition requires the solution to the differential 
	# equation at the endpoint to be equal to a given value.
	def addDirchlet(self, _v, pos):

		# Calculate boundary condition value
		val = -1.0 * float(_v) / self.delta**2

		# Update subvectors and "operator"
		if pos in ["0"]:

			vec = [ ( val if i == 0 else 0.0 ) for i in range(self.size) ]

			# Subvector
			self.b0["vec"] = np.array(vec)
			self.b0["val"] = val
			self.b0["set"] = True
			self.b0["type"] = "Dirchlet"

			# Operator
			self.B[0][1] = 0.0

		if pos in ["N", "n"]:

			vec = [ ( val if i == (self.size -1) else 0.0 ) for i in range(self.size) ]

			# Subvector
			self.bN["vec"] = np.array(vec)
			self.bN["val"] = val
			self.bN["set"] = True
			self.bN["type"] = "Dirchlet"

			# Operator
			self.B[self.size-1][self.size-2] = 0.0

	# A Newmann boundary condition requires the derivative of the solution to 
	# the differential equation at the endpoint to be equal to a given value.
	def addNewmann(self, _v, pos):

		# Calculate boundary condition value
		val = -2.0 * float(_v) / self.delta

		# Update subvectors and "operator"
		if pos in ["0"]:

			vec = [ ( val if i == 0 else 0.0 ) for i in range(self.size) ]

			# Subvector
			self.b0["vec"] = np.array(vec)
			self.b0["val"] = val
			self.b0["set"] = True
			self.b0["type"] = "Newmann"

			# Operator
			self.B[0][1]  = 1.0 / self.delta**2

		if pos in ["N", "n"]:

			vec = [ ( val if i == (self.size -1) else 0.0 ) for i in range(self.size) ]

			# Subvector
			self.bN["vec"] = np.array(vec)
			self.bN["val"] = val
			self.bN["set"] = True
			self.bN["type"] = "Newmann"

			# Operator
			self.B[self.size-1][self.size-2] = 1.0 / self.delta**2

	# Method to check if boundary condition are adequately defines
	def areValid(self):

		# Check that both boundary conditions are set
		if ( ( self.b0["set"] == False ) or
			 ( self.bN["set"] == False ) ):

			print("Invalid Boundary Conditions: Undefined")			
			return False

		# Check that user has not set a double Newmann condition
		elif ( ( self.b0["type"] == "Newmann" ) and
			   ( self.bN["type"] == "Newmann" ) ):
			
			print("Invalid Boundary Conditions: Double Newmann")
			return False 

		# Otherwise conditions are valid
		else:		
	
			return True

	# Method to construct and return the boundary condition vector
	def getVector(self):

		return np.add( self.b0["vec"], self.bN["vec"] )

	# Method to return boundary condition operator
	def getOperator(self):

		return self.B

	# Evaluate boundary conditions against a test vector (v)
	def evaluate(self, v):

		return np.subtract( np.dot(self.B, v), self.getVector() )
