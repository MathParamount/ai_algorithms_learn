# first principles
# test driven development
# criterion: self contained coded
# clue: Grok has accuratly more detailed algorithms

'''References:
[1] A. C. Lisboa. "Artificial intelligence", technical report, Gaia, gaiasd.com/AI.pdf, 2026.
'''

import numpy as np
import sys
import io
import time
import json
import copy
import math
import logging

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(precision=0, suppress=True)

# modular algebra
class modular_residue:
	# Construction.
	def __init__(self, r, M, b=None):
		''' Construct a modular residue number.

		Args:
			r (numpy array of int): residue
			M (int): modulus
			b (bool, optional): modulus is prime indicator (default is a prime predicate)

		Returns:
			(modular_residue): a modular residue number
		'''

		# Setup.
		self._modulus = M
		self._residue = r
		self._is_prime = is_prime(M) if b is None else b

	# Read only attributes.
	def residue(self):
		return self._residue

	def dtype(self):
		return str(self._residue.dtype)

	def modulus(self):
		return self._modulus

	def is_prime(self):
		return self._is_prime

	# Operators.
	def __setattr__(self,name,value):
		# Error check.
		if name == '_residue':
			if not isinstance(value,np.ndarray) or str(value.dtype) not in ['uint8', 'uint16', 'uint32', 'uint64']:
				raise ValueError('Type for residue must be unsigned integer')
			value %= self._modulus
		elif name == '_modulus':
			if not isinstance(value,(int,np.uint8,np.uint16,np.uint32,np.uint64)):
				raise ValueError('Modulus must be an integer')
		elif name == '_is_prime':
			if not isinstance(value,bool):
				raise ValueError('Is prime indicator must be a boolean')
		else:
			raise ValueError('Non existing attribute name \'{}\''.format(name))
		
		# Setup.
		super().__setattr__(name,value)

	def __add__(self, other):
		# Error check.
		if isinstance(self,modular_residue) and isinstance(other,modular_residue) and self._modulus != other._modulus:
			raise ValueError('Operand modulus must be the same')
		
		# Extract info.
		bp = self._is_prime if isinstance(self,modular_residue) else other._is_prime # is prime indicator
		a = self._residue if isinstance(self,modular_residue) else self # first operand
		b = other._residue if isinstance(other,modular_residue) else other # second operand
		M = self._modulus if isinstance(self,modular_residue) else other._modulus # modulus

		# Modular sum.
		return modular_residue(a+b, M, bp)

	def __radd__(self,other):
		return self.__add__(other)

	def __neg__(self):
		return modular_residue(self._modulus-self._residue, self._modulus, self._is_prime)

	def __sub__(self, other):
		# Error check.
		if isinstance(other,modular_residue) and self._modulus != other._modulus:
			raise ValueError('Operand modulus must be the same.')
		
		# Extract info.
		bp = self._is_prime if isinstance(self,modular_residue) else other._is_prime # is prime indicator
		a = self._residue if isinstance(self,modular_residue) else self # first operand
		b = other._residue if isinstance(other,modular_residue) else other # second operand
		M = self._modulus if isinstance(self,modular_residue) else other._modulus # modulus

		# Modular sum.
		return modular_residue(a+M-b, M, bp)

	def __rsub__(self,other):
		return -(self.__sub__(other)) # there is some switch flip in this overload...

	def __mul__(self, other):
		# Error check.
		if isinstance(other,modular_residue) and self._modulus != other._modulus:
			raise ValueError('Operand modulus must be the same')
		
		# Extract info.
		bp = self._is_prime if isinstance(self,modular_residue) else other._is_prime # is prime indicator
		a = self._residue if isinstance(self,modular_residue) else self # first operand
		b = other._residue if isinstance(other,modular_residue) else other # second operand
		M = self._modulus if isinstance(self,modular_residue) else other._modulus # modulus

		# Modular multiplication.
		return modular_residue(a*b, M, bp)

	def __rmul__(self, other):
		return self*other

	def __matmul__(self, other):
		# Error check.
		if isinstance(other,modular_residue) and self._modulus != other._modulus:
			raise ValueError('Operand modulus must be the same')
		if len(self._residue.shape) != 2 or len(other._residue.shape) != 2:
			raise ValueError('Operands must be matrices')
		
		# Extract info.
		bp = self._is_prime if isinstance(self,modular_residue) else other._is_prime # is prime indicator
		a = self._residue if isinstance(self,modular_residue) else self # first operand
		b = other._residue if isinstance(other,modular_residue) else other # second operand
		M = self._modulus if isinstance(self,modular_residue) else other._modulus # modulus

		# Modular multiplication.
		return modular_residue(a@b, M, bp)

	def __mod__(self, other):
		# Error check.
		if not isinstance(self,modular_residue):
			raise TypeError('Operand must be modular residues')
		if not isinstance(other,(int,np.uint8,np.uint16,np.uint32,np.uint64)):
			raise TypeError('Modulus must be an integer')

		# Extract info.
		a = self._residue # residue
		M = other # modulus

		# New modulus
		return modular_residue(a%M, M, is_prime(M))

	def __pow__(self, other):
		# Error check.
		if isinstance(other,modular_residue) and self._modulus != other._modulus:
			raise ValueError('Operand modulus must be the same')
		if isinstance(other,np.ndarray) and other.dtype != self._residue.dtype:
			other.astype(self._residue.dtype)
		
		# Extract info.
		bp = self._is_prime if isinstance(self,modular_residue) else other._is_prime # is prime indicator
		a = self._residue if isinstance(self,modular_residue) else self # first operand
		b = other._residue if isinstance(other,modular_residue) else other # second operand
		M = self._modulus if isinstance(self,modular_residue) else other._modulus # modulus
		s = a.shape if not isinstance(b,np.ndarray) or sum(a.shape) > sum(b.shape) else b.shape # shape
		dtype = str(a.dtype) if isinstance(a,np.ndarray) else str(b.dtype)
		if dtype == 'uint8':
			pdtype = 'uint16'
		elif dtype == 'uint16':
			pdtype = 'uint32'
		elif dtype == 'uint32':
			pdtype = 'uint64'
		else:
			pdtype = 'uint64'

		# Efficient power.
		if M==1:
			r = np.zeros(s,dtype=dtype)
		elif self._is_prime:
			r = (a ** (b % (M-1))) % M
		else:
			r = np.ones(s,dtype=dtype) # power
			ar = (a%M)*np.ones(s,dtype=dtype)# power basis
			br = b*np.ones(s,dtype=dtype) # power exponent
			while np.any(br > 0):
				ib = np.where((br % 2) > 0)[0]
				r[ib] = (r[ib].astype(pdtype)*ar[ib].astype(pdtype) % M).astype(dtype)
				ar = (ar.astype(pdtype)*ar.astype(pdtype) % M).astype(dtype)
				br //= 2
		r[np.logical_and(a==0,b==0)] = 1
		r[np.logical_and(a==0,b!=0)] = 0
		return modular_residue(r, M, bp)

	def __rpow__(self, other):
		if isinstance(other,int):
			other = modular_residue(np.array([other],dtype=self._residue.dtype), self._modulus, self._is_prime)
		else:
			raise TypeError('Invalid basis type')
		return other.__pow__(self) # there is some switch flip in this overload...

	def __truediv__(self, other):
		# Error check.
		if isinstance(other,modular_residue) and self._modulus != other._modulus:
			raise ValueError('Operand modulus must be the same')
		if isinstance(other,np.ndarray) and other.dtype != self._residue.dtype:
			other.astype(self._residue.dtype)
		
		# Extract info.
		bp = self._is_prime if isinstance(self,modular_residue) else other._is_prime # is prime indicator
		a = self._residue if isinstance(self,modular_residue) else self # first operand
		b = other._residue if isinstance(other,modular_residue) else other # second operand
		M = self._modulus if isinstance(self,modular_residue) else other._modulus # modulus
		s = a.shape if not isinstance(b,np.ndarray) or sum(a.shape) > sum(b.shape) else b.shape # shape
		ns = len(s) # tensor dimension
		if a.shape != b.shape:
			st = np.array([1]*ns,dtype=int)
			for ins in range(ns):
				if a.shape[ins] != b.shape[ins]:
					ds = np.array([1]*ns,dtype=int)
					if a.shape[ins]==1:
						ds[ins] = b.shape[ins]
						a = np.tile(a,tuple(ds))					
					elif b.shape[ins]==1:
						ds[ins] = a.shape[ins]						
						b = np.tile(b,tuple(ds))					
					else:
						raise ValueError('Inconsistent operant sizes')
		dtype = a.dtype if isinstance(a,np.ndarray) else b.dtype
		n = np.prod(s)

		# Efficient divide: Euclidean algorithm.
		bm = np.vstack(((b.reshape(-1)%M).astype('int64'), np.tile(np.array([M],dtype='int64'),(n)))) # put in range [0, M-1]
		x = np.vstack((np.zeros((n),dtype='int64'),np.ones((n),dtype='int64')))
		iin = np.where(bm[1,:]!=0)[0]
		while sum(iin.shape) > 0:
			# Euclidean expansion.
			q = bm[0,iin] // bm[1,iin] # quotient
			bm[:,iin] = np.vstack((bm[1,iin], bm[0,iin]%bm[1,iin]))
			x[:,iin] = np.vstack((x[1,iin]-q*x[0,iin],x[0,iin]))
			iin = iin[bm[1,iin]!=0]

		# Error check.
		if M!=1 and np.any(np.logical_and(bm[0,:]!=1,b.reshape(-1)!=0)):
			raise ValueError('Divisor and modulus must be coprime')

		# Output arguments.
		x[1,x[1,:]<0] += M
		return modular_residue((np.array([a],dtype=dtype) if isinstance(a,int) else a)*x[1,:].astype(b.dtype).reshape(s), M, bp)

	def __rtruediv__(self, other):
		if isinstance(other,int):
			other = np.array([other],dtype=self._residue.dtype)
		if isinstance(other,np.ndarray):
			return modular_residue(other, self._modulus, self._is_prime) / self._residue # there is some switch flip in this overload...
		else:
			raise TypeError('Not implemented operator for these operand types')

	def __floordiv__(self, other):
		return self.__truediv__(other)

	def __rfloordiv__(self, other):
		return self.__floordiv__(other) # there is some switch flip in this overload... and every non-commutative operator

	def __eq__(self, other):
		a = self._residue if isinstance(self,modular_residue) else self # first operand
		b = other._residue if isinstance(other,modular_residue) else other # second operand
		return a == b

	def __ne__(self, other):
		a = self._residue if isinstance(self,modular_residue) else self # first operand
		b = other._residue if isinstance(other,modular_residue) else other # second operand
		return a != b

	def __lt__(self, other):
		a = self._residue if isinstance(self,modular_residue) else self # first operand
		b = other._residue if isinstance(other,modular_residue) else other # second operand
		return a < b

	def __le__(self, other):
		a = self._residue if isinstance(self,modular_residue) else self # first operand
		b = other._residue if isinstance(other,modular_residue) else other # second operand
		return a <= b

	def __gt__(self, other):
		a = self._residue if isinstance(self,modular_residue) else self # first operand
		b = other._residue if isinstance(other,modular_residue) else other # second operand
		return a > b

	def __ge__(self, other):
		a = self._residue if isinstance(self,modular_residue) else self # first operand
		b = other._residue if isinstance(other,modular_residue) else other # second operand
		return a >= b

	def __getitem__(self, key):
		return modular_residue(self._residue[key], self._modulus, self._is_prime)

	def __setitem__(self, key, value):
		value = value._residue if isinstance(value,modular_residue) else value
		self._residue[key] = value

	def __str__(self):
		d = int(np.ceil(np.log10(self._modulus)))
		fmt = 'mod\n{{:{:d}d}}'.format(d+1)
		if self._is_prime:
			fmt += '   (prime)'
		fmt += '   ({})'.format(str(self._residue.dtype))
		return print_to_string(self._residue) + fmt.format(self._modulus)
	
	def __repr__(self):
		return str(self).replace('\n',',').replace(',mod,',' mod')
	
	def tile(self,s):
		r = np.tile(self._residue,s)
		return modular_residue(r, self._modulus, self._is_prime)
	
	def copy(self):
		return modular_residue(self._residue.copy(), self._modulus, self._is_prime)
	
	def shape(self):
		return self._residue.shape
	
	def len(self):
		return len(self._residue)

	def reshape(self,*args):
		r = self._residue.copy()
		return modular_residue(r.reshape(*args), self._modulus, self._is_prime)
	
	def hstack(*args):
		return modular_residue(np.hstack(tuple([arg._residue for arg in args])), args[0]._modulus, args[0]._is_prime)
	
	def vstack(*args):
		return modular_residue(np.vstack(tuple([arg._residue for arg in args])), args[0]._modulus, args[0]._is_prime)
	
	def argmax(self,*args):
		return np.argmax(self._residue,*args)
	
	def min(self,*args):
		if args:
			return modular_residue(np.min(self._residue,*args),self._modulus,self._is_prime)
		else:
			return np.min(self._residue)
	
	def max(self,*args):
		if args:
			return modular_residue(np.max(self._residue,*args),self._modulus,self._is_prime)
		else:
			return np.max(self._residue)
	
	def transpose(self):
		return modular_residue(np.transpose(self._residue),self._modulus,self._is_prime)
	
	def astype(self,dtype):
		return modular_residue(self._residue.astype(dtype),self._modulus,self._is_prime)
	
	def enumerate(self,index=0):
		return enumerate(self._residue,index)

def print_to_string(*args, **kwargs):
    output = io.StringIO()
    print(*args, file=output, **kwargs)
    contents = output.getvalue()
    output.close()
    return contents



# Spaces.
def truth_space(S,M,dtype='uint64'):
	# Every possible arange of S samples in modulus M, for universality and tightness tests (complete output space).
	y = np.zeros((0,1))
	for i in range(S):
		y = np.vstack((np.tile(np.arange(0,M).reshape(-1,1),(1,M**i)).reshape(-1),np.tile(y,(1,M))))
	return modular_residue(y.astype(dtype),M)

def polynomial_space(x,E,S):
	# Power of modular residue samples at given exponents, for single input training.
	P = modular_residue(np.tile(x.residue().reshape(-1,1),(1,S)),x.modulus(),x.is_prime())
	if isinstance(E,modular_residue):
		E = E.residue()
	for i, e in enumerate(E):
		if i > 0 and E[i] == E[i-1]:
			P[:,i] *= P[:,i-1]
		else:
			P[:,i] **= e
	return P



# Training.
def mles_solve(A,b):
	# Solve modular linear equation system Ax = b using Gaussian elimination.

	# Error check.
	if not isinstance(A,modular_residue) or not isinstance(b,modular_residue):
		raise TypeError('Tensors must be modular residues')
	if len(A.shape()) != 2 or len(b.shape()) != 2:
		raise ValueError('Tensors in linear system must be matrices')
	if A.shape()[0] != b.shape()[0]:
		raise ValueError('Matrices must have the same number of rows')

	# Cardinalities.
	M,N = A.shape() # number of rows and inputs
	if M > N:
		raise ValueError('Over determined system of equations')
	M,O = b.shape() # number of rows and outputs

	# Gauss elimination algorithm.
	T = A.hstack(b) # tableau
	p = np.arange(0,M) # pivot rows
	for k in range(M):
		i = k + T[p[k:],k].argmax() # pivot row
		p[k], p[i] = p[i], p[k] # swap pivot to current iteratino position
		T[p[k],k:] /= T[p[k],k:k+1] # normalize pivot row
		T[p[k+1:],k:] -= T[p[k]::M,k:]*T[p[k+1:],k:k+1] # elimination

	# Backward substitution.
	T = T[p,:] # sort tableau
	x = T[:,N:] if M==N else T[:,N:].vstack(modular_residue(np.zeros((N-M,O),dtype=A.dtype),A.modulus(),A.is_prime())) # initialize solution
	for k in range(0,M-1):
		x[0:M-k-1,:] -= T[0:M-k-1,M-k-1:M-k]*x[M-k-1:M-k,:]
	return x

def error_reduction_direct_search(xh,yh,G,M,X,g=None,gmin=0,binary=False,exhaustive=False):
	S, N = xh.shape # number of samples and inputs
	dyb, gb, yhb = 0, np.array([],dtype=int), np.zeros((S,1),xh.dtype) # output arguments
	if np.any(yh == 0) and np.all(xh==xh[0:1,:]): # trivial case: no solution
		return dyb, gb, yhb
	if g is None:
		g = np.arange(G,dtype=int)  # input group indexation
		g[-1] = max(g[-1],gmin)
	ng = nchoosek(N-g[0],G) # number of combinations
	if ng == 0:
		return dyb, gb, yhb
	logger.info('searching error reduction among {:9d} combinations for {:6d} inputs'.format(ng,N))
	while True:
		# Test current input group.
		yhe = prune_error(group(xh[:,g],M,X,binary).residue(),yh)
		dy = np.sum(yhe)
		if dy > dyb:
			dyb = dy # update error reduction
			gb = g.copy() # update input group index
			yhb = yhe.copy() # update output error
			if not exhaustive:
				break

		# Increment input group.
		jg = G-1
		while jg >= 0 and g[jg]==N-G+jg:
			jg -= 1
		if jg < 0:
			break
		g[jg] += 1
		for kg in range(jg+1,G):
			g[kg] = g[kg-1] + 1
		g[-1] = max(g[-1],gmin)
	return dyb, gb, yhb

def error_reduction_search(xh,yh,G,M,X,g=None,gmin=0,binary=False,batch=True):
	S, N = xh.shape # number of samples and inputs
	dyb, gb, yhb = 0, np.array([],dtype=int), np.zeros((S,1),xh.dtype) # output arguments
	bz = yh == 0 # output zero indicator
	Z = np.where(bz)[0] # zero output indexes
	P = np.where(np.logical_not(bz))[0] # positive output indexes
	if Z.size > 0 and np.all(xh==xh[0:1,:]): # trivial case: no solution
		return dyb, gb, yhb
	Xp = np.cumprod(np.append(1,np.tile(X,G-1))).astype(xh.dtype)
	if g is None:
		g = np.arange(G,dtype=int)  # input group indexation
		g[-1] = max(g[-1],gmin)
	ng = nchoosek(N-g[0],G) # number of combinations
	if ng == 0:
		return dyb, gb, yhb
	mp = P.shape[0] # number of positive outputs
	logger.info('searching error reduction among {:9d} combinations for {:6d} inputs'.format(ng,N))
	for ip, p in enumerate(P):
		gp = g.copy() # group input indexes
		ig = 0
		while ig < ng:
			if batch:
				if Z.size == 0 or N == 1:
					gb = gp
					yhb = prune_error(group(xh[:,gb],M,X).residue(),yh)
					dyb = np.sum(yhb)
					break
				else:
					glast = np.arange(gp[-2]+1,N,dtype=int)
					i = np.array([0],int)
					gp[-1] = N-1
					i = np.where(np.all((xh[p,gp[0:-1]]@Xp[0:-1] + xh[p,glast]*Xp[-1]).reshape(1,-1) != (xh[:,gp[0:-1]][Z,:]@Xp[0:-1]).reshape(-1,1) + xh[:,glast][Z,:]*Xp[-1],axis=0))[0]
					if i.size > 0:
						gb = np.append(gp[0:-1],glast[i[0]])
						yhb = prune_error(group(xh[:,gb],M,X).residue(),yh)
						dyb = np.sum(yhb)
						break
			else:
				# Test current input group.
				if Z.size == 0 or np.all(xh[p,gp]@Xp != xh[:,gp][Z,:]@Xp):
					gb = gp
					yhb = prune_error(group(xh[:,gb],M,X).residue(),yh)
					dyb = np.sum(yhb)
					break

			# Increment input group.
			ig += N-gp[-2]-1 if batch else 1
			jg = G-1-batch
			while jg >= 0 and gp[jg]==N-G+jg:
				jg -= 1
			if jg < 0:
				break
			gp[jg] += 1
			for kg in range(jg+1,G-batch):
				gp[kg] = gp[kg-1] + 1
			if not batch:
				gp[-1] = max(gp[-1],gmin)
		logger.debug('{:6d} of {:6d} positive values explored'.format(ip,mp))
		if dyb > 0:
			break
	return dyb, gb, yhb

def fit(xh,yh,M=None,X=None,G=2,Gp=None,Np=None,Mp=0,Eh=None,dtype=None,incremental=False,
		searcher=error_reduction_search,file='',log_level=logging.WARNING):
	''' Fit input to output using modular artificial intelligence model.

	Args:
		xh (numpy array or moduluar_residue): input samples
		yh (numpy array or moduluar_residue): output samples
		M (int, optional): specify modulus (default None)
		X (tuple of int, optional): specify input range (default None)
		G (int, optional): input group size (default 2)
		Gp (int or numpy array of int, optional): input group tensor resize (default G)
		Np (numpy array of int, optional): input tensor size (default np.array([xh.shape[1]],int))
		Mp (int, optional): reduced modulus (default 0)
		Eh (tuple of numpy array or modular_residue, optional): specify probing exponents (default None)
		dtype (str, optional): specify signed integer data type
		incremental (bool, optional): incremental training indicator (default False)
		searcher (function, optional): error reduction search function (default error_reduction_search)
		file (str, optional): intermediate model json-file (default '')
		log_level (level, optional) logging level (default logging.WARNING)

	Retrun:
		(dict): modular artificial intelligence with fields:
			bits (int): memory usage
			dtype (str): data type
			exponents (moduluar_residue): probing exponents for training space
			input (dict): input configuration with fields:
				encoder (function): input encoder
				minimum (numpy array): input minimum values
				maximum (numpy array): input minimum values
				size (tuple of int): input size
			layers (dict): layer info with fields:
				inputs (list of list of numpy array of int): input indexes
				refined (list of numpy array of bool): refined input indicator
			maximum_refinement (int): maximum refinement level
			model (string): AI model
			modulus (int): modulus
			modulus_reduced (int): reduced modulus
			output (dict): output configuration with fields:
				decoder (function): output decoder
				minimum (numpy array): output minimum values
				maximum (numpy array): output minimum values
				size (tuple of int): output size
			parameters (modular_residue): parameters for inference
			samples (int): number of samples
			spacer (function): create space for training
			time (float): elapsed training time [s]
			training (dict): training info with fields:
				error (modular_residue): output error for each sample
				samples (int): number of samples used in training
				time (float): training time [s]
	'''

	# Error check.
	logger.setLevel(log_level) # set logging level
	if not isinstance(xh,(np.ndarray,modular_residue)):
		raise TypeError('Input must be a numpy array or a modular residue')
	if not isinstance(yh,(np.ndarray,modular_residue)):
		raise TypeError('Output must be a numpy array or a modular residue')
	if Eh is not None and not isinstance(Eh,(np.ndarray,modular_residue)):
		raise ValueError('Probing exponents must be in a tuple of either numpy arrays or modular residues')
	sx = xh.shape if isinstance(xh,np.ndarray) else xh.shape()
	if not sx:
		raise ValueError('Empty input data not suppoted')
	sy = yh.shape if isinstance(yh,np.ndarray) else yh.shape()
	if not sy:
		raise ValueError('Empty output data not suppoted')
	if sx[0] != sy[0]:
		raise ValueError('Inconsistent number of input and output samples')
	if M is not None and not isinstance(M,int):
		raise TypeError('Modulus must be an integer')
	if M is not None and M < 1:
		raise ValueError('Modulus must be a positive integer')
	if isinstance(xh,modular_residue) and isinstance(yh,modular_residue) and xh.modulus() != yh.modulus():
		raise ValueError('Input and output modulus must be the same')
	if X is not None and (not isinstance(X,(tuple,list,np.ndarray)) or not all([isinstance(x,int) for x in X])):
		raise TypeError('Input range must be a tuple of integers')

	# Cardinalities.
	if len(sx) == 1 or sx[1]==1:
		# Scalar model.
		Np = np.array([1],int) # number of input variables
		O = (1,) if len(sy)==1 else (sy[1],) # number of input variables
		xh = xh if len(sx)==2 else xh.reshape(-1,1) # canonical output form
		yh = yh if len(sy)==2 else yh.reshape(-1,1) # canonical output form
		S = sx[0] # number of samples
		model = 'MAI' # AI model
	elif len(sx) == 2:
		# Vector model.
		S = sx[0] # number of samples
		if Np is None:
			Np = np.array([sx[1]],int) # number of input variables
		elif np.prod(Np) != sx[1]:
			raise ValueError('Invalid input tensor size.')
		O = (1,) if len(sy)==1 else (sy[1],) # number of input variables
		yh = yh if len(sy)==2 else yh.reshape(-1,1) # canonical output form
		model = 'VMAI' if len(Np.shape)==1 else 'CMAI' # AI model
	else:
		raise ValueError('Unsupport data')
	if M is None:
		if isinstance(xh,modular_residue):
			M = xh.modulus()
			xmin = np.array([xh[:,n].min() for n in range(Np[0])]) # minimum input value
			xmax = np.array([xh[:,n].max() for n in range(Np[0])]) # maximum input value
			if isinstance(yh,modular_residue):
				ymin = np.array([yh[:,o].min() for o in range(O[0])]) # minimum output value
				ymax = np.array([yh[:,o].max() for o in range(O[0])]) # maximum output value
			else:
				ymin = np.array([np.min(yh[:,o]) for o in range(O[0])]) # minimum output value
				ymax = np.array([np.max(yh[:,o]) for o in range(O[0])]) # maximum output value
		elif isinstance(yh,modular_residue):
			M = yh.modulus()
			xmin = np.array([np.min(xh[:,n]) for n in range(Np[0])]) # minimum input value
			xmax = np.array([np.max(xh[:,n]) for n in range(Np[0])]) # maximum input value
			ymin = np.array([yh[:,o].min() for o in range(O[0])]) # minimum output value
			ymax = np.array([yh[:,o].max() for o in range(O[0])]) # maximum output value
		else:
			xmin = np.array([np.min(xh[:,n]) for n in range(Np[0])]) # minimum input value
			xmax = np.array([np.max(xh[:,n]) for n in range(Np[0])]) # maximum input value
			ymin = np.array([np.min(yh[:,o]) for o in range(O[0])]) # minimum output value
			ymax = np.array([np.max(yh[:,o]) for o in range(O[0])]) # maximum output value
			M = next_prime(max(S,2**G,np.max(xmax+1),np.max(ymax+1))) # modulus
	elif not is_prime(M):
		raise ValueError('Modulus must be a prime number')
	if S > M:
		raise ValueError('Too many samples')
	if X is None:
		X = xmax + 1 # input range
	binary = G==2 and Mp==2 and np.all(xmin==0) and np.all(ymin==0) and np.all(xmax==1) and np.all(ymax==1) # binary data indicator
	if Gp is None:
		Gp = G # input group tensor resize

	# Type definition.
	if not isinstance(xh,modular_residue):
		xh = modular_residue(xh,M,True)
	if np.unique(xh.residue(),axis=0).shape[0] < S:
		raise ValueError('Input must be unique')
	if not isinstance(yh,modular_residue):
		yh = modular_residue(yh,M,True)
	if dtype is None:
		dtype = np.uint64
		if M < 2**7:
			dtype = np.uint8
		elif M < 2**15:
			dtype = np.uint16
		elif M < 2**31:
			dtype = np.uint32
	elif dtype not in ['uint8', 'uint16', 'uint32', 'uint64']:
		raise ValueError('Data type must be unsigned integer')
	M = dtype(M)
	Mp = dtype(Mp)
	xh = xh.astype(dtype) % M
	yh = yh.astype(dtype) % M
	xmin = xmin.astype(dtype)
	xmax = xmax.astype(dtype)
	ymin = ymin.astype(dtype)
	ymax = ymax.astype(dtype)
	rmax = dtype(0) if Mp == 0 or Mp >= M else dtype(np.ceil(math.log(max([np.max(xmax),np.max(ymax)])+1,Mp)))
	if not isinstance(X,np.ndarray):
		X = np.array(X,dtype=dtype)
	else:
		X = X.astype(dtype)

	# Polynomial powers.
	if Eh is None:
		# Exponent definition.
		Eh = modular_residue(np.arange(S,dtype=dtype),M,True)
	else:
		# Error check.
		E = Eh % M
		if E.shape[0] != S:
			raise ValueError('There must be a unique probing exponent for each sample')
		Eu = np.unique(E)
		if Eu.shape[0] != E.shape[0]:
			raise ValueError('Probing exponents must be unique in modulus')
		Eh = modular_residue(E.astype(dtype),M,True)
	
	# Training.
	timing = time.perf_counter(); # starting training time
	lmax = math.log(M-1,G) # maximum refinement layer
	Xh = [modular_residue(np.zeros((S,0),dtype),M,True)] # layer inputs
	i = [[]] # input indexes
	t = [np.array([],bool)] # input type
	lw = np.array([],int) # parameter layer
	kl = 0 # layer count
	nl = 0 # number of layers
	ks = 0 # refinent step
	ns = max(1,rmax) # number of refinent steps
	nr = int(np.ceil(math.log(max(Np),G))) # number of resizes
	kr = 0 if incremental else nr # current resize
	ey = yh.copy() # current output error
	error = np.sum(ey.residue()) # current error
	w = modular_residue(np.zeros((S,0),dtype=dtype),M,True) # parameters
	Ni = Xh[kl].shape()[1] # number of current layer inputs
	xhr = xh.residue() # resized input
	Nir = xhr.shape[1] # number of resized inputs
	gmin = np.zeros(1,int) # minimum group input index
	glast = np.zeros(1,int) # last group input index
	logger.info('{} samples in modulo {}'.format(S,M))
	while error > 0 and kr <= nr:
		# Next layer.
		logger.info('error of {:6d} for layer {:2d} of {:2d}, step {:2d} of {:2d}, resize {:2d} of {:2d}'.format(error,kl+1,nl,ks+1,ns,kr,nr))
		if kl == 0:
			if ks == 0:
				xhr = resize(xh.residue(),Np,kr,Gp)
			Nir = xhr.shape[1] # number of resized inputs
			if rmax > 0:
				Xh[0] = Xh[0].hstack(modular_residue(refine(xhr,Mp,ks,ns),M,True))
			else:
				Xh[0] = Xh[0].hstack(modular_residue(xhr,M,True))
			i[0] += [np.array([Ni+i],int) for i in range(Nir)] # input indexes
			t[0] = np.append(t[0],np.ones(Nir,bool)) # input type
			Ni = Xh[kl].shape()[1] # number of current layer inputs
		gmax = min(Ni,G) # current input group size
		X = subsizes(M,gmax) # input range in group
		if kl >= nl:
			Xh.append(modular_residue(np.zeros((S,0),dtype=dtype),M,True))
			i.append([]) # new layer input indexes
			t.append(np.array([],bool)) # append input type
			gmin = np.append(gmin, 0) # append minimum group input index
			glast = np.append(glast, 0) # append last group input index
			nl += 1
		if kl < lmax:
			Xh[kl+1] = Xh[kl+1].hstack(Xh[kl][:,gmin[kl]:]*G)
			i[kl+1] += [np.array([i],int) for i in range(gmin[kl],Xh[kl].shape()[1])]
			t[kl+1] = np.append(t[kl+1], np.ones(Xh[kl].shape()[1]-gmin[kl],bool))
		xs = scale(Xh[kl].residue(),M,X) # scale previous layer inputs
		g = np.arange(glast[kl],gmax+glast[kl],dtype=int) # input group indexation
		g[-1] = max(g[-1],gmin[kl])
		kl += 1 # update layer counter
		while True:
			# Error reduction.
			dy, g, eye = searcher(xs,ey.residue(),gmax,M,X,g,gmin[kl-1],binary)
			if dy == 0:
				break
			xhe = group(xs[:,g],M,X,binary) # group encoding
			eye = modular_residue(eye,M,True)
			V = polynomial_space(xhe,Eh,S) # Vandermonde matrix
			logger.info('solving {} x {} modular linear equation system to reduce error in {}'.format(V.shape()[0],V.shape()[1],dy))
			w = w.hstack(mles_solve(V,eye)) # training itself
			if np.any(V@w[:,-1:] != eye):
				logger.error('Invalid solution of modular linear equation system.')
			lw = np.append(lw,kl) # append layer of parameter
			ey -= eye # update output error
			Xh[kl] = Xh[kl].hstack(eye) # append input for next layer
			i[kl].append(g.copy())
			t[kl] = np.append(t[kl], False)
			error = np.sum(ey.residue())
			logger.info('error reduced to {:6d}'.format(error))
			if file:
				save_model({'model': model, 'dtype': dtype,
					'modulus': M, 'modulus_reduced': Mp, 'group_size': G,
					'resized': incremental, 'binary': binary,
					'maximum_refinement': rmax,
					'bits': int(np.ceil(2*np.sum(w>0)*math.log(float(M),2))),
					'parameters': w[:,np.argsort(lw,kind='stable')], 'exponents': Eh,
					'layers': prune_input({'inputs': i, 'refined': t}),
					'input': {'minimum': xmin, 'maximum': xmax, 'size': Np},
					'output': {'minimum': ymin, 'maximum': ymax, 'size': O},
					'training': {'error': ey, 'samples': S, 'timing': round(time.perf_counter() - timing,6)}
					},
					file,
					xh,yh
				)
			if error == 0:
				break
		gmin[kl-1] = Xh[kl-1].shape()[1] # number of current layer inputs
		Ni = Xh[kl].shape()[1] # number of next layer inputs
		if Ni == 0:
			kl = 0 # reset layer count
			ks += 1 # increment refinement step
			if ks >= ns:
				ks = 0
				kr += 1
				for il in range(nl):
					glast[il] = Xh[il].shape()[1] # reset last group input index
			Ni = Xh[kl].shape()[1] # number of next layer inputs

	# Output arguments.
	return {'model': model, 'dtype': dtype,
		'modulus': M, 'modulus_reduced': Mp, 'group_size': G,
		'resized': incremental, 'binary': binary,
		'maximum_refinement': rmax,
		'bits': int(np.ceil(2*np.sum(w>0)*math.log(float(M),2))),
		'parameters': w[:,np.argsort(lw,kind='stable')], 'exponents': Eh,
		'layers': prune_input({'inputs': i, 'refined': t}),
		'input': {'minimum': xmin, 'maximum': xmax, 'size': Np},
		'output': {'minimum': ymin, 'maximum': ymax, 'size': O},
		'training': {'error': ey, 'samples': S, 'timing': round(time.perf_counter() - timing,6)}
	}

def encode(x,M,N,X,dtype):
	x = x.astype(dtype) if isinstance(x,modular_residue) else modular_residue(x.astype(dtype),M,True)
	if len(x.shape()) == 1:
		x = x.reshape(-1,1)
	return modular_residue(
		x,
		x.modulus(),
		x.is_prime()
	).astype(dtype)

def decode(y):
	return y

def nchoosek(n,k):
	# Trivial cases.
	if k < 0 or k > n:
		return 0
	if k == 0 or k == n:
		return 1
	
	# General case.
	k = min(k, n - k)
	c = 1
	for i in range(k):
		c *= (n - i)
		c //= (i + 1)
	return c

def prune_error(xh,yh):
	_, iu, ju = np.unique(xh,return_index=True,return_inverse=True)
	yhe = yh.copy()
	for i in range(iu.shape[0]):
		yhe[ju==i] = np.min(yh[ju==i])
	return yhe

def prune_input(layers):
	layers = copy.deepcopy(layers)
	nl = len(layers['inputs']) # number of layerss
	map = [[] for il in range(nl)]
	use = [[] for il in range(nl)]
	for il in range(nl,0,-1):
		use[il-1] = np.logical_not(layers['refined'][il-1])
		if il < nl and layers['inputs'][il]:
			use[il-1][np.hstack(tuple(layers['inputs'][il]))] = True
		map[il-1] = np.cumsum(use[il-1]) - 1
		if il < nl and layers['inputs'][il]:
			for i in range(len(layers['inputs'][il])):
				layers['inputs'][il][i] = map[il-1][layers['inputs'][il][i]]
		layers['inputs'][il-1] = [i for i, b in zip(layers['inputs'][il-1],use[il-1]) if b]
		layers['refined'][il-1] = layers['refined'][il-1][use[il-1]]
	return layers

def refine(x,Mp,r,R):
	M = Mp**R
	return np.floor((x*Mp**r % M) / np.ceil(M/Mp)).astype(x.dtype)

def resize(x,Np,r,R):
	''' Resize tensor.

	Args:
		x (numpy array of int): input
		s (numpy array of int): tensor size
		r (int): resizing scale
		R (numpy array of int): input group tensor size

	Returns:
		(numpy array of int): resized input
	'''

	S, N = x.shape # number of samples and inputs
	D = Np.shape[0] # tensor dimension
	Nr = np.minimum(Np,R**r) # reduced number of inputs
	if np.all(Nr == Np):
		return x # trivial case
	N = np.prod(Np) # input size
	i = np.arange(N,dtype=int)
	ip = np.zeros(N,dtype=int)
	for k in range(D):
		ip += np.floor(np.floor((i % np.prod(Np[0:k+1])) / np.prod(Np[0:k])) * \
			Nr[k] / (1 + np.finfo(float).eps) / (Np[k] - 1)).astype(int) * \
			np.prod(Nr[0:k])
	xr = np.zeros((S,np.prod(Nr)))
	cr = np.zeros((1,np.prod(Nr)))
	for ii, iip in enumerate(ip):
		xr[:,iip] += x[:,ii]
		cr[0,iip] += 1
	xr = np.floor(xr/cr + 0.5).astype(x.dtype)
	return xr

def subsizes(M,G):
	return math.floor(M ** (1/G))

def scale(x,M,X):
	if isinstance(X,int):
		X = np.array([X],dtype=x.dtype)
	return np.floor(x/(M-1)/(1+np.finfo(float).eps)*X.reshape((1,-1))).astype(x.dtype)

def subs2ind(I,S,binary=False):
	if binary:
		if I.shape[1]==1:
			return I[:,0]
		elif I.shape[1]==2:
			return (I[:,0]==I[:,1]).astype(I.dtype)
		else:
			raise ValueError('Binary linear indexes can be obtained only for 1 and 2 subindexes.')
	else:
		o = np.cumprod(np.append(1,np.tile(S,I.shape[1]-1)),dtype=I.dtype) if isinstance(S,int) else np.cumprod(np.append(1,S[0:-1]),dtype=I.dtype)
		return np.sum(I*o.reshape(1,-1),axis=1)

def group(x,M,X,binary=False):
	return modular_residue(subs2ind(x,X,binary),M,True)

def infer(x,model):
	''' Modular artificial intelligence inference.
	
	Args:
		x (numpy array or modular_residue): input
		model (dict): MAI model

	Returns:
		(modular_residue): model output given the input
	'''

	# Convertion.
	if isinstance(x,modular_residue):
		x = x.residue()
	if len(x.shape) == 1:
		x = x.reshape(-1,1)

	# Extract parameters.
	w = model['parameters']
	i = model['layers']['inputs']
	t = model['layers']['refined']
	E = model['exponents']
	dtype = model['dtype']
	br = model['resized']
	bb = model['binary']

	# Cardinalities.
	nx, N = x.shape # number o sampes and inputs
	nl = len(model['layers']['inputs']) - 1 # number o layers
	nw = w.shape()[1] # number of parameter sets
	M = model['modulus'] # modulus
	G = model['group_size'] # group size
	Mp = model['modulus_reduced'] # reduced modulus
	Np = model['input']['size'] # input tensor size

	# Input refinement.
	ns = int(math.log(N,G)) + 1 # number of steps
	xs = []
	for ks in range((ns-1)*(not br),ns):
		xs.append(resize(x,Np,ks,G))
		if model['maximum_refinement']:
			xr = []
			for r in range(model['maximum_refinement']):
				xr.append(refine(xs[-1],Mp,r,model['maximum_refinement']))
			xs[-1] = np.hstack(tuple(xr))
	x = np.hstack(tuple(xs))

	# Inference.
	y = modular_residue(np.zeros((nx,1),dtype=model['dtype']),M,True)
	if nw > 0:
		Xs = [modular_residue(x[:,np.hstack(tuple([ii for ii, ti in zip(i[0],t[0]) if ti]))],M,True)]
		iw = 0 # parameter counter
		for il in range(nl):
			Xs.append(modular_residue(np.zeros((nx,t[il+1].shape[0]),dtype=dtype),M,True))
			indexes = tuple([ii for ii, ti in zip(i[il+1],t[il+1]) if ti])
			if indexes:
				Xs[il+1][:,t[il+1]] = Xs[il][:,np.hstack(indexes)]*G
			for ixw, ti in enumerate(t[il+1]):
				if not ti:
					X = subsizes(M,i[il+1][ixw].shape[0]) # input scaling size
					xs = scale(Xs[il][:,i[il+1][ixw]].residue(),M,X) # input scaling
					xe = group(xs,M,X,bb) # group encoder
					ie = np.where(w[:,iw].residue())[0] # exponent pruning
					V = polynomial_space(xe,E[ie],ie.shape[0]) # Vandermonde matrix
					yw = V@w[ie,iw:iw+1] # current output
					y += yw # next bit of output
					Xs[il+1][:,ixw] = yw.reshape(-1) # memory to next layer
					iw += 1 # increment parameter counter
	return y.residue()



def save_model(model,file=None,xh=None,yh=None):
	''' Convert MAI model to JSON string.

	Args:
		form (dict): model dictionary
		file (str, optional): file name to save JSON string (defaut None)

	Returns:
		(str): JSON string for the model
	'''

	# Model consistency check.
	if xh is not None and yh is not None and np.any(yh - infer(xh,model) != model['training']['error']):
		raise ValueError('Inconsistent model.')

	# Data type conversion.
	model = copy.deepcopy(model) # avoid changing model outside this function
	model['dtype'] = np.dtype(model['dtype']).name
	model['resized'] = bool(model['resized'])
	model['binary'] = bool(model['binary'])
	model['modulus'] = int(model['modulus'])
	model['modulus_reduced'] = int(model['modulus_reduced'])
	model['maximum_refinement'] = int(model['maximum_refinement'])
	model['input']['minimum'] = model['input']['minimum'].tolist()
	model['input']['maximum'] = model['input']['maximum'].tolist()
	model['input']['size'] = model['input']['size'].tolist()
	model['output']['minimum'] = model['output']['minimum'].tolist()
	model['output']['maximum'] = model['output']['maximum'].tolist()
	model['training']['error'] = model['training']['error'].residue().reshape(-1).tolist()
	del model['exponents']
	layers = []
	kl = 0
	for il, tl in zip(model['layers']['inputs'],model['layers']['refined']):
		layers.append([])
		for ii, ti in zip(il, tl):
			layers[kl].append({'layer': kl, 'inputs': ii.tolist(), 'refined': bool(ti)})
		kl += 1
	model['layers'] = layers
	parameters = []
	w = model['parameters'].residue()
	for iw in range(w.shape[1]):
		e = np.where(w[:,iw])[0]
		parameters.append({'exponents': e.tolist(), 'coefficients': w[e,iw].tolist()})
	model['parameters'] = parameters

	# Form dictionary o JSON string.
	json_string = json.dumps(model, indent=4, ensure_ascii=False)

	# Save model JSON to file.
	if file is not None:
		with open(file, 'w') as fid:
			json.dump(model, fid, indent=4, ensure_ascii=False)

		# Model file consistency check.
		if xh is not None and yh is not None:
			model = load_model(file)
			if np.any(yh - infer(xh,model) != model['training']['error']):
				raise ValueError('Inconsistent model file.')

	# Output arguments.
	return json_string

def load_model(file):
	''' Load MAI model from file.

	Args:
		file (str): file name to load JSON string into dictionary

	Returns:
		(dict): model dictionary
	'''

	# Load model dictionary from JSON file.
	with open(file, 'r') as fid:
		# Load model from file.
		model = json.load(fid)

		# Data type.
		if model['dtype'] == 'uint8':
			model['dtype'] = np.uint8
		elif model['dtype'] == 'uint16':
			model['dtype'] = np.uint16
		elif model['dtype'] == 'uint32':
			model['dtype'] = np.uint32
		elif model['dtype'] == 'uint64':
			model['dtype'] = np.uint64

		# Cardinalities.
		dtype = model['dtype'] # data type
		model['modulus'] = dtype(model['modulus'])
		model['modulus_reduced'] = dtype(model['modulus_reduced'])
		model['maximum_refinement'] = dtype(model['maximum_refinement'])
		M = model['modulus'] # modulus
		nw = len(model['parameters']) # number of parameters
		Emax = max([p['exponents'] if isinstance(p['exponents'],int) else max(p['exponents']) for p in model['parameters']])

		# parameters
		w = modular_residue(np.zeros((Emax+1,nw),dtype=dtype), M, True)
		for iw, p in enumerate(model['parameters']):
			e = np.array(p['exponents'],dtype=int)
			w[e,iw] = np.array(p['coefficients'],dtype=dtype)
		model['parameters'] = w
		model['exponents'] = modular_residue(np.arange(Emax+1,dtype=dtype),M,True)

		# Inputs.
		model['input']['size'] = np.array(model['input']['size'],int)
		i = []
		t = []
		for layer in model['layers']:
			i.append([])
			t.append(np.array([],dtype=bool))
			for input in layer:
				li = input['layer']
				ii = np.array(input['inputs'] if isinstance(input['inputs'],list) else [input['inputs']],dtype=int)
				ti = input['refined']
				i[li].append(ii)
				t[li] = np.append(t[li], ti)
		model['layers'] = {'inputs': i, 'refined': t}

		# Training info.
		model['training']['error'] = modular_residue(np.array(model['training']['error'],dtype).reshape(-1,1),M,True)

		# Output arguments.
		return model



# Prime numbers.
def is_prime(n):
# Is prime predicate.
	if n < 4:
		return True
	imax = int(np.ceil(np.sqrt(n)))
	for i in range(2,imax+1):
		if (n % i) == 0:
			return False
	return True

def next_prime(n,s=1):
# Find next prime given a starting number.
	while not is_prime(n):
		n += s
	return n



# Unit tests.
def test_wrong_instance():
	try:
		modular_residue([1,2],2)
		assert False
	except:
		pass

def test_wrong_instance():
	try:
		for dtype in ['uint8', 'uint16', 'uint32', 'uint64']:
			modular_residue(np.array([1,2],dtype=dtype),2,True)
	except:
		assert False

def test_attr():
	a = modular_residue(np.array([0,1,2,3,4,5],dtype='uint64'),5,True)
	assert np.all(a[1:3] == np.array([1,2],dtype='uint64'))
	a[1::2] = 2
	assert np.all(a == np.array([0,2,2,2,4,2],dtype='uint64'))

def test_sum():
	a = modular_residue(np.array([0,1,2,3,4,5],dtype='uint64'),5,True)
	b = modular_residue(np.array([2,2,2,2,2,2],dtype='uint64'),5,True)
	c = modular_residue(np.array([2,3,4,0,1,2],dtype='uint64'),5,True)
	assert np.all(a + b == c)
	assert np.all(a + b._residue == c)
	assert np.all(a + 2 == c)
	assert np.all(2 + a == c)
	a = modular_residue(np.array([120],dtype='uint8'),127,True)
	b = modular_residue(np.array([122],dtype='uint8'),127,True)
	c = modular_residue(np.array([115],dtype='uint8'),127,True)
	assert np.all(a + b == c)

def test_sub():
	a = modular_residue(np.array([0,1,2,3,4,5],dtype='uint64'),5,True)
	b = modular_residue(np.array([2,2,2,2,2,2],dtype='uint64'),5,True)
	c = modular_residue(np.array([3,4,0,1,2,3],dtype='uint64'),5,True)
	assert np.all(a - b == c)
	assert np.all(-(b - a) == c)
	assert np.all(a - b._residue == c)
	assert np.all(a - 2 == c)
	assert np.all(-(2 - a) == c)

def test_mul():
	da = np.array([0,1,2,3],dtype='uint64')
	a = modular_residue(np.array([2**32]*4,dtype='uint64')+da,5,True)
	b = modular_residue(np.array([2**32,2**33,2**34,2**35],dtype='uint64'),5,True)
	c = modular_residue(np.array([1,4,2,2],dtype='uint64'),5,True)
	assert np.all(a*b == c)
	assert np.all(a*b._residue == c)
	assert np.all((2**32)*a == da+1)
	assert np.all(a*(2**32) == da+1)

def test_pow():
	a = modular_residue(np.array([0]+[3]*6,dtype='uint64'),5,False)
	b = np.array([0,0,1,2,3,4,5],dtype='uint64')
	c = modular_residue(np.array([1,1,3,4,2,1,3],dtype='uint64'),5,False)
	assert np.all(a**b == c)
	assert np.all(a**5 == a)

def test_pow_at_short_data():
	a = modular_residue(np.array([6],dtype='uint8'),11,False)
	b = np.array([3],dtype='uint8')
	c = modular_residue(np.array([7],dtype='uint8'),11,False)
	assert np.all(a**b == c)
	assert np.all(a**11 == a)

def test_pow_in_prime_modulus():
	a = modular_residue(np.array([0]+[3]*6,dtype='uint64'),5,True)
	b = np.array([0,0,1,2,3,4,5],dtype='uint64')
	c = modular_residue(np.array([1,1,3,4,2,1,3],dtype='uint64'),5,True)
	assert np.all(a**b == c)
	assert np.all(a**5 == a)

def test_pow_in_prime_modulus_at_short_data():
	a = modular_residue(np.array([6],dtype='uint8'),11,True)
	b = np.array([3],dtype='uint8')
	c = modular_residue(np.array([7],dtype='uint8'),11,True)
	assert np.all(a**b == c)
	assert np.all(a**11 == a)

def test_div():
	a = modular_residue(np.array([1,2,3,4,3],dtype='uint64'),5,True)
	b = np.array([1,1,1,1,2],dtype='uint64')
	c = np.array([1,3,2,4,4],dtype='uint64')
	assert np.all(1/(1/a) == a)
	assert np.all((1/a)*b == c)

def test_div_at_short_data():
	a = modular_residue(np.array([1,2,3,4,3],dtype='uint8'),127,True)
	b = np.array([1,1,1,1,2],dtype='uint8')
	c = np.array([1,64,85,32,43],dtype='uint8')
	assert np.all(1/(1/a) == a)
	assert np.all((1/a)*b == c)

def test_prime_predicate():
	N = [4, 8, 11, 47, 31243, 31249]
	B = [False, False, True, True, False, True]
	for b,n in zip(B,N):
		assert is_prime(n) == b

def test_spaces():
	assert np.all(truth_space(2,2) == np.array([[0, 0, 1, 1], [0, 1, 0, 1]]))
	assert np.all(truth_space(2,3) == np.array([[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]]))
	x = modular_residue(np.array([0,1,2],dtype='uint8'),3)
	assert np.all(polynomial_space(x,[0,1,2],3) == np.array([[1,0,0], [1,1,1], [1,2,1]]))

def test_mles_solve():
	dtype = 'uint64'
	for M in [2, 3, 5, 7]:
		P = polynomial_space(modular_residue(np.arange(M,dtype=dtype),M),np.arange(M,dtype=dtype),M)
		y = truth_space(M,M,dtype)
		y = y[:,-4:]
		w = mles_solve(P,y)
		assert np.all(P@w == y)

def test_prune_error():
	yh = prune_error(np.array([0,1,2,2,1,2]),np.array([2,1,2,1,1,0]))
	assert np.all(yh == np.array([2,1,0,0,1,0]))

def test_error_reduction_exhaustive_search():
	X = 3 # input range
	G = 2 # input group size
	M = 11 # modulus
	xh = np.array([[0,0,0,0,0],[1,1,1,1,1],[1,2,2,2,2],[0,1,0,0,0],[1,1,1,1,1],[2,2,2,0,0],[0,1,0,0,2]],dtype='uint16')
	yh = np.array([0,0,1,10,0,0,2],dtype='uint16')
	dy, g, yhe = error_reduction_direct_search(xh,yh,G,M,X)
	assert dy > 0
	gmin = 0
	g = np.arange(G,dtype=int)
	dy, g, yhe = error_reduction_direct_search(xh,yh,G,M,X,g,gmin,False,True)
	assert dy == 13
	assert np.all(g == np.array([1,4]))
	dy, g, yhe = error_reduction_search(xh,yh,G,M,X)
	assert dy > 0

def test_mai():
	dtype = 'uint32'
	for M in [2, 3]: # modulus
		xh = np.arange(M,dtype=dtype) # input
		yh = truth_space(M,M).astype(dtype) # all possible outputs
		for iy in range(yh.shape()[1]):
			model = fit(xh,yh[:,iy]) # MAI training
			assert np.all(infer(xh,model) == yh[:,iy:iy+1].residue())

def test_vmai():
	dtype = 'uint16'
	xh = np.array([[0,2,2,1],[0,4,2,2],[1,2,4,2],[2,2,1,4],[2,4,3,1],[4,1,3,3],[4,1,3,4],[4,4,4,1]],dtype=dtype) # input
	yh = np.array([0,1,1,1,4,2,2,3],dtype=dtype) # all possible outputs
	model = fit(xh,yh)
	assert np.all(infer(xh,model) == yh.reshape(-1,1))
	xh = np.array([[0,1,0,0],[0,1,1,0],[1,2,2,2],[2,2,0,1]],dtype=dtype) # input
	yh = np.array([1,2,1,2],dtype=dtype) # all possible outputs
	model = fit(xh,yh)
	assert np.all(infer(xh,model) == yh.reshape(-1,1))
	model = fit(xh,yh,Mp=2)
	assert np.all(infer(xh,model) == yh.reshape(-1,1))
	xh = np.array([
		[ 12, 245,  48, 204,  50,  67, 148, 204],
		[ 19, 100, 188, 112, 150, 251,  67, 101],
		[ 28,  43, 199, 141,  47, 130, 160, 168],
		[ 31, 103, 115,  78,  77,  28,   7, 253],
		[ 47,  24, 140, 130, 121,  66, 238,   9],
		[ 61,  33,  76, 131,  59, 105, 187, 227],
		[ 61,  62, 166, 114,  53, 112, 205,  94],
		[ 86, 211, 200,  90,  79,  76, 133,  35],
		[ 94,  11, 238, 225, 110, 109, 125,  27],
		[103, 200, 166, 125, 159, 232,  21, 174],
		[107, 242, 191, 210, 216, 152, 125, 234],
		[125,  90, 160, 136, 111,  30, 140, 174],
		[126,  60,  94, 208,  58,  56, 247,  86],
		[231,   3,  20, 241, 237,  81,  59, 185],
		[231, 147, 176, 165,  58, 154,  60,  25],
		[242,  15,  47,  97,  43, 182, 117,  67]],dtype=dtype) # input
	yh = np.array([127, 200, 183, 232, 228, 85, 179, 50, 7, 191, 128, 123, 232, 156, 158, 220],dtype=dtype) # all possible outputs
	model = fit(xh,yh,Mp=2)
	assert np.all(infer(xh,model) == yh.reshape(-1,1))

def test_cmai():
	dtype = 'uint16'
	xh = np.array([[1,2,2,1],[2,1,2,0],[2,2,0,1],[2,2,2,1]],dtype=dtype) # input
	yh = np.array([2,0,0,1],dtype=dtype) # all possible outputs
	model = fit(xh,yh,incremental=True)
	assert np.all(infer(xh,model) == yh.reshape(-1,1))
	model = fit(xh,yh,incremental=True,Mp=2)
	assert np.all(infer(xh,model) == yh.reshape(-1,1))

def unit_test():
	# List functions in unit test file.
	functions = [test_wrong_instance, test_wrong_instance, test_attr,
		test_sum, test_sub, test_mul,
		test_div, test_div_at_short_data,
		test_pow, test_pow_at_short_data,
		test_pow_in_prime_modulus, test_pow_in_prime_modulus_at_short_data,
		test_prime_predicate, test_spaces, test_mles_solve,
		test_prune_error, test_error_reduction_exhaustive_search,
		test_mai, test_vmai, test_cmai] # all unit test functions

	# Tests.
	format_str = '{{:{}s}} test {{}} of {{}}\r'.format(len(functions)+1) # formatting string
	pass_fail_str = '' # pass or fail string
	failures = [] # test failures
	for counter, function in enumerate(functions):
		# Perform test.
		try:
			function()
			pass_fail_str += '.'
		except Exception as e:
			pass_fail_str += 'F'
			failures.append(function)
		print(format_str.format(pass_fail_str, counter+1, len(functions)), end='', flush=True) # progress
	
	# Report.	
	print(format_str.format(pass_fail_str, len(functions), len(functions)), flush=True)
	for fail in failures:
		print('fail at',fail.__name__)

	# Known specifications.
	print('\nKnown specifications:')
	print('1. modulus is a positive integer.')

# Test scripts.
if __name__ == "__main__":
	if len(sys.argv) > 1 and sys.argv[1].lower() == '-d':
		# Debug tests (temporary coded).
		if False: # div
			a = modular_residue(np.array([1,2,3,4,3],dtype='uint8'),127,True)
			b = np.array([1,1,1,1,2],dtype='uint8')
			c = np.array([1,3,2,4,4],dtype='uint8')
			print('a =',a)
			print('b =',b)
			print('1/(1/a) =',1/(1/a))
			print('(1/a)*b =',(1/a)*b)

		if False: # spaces
			print(truth_space(2,2))
			print(truth_space(2,3))
			x = modular_residue(np.array([0,1,2],dtype='uint8'),3)
			print(polynomial_space(x,[0,1,2],3))

		if False: # MAI
			M = 2 # modulus
			xh = np.arange(M,dtype='uint8') # input
			print('xh =',xh)
			yh = truth_space(M,M) # all possible outputs
			for iy in range(yh.shape()[1]):
				print('yh =',yh[:,iy])
				model = fit(xh,yh[:,iy]) # MAI training
				print('y =',infer(xh,model))
			save_model(model,'mai.json')
			model = load_model('mai.json')
			print('y =',infer(xh,model))

		if False: # VMAI
			dtype = 'uint16'
			xh = np.array([[0,2,2,1],[0,4,2,2],[1,2,4,2],[2,2,1,4],[2,4,3,1],[4,1,3,3],[4,1,3,4],[4,4,4,1]],dtype=dtype) # input
			yh = np.array([0,1,1,1,4,2,2,3],dtype=dtype) # all possible outputs
			print('xh =',xh)
			print('yh =',yh)
			model = fit(xh,yh) # VMAI training
			print('w =',model['parameters'])
			print('y =',infer(xh,model))
			save_model(model,'vmai.json')
			model = load_model('vmai.json')
			print('y =',infer(xh,model))
			xh = np.array([[0,1,0,0],[0,1,1,0],[1,2,2,2],[2,2,0,1]],dtype=dtype) # input
			yh = np.array([1,2,1,2],dtype=dtype) # all possible outputs
			print('xh =',xh)
			print('yh =',yh)
			model = fit(xh,yh)
			print('y =',infer(xh,model))
			model = fit(xh,yh,Mp=2)
			print('y =',infer(xh,model))

		if False: # VMAI w sorting
			logger.setLevel(logging.INFO)
			dtype = 'uint16'
			xh = np.array([
				[ 12, 245,  48, 204,  50,  67, 148, 204],
				[ 19, 100, 188, 112, 150, 251,  67, 101],
				[ 28,  43, 199, 141,  47, 130, 160, 168],
				[ 31, 103, 115,  78,  77,  28,   7, 253],
				[ 47,  24, 140, 130, 121,  66, 238,   9],
				[ 61,  33,  76, 131,  59, 105, 187, 227],
				[ 61,  62, 166, 114,  53, 112, 205,  94],
				[ 86, 211, 200,  90,  79,  76, 133,  35],
				[ 94,  11, 238, 225, 110, 109, 125,  27],
				[103, 200, 166, 125, 159, 232,  21, 174],
				[107, 242, 191, 210, 216, 152, 125, 234],
				[125,  90, 160, 136, 111,  30, 140, 174],
				[126,  60,  94, 208,  58,  56, 247,  86],
				[231,   3,  20, 241, 237,  81,  59, 185],
				[231, 147, 176, 165,  58, 154,  60,  25],
				[242,  15,  47,  97,  43, 182, 117,  67]],dtype=dtype) # input
			yh = np.array([127, 200, 183, 232, 228, 85, 179, 50, 7, 191, 128, 123, 232, 156, 158, 220],dtype=dtype) # all possible outputs
			model = fit(xh,yh,Mp=2)
			save_model(model,'vmai.json')
			print('y =',np.hstack((infer(xh,model),yh.reshape(-1,1))))

		if False: # MLSE
			xh = np.array([[0, 2, 0, 4], [2, 2, 3, 4], [2, 4, 4, 3], [3, 4, 4, 3], [4, 1, 3, 2], [4, 2, 2, 4], [4, 3, 2, 3], [4, 4, 1, 3]],dtype='uint64')
			yh = np.array([[2], [0], [0], [4], [2], [4], [1], [2]],dtype='uint64')
			model = load_model('essay.json')
			print(model)
			print('yh =',np.hstack((yh,infer(xh,model))))

		if False: # prune error
			yh = prune_error(np.array([0,1,2,2,1,2]),np.array([2,1,2,1,1,0]))
			print(yh)
			print(np.array([2,1,0,0,1,0]))

		if False: # error reduction search
			X = 3 # input range
			G = 2 # input group size
			M = 11 # modulus
			xh = np.array([[0,0,0,0,0],[1,1,1,1,1],[1,2,2,2,2],[0,1,0,0,0],[1,1,1,1,1],[2,2,2,0,0],[0,1,0,0,2]],dtype='uint16')
			yh = np.array([0,0,1,10,0,0,2],dtype='uint16')
			dy, g, yhe = error_reduction_exhausive_search(xh,yh,G,M,X)
			print(dy)
			print(g)

		if False: # CMAI
			dtype = 'uint16'
			xh = np.array([[1,2,2,1],[2,1,2,0],[2,2,0,1],[2,2,2,1]],dtype=dtype) # input
			yh = np.array([2,0,0,1],dtype=dtype) # all possible outputs
			print('xh =',xh)
			print('yh =',yh)
			model = fit(xh,yh,incremental=True)
			print(infer(xh,model))

		if True:
			dtype = 'uint16'
			xh = np.array([[1,0],[0,1],[1,1]],dtype=dtype) # input
			yh = np.array([1,0,1],dtype=dtype) # all possible outputs
			print('xh =',xh)
			print('yh =',yh)
			model = fit(xh,yh,incremental=True,Mp=0,G=2)
			print(infer(xh,model))
			save_model(model,'bmai.json')

	else:
		# Unit tests.
		unit_test()