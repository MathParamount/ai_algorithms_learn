#import libraries
import numpy as np
import pandas as pd
import math
import array as arr
import itertools
from scipy import scipy.signal as ssg

class train_vect_module(xh,yh, G = None):
	
	def __init__(self):
		self.xh = xh;
		self.yh = yh;
		self.G = 0;

	def train(self):
		
		n_input = xh.shape[1];
		S = xh.shape[0];
		
		#refinement method (2 inputs)
		if n_input <= 2:

			if n_input == 1:
				M = 2;
				l_bar = 0;
			else:
				M = 5;
				l_bar = math.log2(M-1);
			
			if G is None or len(G) == 0:
				G = n_input + 1;

			X = [xh];
			l = 1;
			Dy = yh;
			W = [];
			S_rand = arr.array('i', [0:M-1]);
			L = np.zeros((1,), dtype=np.int);
			I = np.zeros((G,), dtype=np.int);

			while (X and X[l] != []):
				
				if 0 <= l < len(X):
					N = len(X[l]);
				else:
					raise IndexError("index out of specification");
				
				if G <= N:
					Gs = list(itertools.combinations(range(1, N+1), G));
				else:
					Gs = [1:N];

				#Layer refinament (binary shift)
				if l <= l_bar:
					shift = l_bar - l - 1;
					X[l+1] = (xh // (2 ** shift )) % 2;
				else:
					X[l+1] = [];

				for g in len(Gs[0]):
					
					xs = X[l] [:, Gs] @ (2 ** np.arrange(len(Gs)));

					if xs.shape[0] == 1:
						xs = xs.T

					n = max(3, len(Gs))
					
					#Creating base matrix and using hstack to concatenate horizontally
					####

					base = np.hstack([np.ones((m,1)), np.tile(xs, (1, n -1))]);
					V = np.cumprod(base, axis = 1)

					v_residue = V % M;
					#vandermont numerical value
					v = v_residue.astype(np.float64);

					####

					_ju = np.unique(xs, return_inverse=True)
					sums = np.bincount(ju, weights = Dy)
					counts = np.bincount(ju)
					
					Dyp = [ju]*(sums/counts);

					if np.any(Dyp):
						rows_v, cols_v = v.shape;

						Dy = np.asarray(Dy).flatten();
						rows_Dy = Dy.shape[0];

						if rows_Dy != rows_v:

							if rows_Dy > rows_v:
            				Dy_adj = Dy[:rows_v]  # catch first rows_v elements
        					else:
            				# Adding zeros in final spaces to complete
            				Dy_adj = np.concatenate([Dy, np.zeros(rows_v - rows_Dy)])
    					else:
        					Dy_adj = Dy

						w, residuals, rank, s = np.linalg.lstsq(v, Dy_adj, rcond=None);
						w_mod = w % M;		#Applying modular residuo

						w = w_mod.astype(np.float64);

						#output error reduction
						error = v * double(w);

						if len(error[0]) < len(Dy[0]):
							error = np.concatenate([error,np.zeros ((Dy.shape[0] - error.shape[0], 1))], axis = 0) 
						else:
							error = error[:Dy.shape[0]]

						#consistent error
						Dy_temp =

	def modular_residue():

