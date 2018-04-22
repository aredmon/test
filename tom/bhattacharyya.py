import numpy as np
import TOM_SAPS
from math import *
from Utils import norm

def bhattacharyya(x, y, Sx, Sy, K_factor = 5):
	BIG_L = TOM_SAPS.BIG_L

	(_, m) = x.shape
	(_, n) = y.shape
	M = np.zeros( (m,n) )
	dist = M.copy()
	
	S = (Sx + Sy)/2.0;
	Si = np.linalg.inv(S)
	
	lnS = 0.5*log( np.linalg.det(S) / sqrt( np.linalg.det(Sx) * np.linalg.det(Sy) ) )
	
	for i in range(m):
		for j in range(n):
			dx = x[:,i] - y[:,j]
			dist[i,j] = norm(dx)
			d = np.dot(np.dot(dx.T, Si), dx)  ## NOTE: matrix multiplication
			M[i,j] = d + lnS
			if sqrt(d) > K_factor:
				M[i,j] = BIG_L
				
	return (M,dist)
	
if __name__ == "__main__":
	x = np.array([[ 0, 0],[1, 2]])
	y = np.array([[0,0],[6,10]])
	Sx = np.array( [[4, 0],[0,4]])
	Sy = np.array( [[1,0],[0,1]])
	K_factor = 3
	[M,dist] = bhattacharyya(x,y,Sx,Sy,K_factor)
	print(M)
	print(dist)
	