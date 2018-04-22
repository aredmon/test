import numpy as np
import TOM_SAPS
from math import *
from bhattacharyya import bhattacharyya
from Munkres import munkres



def removeBiasWithCovariance(x1, x2, sig1, sig2):
	"""x1 and x2 are 2 x ? arrays of positions (typically az/el, but the math doesn't require that) that are biased with respect to each other"""
	x1 = x1[:, np.logical_not(np.isnan(x1[0,:]))]
	x2 = x2[:, np.logical_not(np.isnan(x2[0,:]))]

	(_, N) = x1.shape
	(_, M) = x2.shape
	
	costThreshold = np.Inf  ## Maximum acceptible cost for the total solution 
	bestPair = (-1, -1)
	
	for i in range(N):
		for j in range(M):
			bias = x1[:,i] - x2[:,i]
			x_unbiased = x1 + bias.reshape( (2,1) )  ## NOTE: broadcasts the addition of the 2x1 bias vector to each column of the 2xN array "x1"
			
			(D, _) = bhattacharyya(x_unbiased, x2, sig1, sig2)
			P = munkres(D.T)
			
			cost = 0
			num = 0
			for ii in range(M):
				nz = P[ii][np.nonzero(P[ii])]
				
				if len(nz) == 1:
					cost = cost + nz[0]  ## de-referencing so that we don't inadvertently convert "cost" to an ndarray type
					num = num + 1
			
			if num > 0:
				averageCost = cost / num
			else:
				averageCost = np.Inf
			
			if averageCost < costThreshold:
				costThreshold = averageCost
				bestPair = (i,j)
	
	if bestPair[0] >= 0:
		return x1[:,bestPair[0]] - x2[:,bestPair[1]]
	else:
		return np.array([ [0,], [0,] ])  ## return zero bias if no solution is found

if __name__ == "__main__":
	x1 = np.array([ [0, 1, 2, 3], [1, -1, 0, 3]])
	#x2 = np.array([ [ 0, 10, 11, 2], [11, -11, 4, 0]])
	test_bias = np.array( [ [1,],[1,] ])
	x2 = x1 + test_bias
	Sx = np.array( [[4, 0],[0,4]])
	Sy = np.array( [[1,0],[0,1]])

	bias = removeBias(x1,x2,Sx,Sy)
	print ("bias = ", bias, "; test_bias = ", test_bias)
	
	