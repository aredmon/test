import numpy as np
import TOM_SAPS
from math import *
from bhattacharyya import bhattacharyya
from Munkres import munkres
import Utils

def removeBias(x1, x2, option = 1):
	## Remove bias considering position only
	x1 = x1[:, np.logical_not(np.isnan(x1[0,:]))]
	x2 = x2[:, np.logical_not(np.isnan(x2[0,:]))]
	bias = np.array( [0,],[0,] )
	if (option == 1):
		x1_mean = np.mean(x1,1) ## average along rows
		x2_mean = np.mean(x2,1)
		bias = x2_mean - x1_mean
		bias = bias.reshape( (2,1) )  ## the np.mean() method returns a 1D array, so have to convert back to 2D
	elif (option == 2) or (option == 3):
		x1_mean = np.median(x1,1)
		x2_mean = np.median(x2,1)
		bias = x2_mean - x1_mean
		if (option == 3):
			spread1 = np.std(x1, 1)  ## compute standard deviations, component-wise
			spread2 = np.std(x2, 1)
			lim = Utils.norm(spread1) + Utils.norm(spread2)
			if Utils.norm(bias) < lim:
				bias = np.array([ [0,],[0,] ])
	elif (option == 5):
		bias = x2[:,1] - x1[:,1]
	elif (option == 6):
		## Median method
		(_, n1) = x1.shape
		(_, n2) = x2.shape
		dx = np.empty( (2, n1*n2) )
		dx.fill(np.nan) ## initialize the array with NaNs
		count = 1
		for i in range(n1):
			for j in range(n2):
				dx[:,cnt] = x2[:,j] - x1[:,i]
				count = count + 1
		bias = np.median(dx, 1)
	bias = bias.reshape( (2,1) )  ## need to reshape to a column vector
	return bias

def removeBiasWithCovariance(x1, x2, sig1, sig2):
	"""Remove bias considering position and covariance
	x1 and x2 are 2 x ? arrays of positions (typically az/el, but the math doesn't require that) that are biased with respect to each other"""
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
		bias = x1[:,bestPair[0]] - x2[:,bestPair[1]]
		return bias.reshape( (2,1) )  ## reshape to a column vector
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
	
	