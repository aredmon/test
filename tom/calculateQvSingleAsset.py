import numpy as np
from math import *
import TOM_SAPS
from removeBias import removeBias, removeBiasWithCovariance
from Munkres import munkres

def calculateQvSingleAsset(ALG_OPTION, BIAS_METHOD, rfObjects, irObjects, sigRF, sigIR, gatingK, nRF, nIR, indexRF, indexIR):
	
	indexRF = indexRF[np.logical_not(np.isnan(indexRF))]
	indexIR = indexIR[np.logical_not(np.isnan(indexIR))]
	print("indexRF = ", indexRF)
	print("indexIR = ", indexIR)
	rfObjects = rfObjects[:,indexRF]
	irObjects = irObjects[:,indexIR]
	print("rfObjectsw = ", rfObjects)
	print("irObjects = ", irObjects)
	
	lethalObjNum = 0
	score = 0
	
	if ALG_OPTION == 1:
		# Bias Removal
		bias = removeBias(rfObjects, irObjects, BIAS_METHOD)
		x_unbiased = rfObjects + bias  ## TEST broadcast
		
		# Correlation
		## TBD
		# M = MahalDist(x_ub, x_ir, SigRF, SigIR);
		# P = ModMunkres2(M')
		# indrff = indrf
		# 
		# Nrff = Nrf;

	elif ALG_OPTION == 2:
		# TBD
		# %Bias removal:
		# bias = remove_bias(rf_objects, ir_objects, BIAS_METHOD);
		# x_ub(1,:) = rf_objects(1,:) + bias(1);
		# x_ub(2,:) = rf_objects(2,:) + bias(2);
		# [M,D] = MahalDist3(x_ub, ir_objects, SigRF, SigIR, GatingK);
		# indrff = indrf;
		# P = zeros(Nir,1);
		# PP = ModMunkres2(M');
		# %Now break any associations that are too far:
		# j = 0;
		# for i = 1 : Nir
			# j = j + 1;
			# if PP(j) > 0 && PP(j) < BIG_L
				# P(i) = PP(j);
			# end
		# end
	elif ALG_OPTION == 3:
		if BIAS_METHOD == 4:
			bias = removeBiasWithCovariance(rfObjects, irObjects, sigRF, sigIR)
		else:
			bias = removeBias(rfObjects, irObjects, BIAS_METHOD)
			
		x_unbiased = rfObjects - bias
		# Gating
		[M,D] = bhattacharyya(x_unbiased, irObjects, sigRF, sigIR, gatingK)
		#        [M2,D2] = mahal_dist3(x_ub, ir_objects, SigRF, SigIR, GatingK);  %% TODO  check this
		out = np.zeros( (nRF,1) )
		PM = []
		for i in range(nRF):
			if (np.min(M[i,:]) < TOM_SAPS.BIG_L):
				PM = np.vstack(PM, M[i,:])
				out[i] = 1
		indexRF2 = indexRF[out == 1]
		nRF2 = len(indexRF2)
		P = np.zeros( (nIR, 1) )
		
		out = np.zeros( (nIR, 1) )
		
		if PM:  ## True if PM is not empty
			PPM = []
			indexIR2 = np.zeros( (nIR, 1) )
			for i in range(nIR):
				if np.min(PM[:,i]) < TOM_SAPS.BIG_L:
					PPM = np.hstack(PPM, PM[:,i])
					out[i] = 1
			indexIR2 = indexIR[out == 1]
			if PPM:
				PP = munkres(PPM.T)
				## break any associations for which the cost is too high
				j = 0
				for i in range(nIR):
					if out[i] and PP[j]:
						if (PM[PP[j],i] < TOM_SAPS.BIG_L):
							if PP[j]:
								P[i] = PP[j]
						j = j + 1
	
	lethalObjNum = 0
	score = 0
	assign = np.zeros( (1,nIR) )


	for i in range(nIR):
		if P[i]:
			assign[i] = indexRF2[P[i]]
			if (assign[i] == 1):  ## If the IR object was assigned to RF object 1, then it is assumed to be the object of interest
				lethalObjNum = indexRF2[P[i]]
			score = score + PM[P[i],i]

	## TODO: need to return "assign" and calculate correlation metrics for all common objects			
		
	return (lethalObjNum, score)
	

	
if __name__ == "__main__":
	ALG_OPTION = 3
	BIAS_METHOD = 6
	rfObjects = np.array([ [0, 1, 2, 3, 0], [1, -1, 0, 3, 2]])
	#x2 = np.array([ [ 0, 10, 11, 2], [11, -11, 4, 0]])
	#test_bias = np.array( [ [1,],[1,] ])
	irObjects = np.array([ [0, np.nan, np.nan, np.nan, np.nan, 1, 2, 3, 4], [1, np.nan, np.nan, np.nan, np.nan, -1, 0, 3, 5]])
	sigRF = np.array( [[4, 0],[0,4]])
	sigIR = np.array( [[1,0],[0,1]])
	gatingK = 5
	nRF = 5
	nIR = 5
	indexRF = np.array([0,1,2,3,4])
	indexIR = np.array([0,np.nan,5,6,7,8])
	(lethalObjNum, score) = calculateQvSingleAsset(ALG_OPTION, BIAS_METHOD, rfObjects, irObjects, sigRF, sigIR, gatingK, nRF, nIR, indexRF, indexIR)