import numpy as np

def noiseIRObjects(objectFlags, irObjects, irBias, irSigmaMajorAxis, irSigmaMinorAxis, irIndexList):
	(r, nTotal) = objectFlags.shape
	
	## All objects for a given observer will have the same covariance and bias, but with different random noise
	angIR = np.random.rand() * 2 * np.pi
	irBiasMag = np.random.randn() * irBias
	biasIR = np.array([[np.cos(angIR),  np.sin(angIR)]])  ## NOTE: making this 2D to interact properly later
	irTransformation = np.array( [ [np.cos(angIR), np.sin(angIR)], [-np.sin(angIR), np.cos(angIR)] ])
	
	## Covariance matrices
	cov = np.array([ [irSigmaMajorAxis**2, 0], [0, irSigmaMinorAxis**2] ])
	sigIR = np.dot(np.dot(irTransformation.T, cov),  irTransformation)  ## Note the transposes are backward from the MATLAB because I didn't transpose the original irTranformation computation
	(r,c) = irObjects.shape
	
	xIR = np.zeros( (r,2) )
	
	index = 0
	
	## True positions, RF and IR estimates
	
	for i in range(nTotal):
		if objectFlags[1,k] == 0:  ## skip objects that are not IR visible
			continue
			
		index = index + 1
		randX = np.random.randn(2, 1)
		
		indexIR = irIndexList(index)
		randomNoise = np.array( [ [irSigmaMajorAxis * randX[0],] , [irSigmaMinorAxis * randX[1],] ] )
		
		## TODO TODO TEST TEST
		xIR[indexIR,:] = np.dot(irTranformation.T, randomNoise) + irObjects[k,:] + biasIR  ## TODO, test this, not sure all the array dimensions work