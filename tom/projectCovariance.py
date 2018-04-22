import numpy as np
from math import *
import TOM_SAPS
import Utils

def projectCovariance(clusterPos, observerPos, C):
	[D,V] = np.linalg.eig(C)

	## initialize return values
	a = 1
	b = 1
	theta = 0

	v1 = abs(D[0] - D[1]) / abs(max(D[0], D[1]))
	v2 = abs(D[1] - D[2]) / abs(max(D[1], D[2]))
	v3 = abs(D[0] - D[2]) / abs(max(D[0], D[2]))

	threshold = TOM_SAPS.EIGENVALUE_DIFF_THRESHOLD
	duplicateSigma = 0;  ## We will look for a duplicated eigenvalue (sigma) within a certain threshold
	lineOfSight = Utils.unitVector(clusterPos - observerPos)

	pancakeFlag = (v1 < threshold) or (v2 < threshold) or (v3 < threshold)
	if pancakeFlag:
		if (v1 < threshold):
			duplicateSigma = sqrt(abs(D[0]))
		else:
			duplicateSigma = sqrt(abs(D[2]))

	if (v1 < threshold):
		if (v3 < threshold):  ## essentially all 3 eigenvalues are the same, within tolerance, so spherical covariance
			a = atan(sqrt(abs(D[0]))/Utils.norm(lineOfSight))
			b = a
			theta = 0
			return (a,b,theta)

		V[:,1] = Utils.unitVector(np.cross(V[:,0], V[:,2]))
	elif (v2 < threshold) or (v3 < threshold):
		V[:,2] = Utils.unitVector(np.cross(V[:,0], V[:,1]))

	z = np.array([0,0,1])  ## local up vector
	e = Utils.unitVector(np.cross(lineOfSight, z))
	up = Utils.unitVector(np.cross(e, lineOfSight))

	B = np.hstack(lineOfSight.reshape( (3,1) ),  e.reshape( (3,1) ),  up.reshape( (3,1) ) )
	V_rotated = np.dot( np.linalg.inv(B), V)

	C_newBasis = np.dot (np.linalg.inv(B), C)

	v0NewBasis = np.dot (np.linalg.inv(B), V[:,0])
	v0Proj = norm( v0NewBasis * sqrt(abs(D[0])))

	v1NewBasis = np.dot (np.linalg.inv(B), V[:,1])
	v1Proj = norm( v1NewBasis * sqrt(abs(D[1])))

	v2NewBasis = np.dot (np.linalg.inv(B), V[:,2])
	v2Proj = norm( v2NewBasis * sqrt(abs(D[2])))

	projectionMatrix = np.vstack([ np.hstack( [v0Proj, v0NewBasis]) ,
								   np.hstack( [v1Proj, v1NewBasis]) ,
								   np.hstack( [v2Proj, v2NewBasis]) ])

	if pancakeFlag:
		## when a covariance has two equal eigenvalues, and those eigenvalues are larger than the 3rd, then the
		## covariance is pancake-shaped.  This is typical with ground sensors.  Geometrically, the projection of that 
		## covariance into a 2D plane will always preserve that largest eigenvalue
		a = atan(duplicateSigma/Utils.norm(lineOfSight))
		projectionMatrix = projectionMatrix[projectionMatrix[:,0].argsort()]  ## this emulates MATLAB sortrows, sorting by the first column
		b = atan(projectionMatrix[1,0]/Utils.norm(lineOfSight))

	    # need some improvement here; need to take the plane of the major axes of the pancake, and calculate a vector within that plane, orthogonal to the los, as the 
	    # vector for the major axis;  then calculate a vector orthogonal to both the los and the major axis as the minor axis vector.  Then need to calculate the extent of the ellipsoid 
	    # along that vector
	    
	    # Another more practical approach might be just to generate 100 points within the ellipsoid, change the basis to the observer body coordinates, and then project each of the vectors into the 
	    # observer LOS, and calculate an ellipse from that.
		theta = atan2(projectionMatrix[2,3], projectionMatrix[2][2])  # Use the up and east components to calculate a trigonometric angle for the ellipse rotation in 2D
	else:
		## TODO: the general case

