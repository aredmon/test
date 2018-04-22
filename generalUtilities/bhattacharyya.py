"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*   Module Name: MahalanobisDist                                                                            *
*   Author(s): Larry Gariepy, Brent McCoy                                                                   *
*   Version: 1.0                                                                                            *
*   Date: 01/25/18                                                                                          *
*                                                                                                           *
*************************************************************************************************************
"""
import numpy as np
import os
from generalUtilities.config import TOM_SAPS

def bhattacharyya(x, y, xCov, yCov, kFactor=5, threshold=None):
    BIG_L = TOM_SAPS.BIG_L

    xRows = x.shape[0]
    yRows = y.shape[0]
    bhattaDistance = np.zeros( (xRows, yRows) )
    normEuclidDist = np.zeros_like( bhattaDistance )
    # currently can't find a clever way to do this without nested 'for' loops
    for xR in range(xRows):
        for yR in range(yRows):
            cov1 = xCov[xR]
            cov2 = yCov[yR]

            sigmaTot = (cov1 + cov2) / 2.0
            #if np.array_equal(sigmaTot, np.zeros_like(sigmaTot)):
            if np.linalg.det(sigmaTot) < 1.0e-24:
                sigmaInv = sigmaTot
                # gate
                nLogSig = 0.0
            else:
                sigmaInv = np.linalg.inv( sigmaTot )
                # gate
                if np.sign(np.linalg.det(cov1) * np.linalg.det(cov2)) > 0:
                    nLogSig = 0.5*np.log( np.linalg.det(sigmaTot) / np.sqrt( 
                        np.linalg.det(cov1) * np.linalg.det(cov2) ) )
                else:
                    nLogSig = 0.0
                    #print("sigmaTot det: {}".format(np.linalg.det(sigmaTot)))
                    #print("cov determinants: {}, {}".format(np.linalg.det(cov1), np.linalg.det(cov2)))
            # gate
            kAdj = kFactor * ( np.linalg.norm( np.diag(cov1) ) + np.linalg.norm( np.diag(cov2) ) )
    
            try:
                dX = x[xR] - y[yR]
            except ValueError:
                dX = np.zeros(2)
                print("x set: {}, y set: {}".format(x[xR], y[yR]))
                print("x shape: {}, y shape: {}".format(x.shape, y.shape))
            # figure out thresholding for percent based matching
            """ NEED TO ADJUST INDEXING AND FIGURE OUT WHAT CONSTITUTES A 90% MATCH """
    	    normEuclidDist[xR,yR] = np.linalg.norm(dX)
    	    bhattMahalDist = np.dot(np.dot(dX.transpose(), sigmaInv), dX) ## NOTE: matrix multiplication
    	    #bhattMahalDist = np.dot(np.dot(dX.transpose(), sigmaInv), dX) / 8.0  ## NOTE: matrix multiplication

            if threshold == None:
                nonCorrelatedCondition = (np.sqrt( np.absolute(bhattMahalDist) ) > kAdj)
            else:
                nonCorrelatedCondition = ( (dX <= (1-threshold)*x[xR]) & (dX >= (threshold-1)*x[xR]) )
 
            if nonCorrelatedCondition.any():
                bhattaDistance[xR, yR] = BIG_L
            else:
                bhattaDistance[xR, yR] = bhattMahalDist + nLogSig
    # return mahalDistance and normEucliDist parameters
    return bhattaDistance, normEuclidDist
    			
if __name__ == "__main__":
	#x = np.array([[ 0, 0],[1, 2]])
	#y = np.array([[0,0],[6,10]])
	#Sx = np.array( [[4, 0],[0,4]])
	#Sy = np.array( [[1,0],[0,1]])
        x = np.random.randint(10, size=(3,2))
        y = np.random.randint(10, size=(3,2))
        Sx = np.zeros((3,2,2))
        Sy = Sx.copy()
        for ii in range(x.shape[0]):
            Sx[ii] = np.diag( np.random.randint(1, 5, size=(2)) )
            Sy[ii] = np.diag( np.random.randint(1, 5, size=(2)) )
            
        print("x: \n{} \ny: \n{} \nSx: \n{} \nSy: \n{}".format(x, y, Sx, Sy))
	K_factor = 3
	[M,dist] = bhattacharyya(x,y,Sx,Sy,K_factor)
        print("bhattacharyya distance matrix: \n{}".format(M))
        print("normal Euclidean distance matrix: \n{}".format(dist))
	
"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""
