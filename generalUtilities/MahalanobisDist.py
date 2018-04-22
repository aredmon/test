"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*   Module Name: MahalanobisDist                                                                            *
*   Author(s): Mark Lambrecht, Brent McCoy                                                                  *
*   Version: 1.0                                                                                            *
*   Date: 01/25/18                                                                                          *
*                                                                                                           *
*************************************************************************************************************
"""
import numpy as np
from generalUtilities.config import TOM_SAPS
"""
----------------------------------------------------------------------------------------------
    Inputs:     x,y		= 2xn sets of points of points (each)
                xCov, yCov      = nx2x2 covariance matrices related to x & y

    Outputs:    mahalDistance   = mxn matrix with corresponding distances
                
----------------------------------------------------------------------------------------------
"""
def mahalDist1(x, y, xCov, yCov, numRows):
    xRows = x.shape[0]
    yRows = y.shape[0]
    mahalDistance = np.empty((xRows, yRows))
    normEucliDist = np.zeros_like(mahalDistance)
    # currently can't find a clever way to do this without nested 'for' loops
    for xR in range(xRows):
        for yR in range(yRows):
            cov1 = xCov[xR]
            cov2 = yCov[yR]
            # find the determinantes of the covariance matrices:
            det1 = np.linalg.det(cov1)
            det2 = np.linalg.det(cov2)

            # check to see if one is larger than the other, if so, transform into that matrix
            if det1 > det2:
                eigVals, eigVecs = np.linalg.eig(cov1)
            else:
                eigVals, eigVecs = np.linalg.eig(cov2)

            sigma1 = np.dot( eigVecs, np.dot(cov1, eigVecs.transpose()) )
            sigma2 = np.dot( eigVecs, np.dot(cov2, eigVecs.transpose()) )

            sigmaTot = sigma1 + sigma2
            sigmaInv = np.linalg.inv( sigmaTot )

            dX = eigVecs*x[xR] - eigVecs*y[yR]
            mahalDistance[xR,yR] = np.dot(dX.transpose(), np.dot(sigmaInv, dX))
            normEucliDist[xR,yR] = np.linalg.norm(dX)
    # return final matrix
    return mahalDistance, normEucliDist

def mahalDist2(x, y, xCov, yCov, kFactor=3):
    BIG_L = TOM_SAPS.BIG_L
    
    #perform mahalanobis calculation of type 2
    xRows = x.shape[0]
    yRows = y.shape[0]
    mahalDistance = np.empty((xRows, yRows))
    normEucliDist = np.zeros_like(mahalDistance)
    # currently can't find a clever way to do this without nested 'for' loops
    for xR in range(xRows):
        for yR in range(yRows):
            cov1 = xCov[xR]
            cov2 = yCov[yR]
            # find the determinantes of the covariance matrices:
            det1 = np.linalg.det(cov1)
            det2 = np.linalg.det(cov2)

            covInv = np.linalg.inv( cov1 + cov2 )

            # check to see if one is larger than the other, if so, transform into that matrix
            if det1 > det2:
                eigVals, eigVecs = np.linalg.eig(cov1)
            else:
                eigVals, eigVecs = np.linalg.eig(cov2)

            sigma1 = np.dot( eigVecs, np.dot(cov1, eigVecs.transpose()) )
            sigma2 = np.dot( eigVecs, np.dot(cov2, eigVecs.transpose()) )

            sigmaTot = sigma1 + sigma2
            sigmaInv = np.linalg.inv( sigmaTot )
            nLogSig = np.log( np.det(sigmaTot) )
            # gate
            kAdj = kFactor * ( np.linalg.norm( np.diag(sigma1) ) + np.linalg.norm( np.diag(sigma2) ) )

            # calcultate distance parameter
            dX = eigVecs * x[xR] + eigVecs * y[yR]
            normEucliDist[xR,yR] = np.linalg.norm(dX)

            bhattMahalDist = np.dot(dX.transpose(), np.dot(sigmaInv, dX))

            # not sure what these were used for previously but here is the python version of dd and ddx
            #bhattMahalDiff = x[xR] - y[yR]     # formerly ddx
            # formerly dd
            #bhattDistance = np.dot(bhattMahalDiff.transpose(), np.dot(covInv, bhattMahalDiff))

            if np.sqrt(bhattMahalDist) > kAdj:
                mahalDistance[xR, yR] = BIG_L
            else:
                mahalDistance[xR, yR] = bhattMahalDist + nLogSig
    # return mahalDistance and normEucliDist parameters
    return mahalDistance, normEucliDist

def mahalDist3(x, y, xCov, yCov, kFactor=3):
    BIG_L = TOM_SAPS.BIG_L
    
    #perform mahalanobis calculation of type 2
    xRows = x.shape[0]
    yRows = y.shape[0]
    mahalDistance = np.empty((xRows, yRows))
    normEucliDist = np.zeros_like(mahalDistance)
    # currently can't find a clever way to do this without nested 'for' loops
    for xR in range(xRows):
        for yR in range(yRows):
            cov1 = xCov[xR]
            cov2 = yCov[yR]

            sigmaTot = cov1 + cov2
            sigmaInv = np.linalg.inv( sigmaTot )
            nLogSig = np.log( np.linalg.det(sigmaTot) )
            # gate
            kAdj = kFactor * ( np.linalg.norm( np.diag(cov1) ) + np.linalg.norm( np.diag(cov2) ) )

            # formerly ddx
            bhattMahalDiff = x[xR] - y[yR]     
            normEucliDist[xR, yR] = np.linalg.norm( bhattMahalDiff )
            # formerly dd
            bhattMahalDist = np.dot(bhattMahalDiff.transpose(), np.dot(sigmaInv, bhattMahalDiff))

            if np.sqrt(bhattMahalDist) > kAdj:
                mahalDistance[xR, yR] = BIG_L
            else:
                mahalDistance[xR, yR] = bhattMahalDist + nLogSig
    # return mahalDistance and normEucliDist parameters
    return mahalDistance, normEucliDist

"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""
