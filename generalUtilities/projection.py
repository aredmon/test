"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*   Module Name: projection                                                                                 *
*   Author(s): Larry Gariepy, Brent McCoy                                                                   *
*   Version: 1.0                                                                                            *
*   Date: 01/18/18                                                                                          *
*                                                                                                           *
*       Module          Takes the Euclidean space covariance and projects it to a 2D covariance in az/el    *
*       Description:    space based on the position of the object and the observer                          *
*                                                                                                           *
*       INPUTS:         cluster_pos     -   position vector (1x3) of cluster center                         *
*                       observer_pos    -   position vector (1x3) of outside observer                       *
*                       covariance      -   covariance matrix (nx3x3)                                       *
*                                                                                                           *
*       OUTPUTS:        semiMajor       -   semi-major axis length of covariance ellipse                    *
*                       semiMinor       -   semi-minor axis length of covariance ellipse                    *
*                       rotationAngle   -   rotation angle for original covariance to projected covariance  *
*                                                                                                           *
*************************************************************************************************************
"""
import numpy as np
#from generalUtilities.Classes import jsonData, jsonInfo
from generalUtilities.utilities import vecNorm
from generalUtilities.config import TOM_SAPS

def normalize(vector):
    if vector.ndim > 1:
        magnitude = vecNorm(vector, axis=1)
    else:
        magnitude = vecNorm(vector, axis=0)
    if magnitude.any() == 0:
        unitVector = vector
    else:
        unitVector = vector / magnitude
    return unitVector

def project_covariance(cluster_pos, observer_pos, covariance):
    if covariance.ndim > 2:
        eVals = np.zeros( (covariance.shape[0], covariance.shape[1]) )
        eVecs = np.zeros_like( covariance )
        index = 0
        for covMat in covariance:
            eVals[index], eVecs[index] = np.linalg.eig(covariance[index])   #eVecs = V in matLab code
            index += 1
    else:
        eVals = np.zeros((1, covariance.shape[0]))
        eVecs = np.zeros((1, covariance.shape[0], covariance.shape[1]))
        eVals[0], eVecs[0] = np.linalg.eig(covariance)   #eVecs = V in matLab code


    eMat = np.diag(eVals)               #eMat = D in matlab code

    # initialize return variables
    semiMajor = np.ones(covariance.shape[0])
    semiMinor = np.ones(covariance.shape[0])
    theta = np.zeros(covariance.shape[0])

    # calculate v1
    v1 = np.zeros(eVals.shape[0])
    v1Divisor = np.absolute(np.fmax(eVals[:,0],eVals[:,1]))
    validV1 = (v1Divisor != 0)
    v1[ validV1 ] = np.absolute(eVals[:,0] - eVals[:,1]) / v1Divisor[validV1]
    # calculate v2
    v2 = np.zeros(eVals.shape[0])
    v2Divisor = np.absolute(np.fmax(eVals[:,1],eVals[:,2]))
    validV2 = (v2Divisor != 0)
    v2[ validV2 ] = np.absolute(eVals[:,1] - eVals[:,2]) / v2Divisor[validV2]
    # calculate v3
    v3 = np.zeros(eVals.shape[0])
    v3Divisor = np.absolute(np.fmax(eVals[:,0],eVals[:,2]))
    validV3 = (v3Divisor != 0)
    v3[ validV3 ] = np.absolute(eVals[:,0] - eVals[:,2]) / v3Divisor[validV3]
    
    threshold = TOM_SAPS.EIGENVALUE_DIFF_THRESHOLD
    sigma = 0
    lineOfSight = normalize(cluster_pos - observer_pos)

    pancakeFlag = np.asarray([v1.any() < threshold, v2.any() < threshold, v3.any() < threshold], dtype=np.bool)

    if np.any(pancakeFlag):
        if v1.any() < threshold:
            sigma = np.sqrt(eVals[:,0])
        else:
            sigma = np.sqrt(eVals[:,2])

    if v1.any() < threshold:
        if v3.any() < threshold:
            print("eigen values: {}".format(eVals[:,0]))
            print("line of sight: {}".format(vecNorm(lineOfSight)))
            semiMajor = np.arctan2(np.sqrt(eVals[:,0])) / vecNorm(lineOfSight)
            semiMinor = semiMajor
            theta = np.zeros(covariance.shape[0])
            # goes straight to the return statment
        else:
            eVecsTmp = np.cross(eVecs[:,0],eVecs[:,2])
            eVecs[:, 2] = normalize(eVecsTmp)
            # check v2 and v3
    elif v2.any() < threshold or v3.any() < threshold:
        eVecsTmp = np.cross(eVecs[:,0],eVecs[:,1])
        eVecs[:, 3] = normalize(eVecsTmp)
    else:
        z = np.array([0, 0, 1])
        ecc = normalize( np.cross(lineOfSight, z) )
        up = normalize( np.cross(ecc, lineOfSight) )

        bMat = np.zeros((3, 3))
        #bInv = np.zeros((covariance.shape[0], 3, 3))
        projection = np.zeros((3,1))
        projectionMatrix = np.zeros((covariance.shape[0], 3, 4))
        for ii in range(covariance.shape[0]):
            bMat = np.transpose( np.vstack((lineOfSight, ecc, up)) )
            #print("bMatrix: {}".format(bMat))
            #bInv[ii] = np.linalg.inv(bMat[ii])
            vecRotated = np.dot(eVecs[ii], bMat) 
            cov_new_basis = np.dot(covariance[ii], bMat)

            v1_new_basis = np.dot(eVecs[ii][0], bMat)
            projection[0] = vecNorm(v1_new_basis[1:3])*np.sqrt(eVals[ii, 0])

            v2_new_basis = np.dot(eVecs[ii][1], bMat)
            projection[1] = vecNorm(v2_new_basis[1:3])*np.sqrt(eVals[ii, 1])

            v3_new_basis = np.dot(eVecs[ii][2], bMat)
            projection[2] = vecNorm(v2_new_basis[1:3])*np.sqrt(eVals[ii, 2])

            newBasisMat = np.vstack((v1_new_basis, v2_new_basis, v3_new_basis))
            projectionMatrix[ii,:] = np.append(projection, newBasisMat, axis=1)

        # sort array
        projectionMatrix = np.sort(projectionMatrix, axis=0)
        if pancakeFlag.any():
            """ 
                when a covariance has two equal eigenvalues, and those eigenvalues are larger than the 
                3rd eigenvalue, then the covariance
                is pancake-shaped.  Geometrically, the projection of that covariance onto any 2D plane must 
                have a semi-major axis equal to the repeated eigenvalue.
            """
            semiMajor = np.arctan2(sigma, norm(lineOfSight));
            semiMinor = np.arctan2(projectionMatrix[:,1,0], vecNorm(lineOfSight)); 
            """
                take the second highest eigenvector projection, scaled by the eigenvalue, as the minor axis
                need some improvement here; need to take the plane of the major axes of the pancake, and 
                calculate a vector within that plane, orthogonal to the lineOfSight, as the 
                vector for the major axis;  then calculate a vector orthogonal to both the lineOfSight and 
                the major axis as the minor axis vector.  Then need to calculate the extent of the ellipsoid 
                along that vector
                
                Another more practical approach might be just to generate 100 points within the ellipsoid, 
                change the basis to the observer body coordinates, and then project each of the vectors into 
                the observer lineOfSight, and calculate an ellipse from that.
            """
        
            theta = np.arctan2(projectionMatrix[:,2,3], projectionMatrix[:,2,2]);  
            # Use the up and east components to calculate a trigonometric angle for the ellipse rotation in 2D
        else:
            semiMajor = np.arctan2(projectionMatrix[:,2,0], vecNorm(lineOfSight))
            semiMinor = np.arctan2(projectionMatrix[:,1,0], vecNorm(lineOfSight)) 
            # take the second highest eigenvector projection, scaled by the eigenvalue, as the minor axis
            theta = np.arctan2(projectionMatrix[:,2,3], projectionMatrix[:,2,2]);
    
    return semiMajor, semiMinor, theta
"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""
