"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*   Module Name: GenerateTestCases                                                                          *
*   Author(s): Joseph Kirk, Brent McCoy                                                                     *
*   Version: 1.0                                                                                            *
*   Date: 03/30/18                                                                                          *
*                                                                                                           *
*       Description: Computes a matrix of pair-wise distances between points in                             *
*                    A and B, using one of {euclidean,cityblock,chessboard} methods                         *
*       Author:                                                                                             *
*             Joseph Kirk                                                                                   *
*             jdkirk630@gmail.com                                                                           *
*       Date: 02/27/15                                                                                      *
*                                                                                                           *
*       Release: 2.0                                                                                        *
*                                                                                                           *
*       Inputs:                                                                                             *
*           A      - (required) MxD matrix where M is the number of points in D dimensions                  *
*           B      - (optional) NxD matrix where N is the number of points in D dimensions                  *
*                       if not provided, B is set to A by default                                           *
*           METHOD - (optional) string specifying one of the following distance methods:                    *
*                       'euclidean'                       Euclidean distance (default)                      *
*                       'taxicab','manhattan','cityblock' Manhattan distance                                *
*                       'chebyshev','chessboard','chess'  Chebyshev distance                                *
*                       'grid','diag'                     Diagonal grid distance                            *
*                                                                                                           *
*       Outputs:                                                                                            *
*           DMAT   - MxN matrix of pair-wise distances between points in A and B                            *
*                                                                                                           *
*       Usage:                                                                                              *
*           dmat = distmat(a)                                                                               *
*             -or-                                                                                          *
*           dmat = distmat(a,b)                                                                             *
*             -or-                                                                                          *
*           dmat = distmat(a,method)                                                                        *
*             -or-                                                                                          *
*           dmat = distmat(a,b,method)                                                                      *
*                                                                                                           *
*************************************************************************************************************
"""
import os
import numpy as np

def twoPointDistance(matrix1, matrix2=np.asarray([]), method='euclidean'):
    if matrix2.size == 0:
        matrix2 = matrix1.copy()
    if not isinstance(matrix1, np.ndarray):
        raise TypeError("Expecting a numpy array for matrix1, instead of {} type".format(type(matrix1)))
    # Check input dimensionality
    if matrix1.ndim != matrix2.ndim:
        raise IndexError("Input matrices have incompatible dimensions: {} and {}".format(matrix1.shape, 
            matrix2.shape))
    # create index matrices
    set1, set2 = np.meshgrid(np.arange(matrix1.shape[0]), np.arange(matrix2.shape[0]))
    # compute the inter-point differences
    delta = matrix1[set1] - matrix2[set2]
    # compute distance based on specific method
    distMat = np.zeros((matrix1.shape[0], matrix2.shape[0]))
    euclidMethods = ['euclidean', 'euclid']
    cityMethods = ['cityblock', 'city', 'block', 'manhattan', 'taxicab', 'taxi']
    gameMethods = ['chebyshev', 'cheby', 'chessboard', 'chess']
    geomMethods = ['grid', 'diag']

    if np.any(method == euclidMethods):
        distMat = np.sqrt( np.sum(np.square(delta), 1) )
    elif np.any(method == cityMethods):
        distMat = np.sum( np.absolute(delta), 1 )
    elif np.any(method == gameMethods):
        distMat = np.fmax( np.absolute(delta), 1 )
    elif np.any(method == geomMethods):
        distMat = np.fmax(np.absolute(delta), 1) + (np.sqrt(2.0) - 1)*np.fmin(np.absolute(delta), 1)
    else:
        raise TypeError("Unrecognized distance calculation method: {}".format(method))
    
    # return result
    return distMat

"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""
