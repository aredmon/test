"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*   Module Name: kMeans                                                                                     *
*   Author(s): Brent McCoy                                                                                  *
*   Version: 1.0                                                                                            *
*   Date: 01/15/18                                                                                          *
*                                                                                                           *
*       Module                                                                                              *
*       Description:    Perform kmeans clustering                                                           *
*                                                                                                           *
*       Input:          dataSet         -   d x n data matrix                                               *
*                       desiredClusters -   number of desired new clusters                                  *
*                                                                                                           *
*       Output:         mu:     - d x k centers of clusters                                                 *
*                       label   -   1 x n sample labels                                                     *
*                                                                                                           *
*                                                                                                           *
*************************************************************************************************************
"""
import numpy as np
from generalUtilities.utilities import vecNorm
    
def kMeans(dataSet, desiredClusters):
    mu = np.zeros((desiredClusters, dataSet.shape[1]))
    isSatisfied = False

    # find the closest mean for each data point
    while not isSatisfied:
        labels = np.random.randint(0, desiredClusters, size=(dataSet.shape[0], 1))
        randIds = np.random.permutation( np.arange(dataSet.shape[0]) )
        subIndices = np.array_split(randIds, desiredClusters)
        counts = np.zeros(desiredClusters)
        newDataSet = np.hstack((dataSet, labels))
        # break up the points into roughly equal sized chunks
        iteration = 0
        for ids in subIndices:
            subSet = dataSet[ids]
            mu[iteration] = np.mean(subSet, axis=0)
            iteration += 1
        # find the minimum distance between each point and the group mean
        for vector in newDataSet:
            vectorId = np.where(newDataSet == vector)[0][0]
            difference = np.subtract(vector[0:3], mu)
            distance = vecNorm(difference, axis=1)
            # figure out which cluster wins and add the vector to that list
            clusterId = np.argmin(distance)
            vector[-1] = clusterId
            labels[vectorId] = clusterId

        # find out how many points are in each cluster, if its not 'equally' 
        # distributed run this again
        for clusterId in range(desiredClusters):
            counts[clusterId] = np.where(newDataSet[-1] == clusterId)[0].size

        if not np.allclose(counts, dataSet.shape[0]/desiredClusters, atol = 1.0):
            isSatisfied = False
        else:
            isSatisfied = True
            break

    # ouput results and labels
    return mu, labels

"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""
