"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*   Module Name: DeployKVsTOM                                                                               *
*   Author(s): Mark Lambrecht, Brent McCoy                                                                  *
*   Version: 1.0                                                                                            *
*   Date: 01/04/18                                                                                          *
*                                                                                                           *
*       Module        This function performs the initial KV assignments in order to                         *
*       Description:  perform zone defense of the threats.  It is called once at or                         *
*                     shortly after dispense.  This function assumes that a good TOM                        *
*                     is furnished by the weapon system, or that the CV has performed                       *
*                     valid acquisition prior to use.                                                       *
*                                                                                                           *
*       Algorithm:    1) Perform initial clustering of the threats based on kinematic                       *
*                        reach                                                                              *
*                     2) Determine the number of KVs to send to each cluster based on                       *
*                        # threats and estimated lethality of those threats                                 *
*                     3) Cluster the threats of each cluster into the number of KVs                         *
*                        assigned to the cluster                                                            *   
*                     4) Divert KVs to the sub-cluster locations                                            *
*                                                                                                           *
*       Inputs:       cvState      - 7x1 CV state (x,y,z,dx,dy,dz,t), [m,m/s,s]                             *
*                     cvFuel       - 1x1 available delta-V on the CV [m/s]                                  *
*                     kvStates     - #KVsx7 KV states (x,y,z,dx,dy,dz,t), [m,m/s,s]                         *   
*                     kvFuel       - #KVsx1 available delta-V on the KVs [m/s]                              *
*                     threatStates - #Threatsx7 threat states (x,y,z,dx,dy,dz,t),                           *
*                                    [m,m/s,s]                                                              *
*                     pLethal      - #Threatsx1 P(RV) values                                                *
*                     tPOCA        - (Scalar) Estimated time of intercept, s                                *
*                     rvID         - (Scalar) ID of RV                                                      *
*                     aH1          - axis handle for plotting                                               *
*                                                                                                           *
*       Outputs:      cvState      - 7x1 CV state (x,y,z,dx,dy,dz,t), [m,m/s,s]                             *
*                     cvFuel       - (Scalar) available delta-V on the CV [m/s]                             *
*                     kvStates     - #KVsx7 KV states (x,y,z,dx,dy,dz,t), [m,m/s,s]                         *
*                     kvFuel       - #KVsx1 available delta-V on the KVs [m/s]                              *
*                     clusters     - 1x#clusters structure:                                                 *
*                                       ctr    - 1x3 cluster center position [m]                            *
*                                       pts    - #ptsx3 cluster points positions [m]                        *
*                                       states - #ptsx7 point states(x,y,z,dx,dy,dz,t)                      *
*                                                [m, m/s, s]                                                *
*                                       ids    - 1x#pts IDs of points in cluster                            *
*                                       tPOCA  - 1x3pts time of intercept for pts                           *
*                                       kr     - 1x1 delta-V to reach cluster                               *
*                                                                                                           *
*       Calls:        SAPs_Null                                                                             *
*                     Cluster                                                                               *
*                     KMeans                                                                                *
*                     Util_GaussProblem                                                                     *
*                                                                                                           *
*       OA:           M.A. Lambrecht                                                                        *
*                                                                                                           *
*       History:      M.A. Lambrecht, 14 Dec 2017 - Initial version                                         *
*                                                                                                           *
*************************************************************************************************************
"""
import os
import numpy as np
import importModules as mods
#from generalUtilities.Classes import SimpleNamespace, jsonData
#from generalUtilities.kMeans import kMeans
#from generalUtilities.Cluster import Cluster

def DeployKVsTOM(cvState, cvFuel, kvStates, kvFuel, threatStates, threatIds, pLethal, tPOCA, rvID, aH1):
    # cluster the threats into manageable regions
    clusterList = mods.Cluster(cvState, kvFuel, threatStates, threatIds, tPOCA, aH1)

    # Now compute the number of KVs to send to each cluster based on the number of 
    # threats and their lethality ranking
    nClusters = len(clusterList)
    pLethalNorm = pLethal / np.sum(pLethal)
    weights = np.zeros(nClusters)
    indx = 0
    for cluster in clusterList:
        weights[indx] = np.sum( pLethal[cluster.ids] )
        indx += 1
    
    nAssigned = np.round(weights * mods.SAPs.KV_NUM_KVS)
    
    # If there are any left over, assign them to the cluster with the largest number of threats.
    # If there are too many assigned, get rid of one with the most assignments.
    nLeftOver = mods.SAPs.KV_NUM_KVS - sum(nAssigned)
    if nLeftOver != 0:
        if nLeftOver < 0 and nAssigned.size > 0:
            #adjIndex = np.unravel_index(nAssigned.argmax(), dims=nAssigned.shape)
            adjIndex = nAssigned.argmax()
        elif nLeftOver > 0 and nAssigned.size > 0:
            #adjIndex = np.unravel_index(weights.argmax(), dims=nAssigned.shape)
            adjIndex = weights.argmax()
        # adjust nAssigned accordingly    
        try:
            nAssigned[adjIndex] = nAssigned[adjIndex] + nLeftOver
        except UnboundLocalError:
            print("something might have broke? \n nAssigned: {} \n nLeftOver: {}".format(nAssigned, nLeftOver))

    # Now divide each cluster into the same number of clusters as there are KVs assigned to it.
    # Compute their centers as well (with k-means clustering)
    clusterId = 0
    nKV_total = mods.SAPs.KV_NUM_KVS
    nKV = 0
    kvPIP = np.zeros_like(kvStates)
    kvPOCA = np.zeros((kvStates.shape[0], 3))
    kvClust = np.zeros_like(kvStates, dtype=int)
    kvKillRadius = np.zeros(kvStates.shape[0])
    for cluster in clusterList:
        nKVsAssigned = int( nAssigned[clusterId] )
        if nKVsAssigned > 0:
            pts = cluster.pts
            nPts = pts.shape[0]
            if nKVsAssigned == 1 or nPts == 1:
                icx = np.zeros(1,int)
                mu = np.average(pts, axis=0)
            else:
                mu, icx = mods.kMeans(pts, int( min(pts.shape[0], nKVsAssigned) ))
            # build the PIP state for each KV
            clstIds = np.unique(icx)
            print("clusterIds: {}".format(clstIds))
            nClsts = clstIds.shape[0]
            kvStart = nKV
            for iKV in clstIds:
                print("nKV: {}, iKV: {}, nKVs: {}, nClsts: {}".format(
                    nKV, iKV, kvStates.shape[0], clstIds.shape[0]))
                nKVmod = nKV % nKV_total 
                kvPIP[nKVmod, 0:3] = mu[iKV]
                kvClust[nKVmod] = iKV
                nKV += 1
            # find out how many KVs still need to be assigned
            nKVsLeftOver = nKVsAssigned - nClsts
            for iKV in range(nKVsLeftOver):
                nKVmod = nKV % nKV_total 
                kvPIP[nKVmod, 0:3] = cluster.ctr
                kvClust[nKVmod] = clusterId
                nKV += 1
            # Perform the KV diverst
            #       tStates = cluster.states
            #       centroid = np.average(tStates, axis=1)
            # divert the KVs to their guard positions
            for iKV in range(kvStart, kvStart + nKVsAssigned):
                iKV = iKV % nKV_total
                tGo = cluster.tPOCA - kvStates[iKV, 6]
                if np.linalg.norm(kvStates[iKV, 0:3]) == 0 or np.linalg.norm(kvPIP[iKV, 0:3]) == 0: 
                    print("Trying to use GaussProblem on an origin vector.")
                    print("pos1: {}, pos2: {}".format(kvStates[iKV, 0:3], kvPIP[iKV]))
                    kv = mods.PropagateECI(kvStates[iKV], 0.0, tGo)
                    poca = mods.PropagateECI(kvPIP[iKV], 0.0, tGo)
                else:
                    kv, poca = mods.GaussProblem(kvStates[iKV, 0:3], kvPIP[iKV, 0:3], tGo)
                # calculate dV
                dV = kv[3:6] - kvStates[iKV, 3:6]
                kvStates[iKV, 3:6] = kvStates[iKV, 3:6] + dV
                kvFuel[iKV] = kvFuel[iKV] - np.linalg.norm(dV)
                kvPOCA[iKV] = poca[0:3]
            # create kvKillRadius vector:
            kvKillRadius = (kvFuel - mods.SAPs.KV_FUEL_RESERVE) * mods.SAPs.CV_FINAL_ASSGN_TGO

    return cvState, cvFuel, kvStates, kvFuel, clusterList

"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""
