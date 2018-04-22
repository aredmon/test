"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*   Module Name: MunkresPairing                                                                             *
*   Author(s): Mark Lambrecht, Brent McCoy                                                                  *
*   Version: 1.0                                                                                            *
*   Date: 02/5/18                                                                                           *
*                                                                                                           *
*       Module          given the matrix M(m,n) of real numbers, find a permutation                         *
*       description:    perm(m) of the integers 1,2,3 ... mrows that minimizes                              *
*                       sum( M(m,perm(m)) ).                                                                *
*                                                                                                           *
*       Inputs:                                                                                             *
*                       costMat   - [MxN] Cost matrix used to define optimal assignement.                   *
*                                   costMat(ij) is the cost associated with the peroformance                *
*                                   of task j by agent i.                                                   *
*                                                                                                           *
*                       cutoff    - scalar decimal value used to ajust the costMat to a desired             *
*                                   threshold value                                                         *
*                                                                                                           *
*                       divertMat - [MxN] divert matrix containing required divert for each                 *
*                                   weapon to get to each threat.                                           *
*                                                                                                           *
*                       threatVal - [Mx1] array of threat values.                                           *
*                                                                                                           *
*                       pk        - Single shot engagement probability of kill.                             *
*                                                                                                           *
*       Outputs:                                                                                            *
*                       assign    - [MxN] Optimal assignment matrix denoting which agents                   *
*                                   are to be assigned to which tasks. ASSIGNij = 1                         *
*                                   denotes the assignment of agent i to task j.                            *
*                                                                                                           *
*       References:     Bourgeois, F. and Lassalle, J.C., An Extension of the Munkres                       *
*                       Algorithm for the Assignment Problem to Rectangular Matrices,                       *
*                       Comm. ACM, Vol. 14, Number 12, (Dec 1971), 802-4.                                   *
*                                                                                                           *
*************************************************************************************************************
"""
import numpy as np
import importModules as mods

def MunkresPairing(costMat, cutoff, divertMat, threatVal, pk):
    assignTemp = np.zeros_like(costMat)

    # condition the cost matrix by removing unavailable rows and columns
    allowedAssigns = np.ceil(costMat)
    adjustedCost = costMat - cutoff
    adjustedCost[ ((adjustedCost < 0) | (adjustedCost == -0)) ] = 0
    available = np.ceil( adjustedCost/np.amax(adjustedCost) )
    # if available is all false, set it to all true
    if not np.any(available):
        available = np.ones_like(costMat)
    # find any rows or columns that are empty
    availableRows = np.any(available, 1)
    emptyRows = np.where(~np.any(available, 1))[0]
    availableCols = np.any(available, 0)
    emptyCols = np.where(~np.any(available, 0))[0]
    unAssignedKVs = np.where(availableRows == 0)[0]
    firstCut = costMat[availableRows, :]
    finalCut = firstCut[:, availableCols]

    redAllowed = allowedAssigns[availableRows, :]
    allowed = redAllowed[:, availableCols]

    _, _, _, assignTemp = mods.ModMunkres2(finalCut, mask=bool)
    # adjust assign by allowed assignments matrix
    assign = assignTemp * allowed

    # pad assign matrix to match original costMat shape
    # if necessary
    if emptyRows.size > 0:
        assign = np.insert(assign, emptyRows, np.zeros(assign.shape[1]), 0)

    if emptyCols.size > 0:
        assign = np.insert(assign, emptyCols, np.zeros(assign.shape[0]), 1)

    unAssignedThreats = np.where(~np.any(assign, 0))[0]
    disallowedThreats = np.where(~np.any(allowed, 0))[0]
    availableThreats = np.setdiff1d(unAssignedThreats, disallowedThreats)

    # assign KVs that are currently unassigned here
    # original costMat should have shape: nKVs x nThreats
    # could have an under-assignment if assign.shape[0] < nKVs
    # or there are empty rows in the assign matrix
    minCost = np.finfo(float).max
    for kvID in unAssignedKVs:
        currentAssign = -1*np.ones(2)
        for threatID in availableThreats:
            if adjustedCost[kvID, threatID] < minCost:
                minCost = adjustedCost[kvID, threatID]
                currentAssign[:] = [kvID, threatID]
        # assign the kv to the id with the lowest cost available threat
        if not np.array_equal(currentAssign, -1*np.ones(2)):
            assign[int(currentAssign[0]), int(currentAssign[1])] = 1
            # update availableThreats by removing the one that was just assigned
            disallowedThreats = np.append(disallowedThreats, currentAssign[1])
            availableThreats = np.setdiff1d(unAssignedThreats, disallowedThreats)

    # minimize divert if possible
    nWeapons, nTargets = assign.shape
    for iwpn in range(nWeapons):
        for itgt in range(nTargets):
            for jwpn in range(nWeapons):
                for jtgt in range(nTargets):
                    if iwpn == jwpn and itgt == jtgt:
                        continue
                    if assign[iwpn, itgt] == 1 and assign[jwpn, jtgt] == 1:
                        if divertMat[iwpn, jtgt] < divertMat[jwpn, itgt] and divertMat[iwpn, jtgt] > 0:
                            assign[iwpn, itgt] = 0
                            assign[iwpn, jtgt] = 1
                            assign[jwpn, jtgt] = 0
                            assign[jwpn, itgt] = 1
    
    # lethality adjustment and allocation of weapons
    targetedThreats = np.where(assign == 1)[1]

    if targetedThreats.size > 0:
        threatVal[targetedThreats] = threatVal[targetedThreats] * (1.0 - pk)

    for iWeapon in range(assign.shape[0]):
        if np.array_equal(assign[iWeapon], np.zeros(assign.shape[1])):
            # allocate KV that has not been allocated yet
            possibleThreats = np.where(divertMat[iWeapon] > 0)[0]
            idealTarget = np.argmax(threatVal[possibleThreats])
            assign[iWeapon, idealTarget] = 1
            threatVal[idealTarget] = threatVal[idealTarget] * (1.0 - pk)

    # release the results
    finalAssign = assign.astype(bool)
    return assign, finalAssign, threatVal

"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""
