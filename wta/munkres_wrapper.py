'''
Created on Nov 21, 2017

@author: mark.lambrecht

Module Description:
    This module performs the preparation, execution, and follow-on tasks for
    the Munkres' Optimal Assignment (MOA) algorithm. Preparation includes
    conditioning the cost matrix. The MOA algorithm is called, resulting in
    an initial set of assignments. Post-MOA, an algorithm that assigns any
    un-assigned kill assets is exercised, followed by an attempt to minimize
    overall kill asset divert.
Class Descriptions:
    -N/A
Important Function Descriptions:
    munkres_wrapper - performs weapon-target assignment of multiple kill
                      assets to multiple threat objects
    Inputs:
        cost           - # Kill Assets x # Threats matrix of costs (lower cost is better)
        cutoff         - minimum threat value (lethality) required for engagement (scalar)
        divertMat      - # Kill Assets x # Threats matrix of divert required for each kill asset to reach each threat object (m/s)
        initValues     - # Threats x 1 vector of threat values (lethality)
        pk             - estimated P(kill) for a single kill asset on a single threat object (scalar)
        maxKvsAssigned - maximum number of kill assets assigned to single threat object (scalar)
    Outputs:
        assign - # Kill Assets x # Threats matrix of assignments (1 = assigned, 0 = not assigned)
'''
from munkres import munkres
import numpy as np


def munkres_wrapper(cost, cutoff, divertMat, initValues, pk, maxKvsAssigned):
    # condition the cost matrix by removing unavailable rows and columns
    allowed = np.logical_not(np.floor(cost))
    cutoffMat = (np.ones(cost.shape)) * cutoff
    costAdj = np.subtract(cost, cutoffMat)
    ndx = np.where(costAdj <= 0)
    costAdj[ndx] = 0
    ndx = np.where(costAdj >= (1 - cutoff))
    costAdj[ndx] = 1
    
    # MOA algorithm
    assignTmp = munkres(costAdj)

    # convert from True/False to 1/0 matrix
    assign = np.zeros(assignTmp.shape)
    iRow = 0
    for x in assignTmp:
        iCol = 0
        for y in x:
            if y:
                assign[iRow, iCol] = 1
            iCol = iCol + 1 
        iRow = iRow + 1   
        
    # remove any "not allowed" assignments (just in case)                
    assign = np.multiply(assign, allowed)
    
    # update threat values based on assignments
    values = np.zeros(initValues.shape)
    nAssigned = np.zeros(initValues.shape)
    iRow = 0
    for row in assign:
        iCol = 0
        for elem in row:
            if elem == 1:
                values[iCol] = initValues[iCol] * (1.0 - pk)
                nAssigned[iCol] = nAssigned[iCol] + 1
            else:
                values[iCol] = initValues[iCol]
            iCol = iCol + 1
        iRow = iRow + 1
    ndx = np.where(values < cutoff)
    values[ndx] = 0
    
    # assign unassigned KVs
    iRow = 0
    for row in assign:
        ndx = np.where(row > 0)
        if np.size(ndx) == 0:
            # unassigned KV found, make an assignment if possible
            allowedRow = allowed[iRow]
            ndx2 = np.where(allowedRow == False)
            tmpValues = np.copy(values)
            tmpValues[ndx2] = 0;
            maxIndx = np.argmax(tmpValues)  # highest value threat
            assign[iRow, maxIndx] = 1
            nAssigned[maxIndx] = nAssigned[maxIndx] + 1
            if nAssigned[maxIndx] >= maxKvsAssigned:
                values[maxIndx] = 0
        iRow = iRow + 1  # go to next row
        
    # minimize divert if possible
    nWeapons = assign.shape[0];
    nThreats = assign.shape[1];
    for iWpn in range(0, nWeapons):
        for iThrt in range(0, nThreats):
            for jWpn in range(0, nWeapons):
                for jThrt in range(0, nThreats):
                    if iWpn == jWpn and iThrt == jThrt:
                        continue  # same assignment
                    if assign[iWpn, iThrt] == 1 and assign[jWpn, jThrt] == 1:
                        if divertMat[iWpn, jThrt] < divertMat[jWpn, iThrt] and divertMat[iWpn, jThrt] > 0:
                            # swap assignments to minimize divert
                            assign[iWpn, iThrt] = 0
                            assign[iWpn, jThrt] = 1
                            assign[jWpn, jThrt] = 0
                            assign[jWpn, iThrt] = 1

    # return the assignments                            
    return assign
