'''
Created on Nov 29, 2017

@author: mark.lambrecht

Module Description:
    Uses a recursive linear indexing method to calculate all possible
    combinations of N agents assigned to M tasks with at most nShot
    agents assigned to any one task.
Class Descriptions:
    N/A
Inputs:
    assmt - [NxM] matrix defining the weapon/threat assignments used 
            to calculate the leakage (M = # weapons, N = # threats)
    nT    - [Nx1] vector containing the number of engage-able threats 
            for each of the remaining unassigned weapons.
    lOpt  - [Scalar] value of the threat leakage for the current best 
            assignment matrix.
    aOpt  - [NxM] matrix denoting the current best set of 
            weapon/threat assignments.
    nShot - [Scalar] value of the maximum number of weapons that can 
            engage the same threat.
    poss  - [sum(nT),1] vector containing the linear indices of the 
            remaining possible weapon/threat assignments.
    pkMat - [NxM] matrix defining the p(kill) values for each 
            weapon/threat pair.
    tVal  - [1xM] vector defining the lethality (P(RV)) of each threat.
Outputs:
    aOpt  - [NxM] optimal assignment matrix denoting which agents are 
            to be assigned to which tasks. ASSIGNij = 1 denotes the 
            assignment of weapon j to threat i.
    lOpt  - [Scalar] value of the threat leakage for the current best 
            assignment matrix.
'''
from lethality_leakage import lethality_leakage
import numpy as np


def assign_opt(assmt, nT, nShot, poss, pkMat, tVals, lOpt=0, aOpt=[]):
    if lOpt == 0:
        lOpt = 1
        aOpt = assmt.copy();
        if nT.size == 0:
            return [aOpt, lOpt]
    assmt0 = np.copy(assmt)
    for ii in range(0, int(nT[0])):
        if ii > 0:
            assmt[poss[ii]] = True;
        if nT.size == 1:
            leakage = lethality_leakage(pkMat, assmt.T, tVals)
            if leakage < lOpt and not np.any(np.sum(assmt.T), axis=0) > nShot:
                lOpt = leakage
                aOpt = assmt
        else:
            n = nT[0] + 1
            [aOpt, lOpt] = assign_opt(assmt, nT[1:nT.size], nShot, poss[(nT[0] + 1):poss.size, 0], pkMat, tVals, lOpt, aOpt)
        assmt = np.copy(assmt0)
