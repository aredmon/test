'''
Created on Nov 27, 2017

@author: mark.lambrecht

Module Description:
    greedy - Performs weapon-target pairing given M-kill assets and N-threat objects
             using greedy optimization techniques. Post-greedy, the algorithm attempts
             to assign any unassigned kill assets. The Greedy algorithm is penalty-based,
             using P(Kill), Global Coverage, and Divert Required as penalties/scores.
Class Descriptions:
    -N/A
Important Function Descriptions:
    greedy - weapon-target pairing for M Kill assets and N threat objects
    Inputs:
        doctrine     - contains the firing doctrine and associated constants for used in the algorithms
        nWES         - # kill assets available for engagement
        nTgt         - # threat objects in target complex
        pkMat        - M Kill Assets x N Threat Objects matrix. Each element is the P(Kill) for that 
                       kill asset on that threat object
        divertMat    - M Kill Assets x N Threat Objects matrix. Each element is the divert required 
                       (m/s) for the kill asset to reach the threat object
        wpnInv       - M Kill Assets x 1 vector containing the # of kill vehicles associated with each 
                       kill asset (usually 1)
        threatValues - N Threat Objects x 1 vector containing the lethality (P(RV)) associated with 
                       each threat object
    Outputs:
        planMat      - M Kill Assets x N Threat Objects matrix. 1 = assigned, 0 = not assigned
'''
import numpy as np


def greedy(doctrine, nWES, nTgt, pkMat, divertMat, wpnInv, threatValues):
    # initialization
    planMat = np.zeros((nWES, nTgt))
#     nAlloc = nTgt * doctrine.maxShotsPerThreat
#     valI = np.zeros((nAlloc, 1))
#     valF = np.zeros((nAlloc, 1))
    alloc = np.zeros((nWES, nTgt))
    tgtVal = np.copy(threatValues)
    origPkMat = np.copy(pkMat)
    
    # ensure each viable candidate can fit in the timeline of the resources
    if not more_shots(pkMat):
        return planMat
    
    # compute global penalties (coverage). idea is to choose weapons that 
    # can only cover a given threat for that threat - vice other weapons 
    # that can cover multiple threats
    coverageScore = coverage_score(nTgt, nWES, pkMat)
    
    # main weapon-target allocation loop
    nShotsWeapon = np.zeros((nWES, 1))
    tgtValTmp = np.copy(tgtVal)
    i_alloc = 0
    finished = False
    
    while not finished:
        # no shots for out of inventory weapons
        ndx = np.where(wpnInv == 0)
        pkMat[ndx, :] = 0
        
        # find the target with the highest value and any candidate shots against it
        shots = False
        while not shots:
            highValThreatIndx = np.argmax(tgtValTmp)
            val = tgtValTmp[highValThreatIndx]
            if val <= 0:
                finished = True
                break
#                 return [planMat, valI, valF]  # no more threat value to engage
            shots = more_shots_threat(pkMat, highValThreatIndx)
            if not shots:
                tgtValTmp[highValThreatIndx] = 0
        if finished:
            break
            
        # divert penalty - used to preferentially select weapons requiring less divert
        divertScore = divert_score(highValThreatIndx, divertMat)
        
        # find the best shot against the highest value threat
        maxScore = 0
        for iWES in range(0, nWES):
            if nShotsWeapon[iWES] < doctrine.maxShotsPerThreat:
                score = doctrine.pKWeight * pkMat[iWES, highValThreatIndx] + \
                doctrine.covWeight * coverageScore[iWES] + \
                doctrine.divWeight * divertScore[iWES]
                if score[0] > maxScore and pkMat[iWES, highValThreatIndx] != 0:
                    maxIndScore = iWES
                    maxScore = score
        if maxScore <= 0:
            finished = True
            break
#             return [planMat, valI, valF]
        nShotsWeapon[maxIndScore] = nShotsWeapon[maxIndScore] + 1
        
        # bookkeeping for chosen plan
        i_alloc = i_alloc + 1
        chosenPk = pkMat[maxIndScore, highValThreatIndx]
        planMat[maxIndScore, highValThreatIndx] = 0
        wpnInv[maxIndScore] = wpnInv[maxIndScore] - 1
        alloc[maxIndScore, highValThreatIndx] = alloc[maxIndScore, highValThreatIndx] + 1
#         valI[i_alloc] = tgtValTmp[highValThreatIndx]
        tgtValTmp[highValThreatIndx] = tgtValTmp[highValThreatIndx] * (1.0 - chosenPk)
#         valF[i_alloc] = tgtValTmp[highValThreatIndx]
        sumVal = alloc.sum(axis=0)
        val = sumVal[highValThreatIndx]
        if tgtValTmp[highValThreatIndx] <= doctrine.valueCutoff or val >= doctrine.maxShotsPerThreat:
            tgtValTmp[highValThreatIndx] = 0
            pkMat[:, highValThreatIndx] = 0
#             for row in pkMat:
#                 row[highValThreatIndx] = 0
        
        # check for stopping condition
        ndx = np.where(tgtValTmp > doctrine.valueCutoff)
        if np.size(ndx) == 0:
            finished = True
            break
#             return [planMat, valI, valF]  # all targets now below cutoff
        if not more_shots(pkMat):
            finished = True  # no more candidates

    # update threat values based on assignments
    values = np.zeros(threatValues.shape)
    nAssigned = np.zeros(threatValues.shape)
    iRow = 0
    for row in planMat:
        iCol = 0
        for elem in row:
            if elem == 1:
                values[iCol] = threatValues[iCol] * (1.0 - origPkMat[iRow, iCol])
                nAssigned[iCol] = nAssigned[iCol] + 1
            else:
                values[iCol] = threatValues[iCol]
            iCol = iCol + 1
        iRow = iRow + 1
    ndx = np.where(values < doctrine.valueCutoff)
    values[ndx] = 0
    
    # assign unassigned KVs
    nAssigned = np.zeros(threatValues.shape)
    allowed = np.zeros(planMat.shape, dtype=bool)
    allowed[np.where(divertMat > 0)] = True;
    iRow = 0
    for row in planMat:
        ndx = np.where(row > 0)
        if np.size(ndx) == 0:
            # unassigned KV found, make an assignment if possible
            allowedRow = allowed[iRow]
            ndx2 = np.where(allowedRow == False)
            tmpValues = np.copy(values)
            tmpValues[ndx2] = 0;
            maxIndx = np.argmax(tmpValues)  # highest value threat
            planMat[iRow, maxIndx] = 1
            nAssigned[maxIndx] = nAssigned[maxIndx] + 1
            if nAssigned[maxIndx] >= doctrine.maxShotsPerThreat:
                values[maxIndx] = 0
        iRow = iRow + 1  # go to next row
    
    return planMat


'''
Module Description:
    more_shots - determines if more potential engagements remain (any threat, any kill asset)
    Inputs:
        pkMat - M Kill Asset x N Threat Object matrix of P(Kill) values for each kill asset/threat object pair
    Outputs:
        moreShots - true if more shots exist, else false
'''


def more_shots(pkMat):
    moreShots = False
    for row in pkMat:
        ndx = np.where(row > 0)
        if np.size(ndx) > 0:
            moreShots = True
            break
    return moreShots


'''
Module Description:
    more_shots_threat - determines if more potential engagements remain (specific threat, any kill asset)
    Inputs:
        pkMat - M Kill Asset x N Threat Object matrix of P(Kill) values for each kill asset/threat object pair
        iTgt  - index of threat of interest
    Outputs:
        moreShots - true of more shots exist, else false
'''


def more_shots_threat(pkMat, iTgt):
    moreShots = False
    pkMatTranspose = pkMat.T
    col = pkMatTranspose[iTgt]
    ndx = np.where(col > 0)
    if np.size(ndx) > 0:
        moreShots = True
    return moreShots


'''
Module Description:
    coverage_score - Computes the global coverage penalty/score for each weapon. Idea is to reserve weapons that
                     can only engage a specific threat for that threat, instead of assigning another kill asset
                     that can engage other objects to that threat. A higher coverage score will result in a better
                     chance of being selected
    Inputs:
        nTgt  - # threat objects available for engagement
        nWES  - # kill assets available for engagement
        pkMat - M Kill Asset x N Threat Object matrix of P(Kill) values for each kill asset/threat object pair
    Outputs:
        coverageScore - M Kill Asset x 1 vector of coverage scores for each kill asset
'''


def coverage_score(nTgt, nWES, pkMat):
    # build the coverage table
    wesCoverage = np.zeros((nWES, nTgt))
    ndx = np.where(pkMat > 0)
    wesCoverage[ndx] = 1
    wesEngThreats = np.zeros((nWES, 1))
    
    # compute the total number of engage-able threats for each target
    totEngThreats = 0
    wesCoverageTranspose = wesCoverage.T
    for col in wesCoverageTranspose:
        ndx = np.where(col > 0)
        if np.size(ndx) > 0:
            totEngThreats = totEngThreats + 1
            
    # compute the number of engage-able threats for each target
    iRow = 0
    for row in wesCoverage:
        ndx = np.where(row > 0)
        wesEngThreats[iRow] = np.size(ndx)
        iRow = iRow + 1
        
    # basis element calculation
    x = wesEngThreats * (1.0 / totEngThreats)
    
    # compute the score
    coverageScore = 1.0 - x
    return coverageScore


'''
Module Description:
    divert_score - Computes the divert penalty/score for each weapon. Idea is to penalize engagements that require
                   higher divert usage more than those that require less divert.
    Inputs:
        trgtIndx  - index of threat of interest
        divertMat - M Kill Asset x N Threat Object matrix of required divert for each kill asset/threat object pair
    Outputs:
        coverageScore - 1 x M Kill Asset vector of divert scores for each kill asset
'''


def divert_score(trgtIndx, divertMat):
    divertScore = np.zeros((divertMat.shape[0]))
    divertMatTranspose = divertMat.T
    diverts = divertMatTranspose[trgtIndx]
    badInd = np.where(diverts <= 0)
    divertsGood = np.copy(diverts)
    divertsGood[badInd] = 0
    maxDivert = np.max(divertsGood)
    for iScore in range(0, np.size(divertScore)):
        divertScore[iScore] = 1.0 - divertsGood[iScore] / maxDivert
    return divertScore
