'''
Created on Nov 28, 2017

@author: mark.lambrecht

Module Description:
    This function computes the lethality leakage across the scenario.
    Lethality leakage is defined as:
    Sum_j(lethality_j * Prod_i(1 - Pk_i)) / Sum_j(lethality_j)
Class Descriptions:
    N/A
Important Function Descriptions:
    lethality_leakage - computes the lethality leakage across the scenario
Inputs:
    pkMat        - M Kill Assets x N Threat Objects matrix of estimated P(Kill) for each potential engagement
    planMat      - M Kill Assets x N Threat Objects matrix of assignments (1 = assigned, 0 = not assigned)
    threatValues - N Threat Objects x 1 vector of threat lethality (P(RV)) estimates
Outputs:
    leakage - lethality leakage across entire scenario
'''
import numpy as np


def lethality_leakage(pkMat, planMat, threatValues):
    leakage = 0.0
    nThreats = np.size(threatValues)
    for iThrt in range(0, nThreats):
        lethality = threatValues[iThrt]
        ndx = np.where(planMat[:, iThrt] > 0)
        if np.size(ndx) > 0:
            for iWpn in range(0, np.size(ndx)):
                lethality = lethality * (1.0 - pkMat[ndx[0][iWpn], iThrt])
        leakage = leakage + lethality
    leakage = leakage / np.sum(threatValues)
    
    return leakage
