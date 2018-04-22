'''
Created on Nov 28, 2017

@author: mark.lambrecht

Module Description:
    fraction_threats_engaged - This function computes the fraction of threats engaged as defined by the weapon-target pairing planMat
Class Descriptions:
    N/A
Important Function Descriptions:
    fraction_threats_engaged - computes fraction of threats engaged as defined by the WTA algorithm
Inputs:
    planMat - M Kill Assets x N Threat Objects matrix of assignments (1 = assigned, 0 = not assigned)
Outputs:
    frac - scalar fraction of threats engaged
'''
import numpy as np


def fraction_threats_engaged(planMat):
    nThreatsTpl = planMat.shape
    nThreats = nThreatsTpl[1]
    engaged = np.sum(planMat, axis=0)
    ndx = np.where(engaged > 0)
    nEngaged = np.size(ndx)
    frac = float(nEngaged) / float(nThreats)
    
    return frac


def inventory_expended(planMat):
    engaged = np.sum(planMat, axis=0)
    inventoryExpended = sum(engaged);
    
    return inventoryExpended
