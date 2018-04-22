'''
Created on Nov 21, 2017

@author: mark.lambrecht

Module Description:
    sim_scenario - Creates a single scenario containing the following information:
        1) # threat objects
        2) # Kill Assets x # Threat Objects matrix (1 = can engage, 0 = cannot engage)
        3) # Kill Assets x # Threat Objects matrix of required divert (m/s)
        4) # kill vehicles in each kill asset (# Kill Assets x 1 vector)
        5) # Threat Objects x 1 vector of threat lethality value (P(RV))
    Inputs:
        nWeapons   - # kill assets available for use in engagements
        minThreats - minimum # threat objects in scenario
        maxThreats - maximum # threat objects in scenario
        missile    - class containing constants describing each kill asset
    Outputs:
        nThreats       - # threat objects in scenario
        weaponCoverage - # Kill Assets x # Threat Objects matrix (1 = can cover, 0 = can't cover)
        requiredDivert - # Kill ASsets x # Threat Objects matrix of required divert (m/s)
        pkMat          - # Kill ASsets x # Threat Objects matrix of P(Kill) for each weapon/threat pair
        inventory      - # Kill Assets x 1 vector containing the # kill vehicles in each object (usually 1)
        threatValues   - # Threat Objects x 1 vector containing threat object lethality (P(RV))
'''
import numpy as np


def sim_scenario(nWeapons, minThreats, maxThreats, missile):
    ''' Simulation scenario setup '''
    # constants
    MAX_DIVERT = missile.maxDivert  # maximum divert for each KV, m/s
    AVE_PK = missile.avePk  # average lethality for each KV

    # set up the weapon states
    inventory = np.ones((nWeapons, 1))

    # set up the threat scenario
    nThreats = int(round(Util_UniformRandRange(minThreats, maxThreats)))
    threatValues = np.random.rand(int(nThreats), 1)  # random threat values between 0 and 1
   
    # # overall scenario
    requiredDivert = np.zeros((nWeapons, nThreats))
    weaponCoverage = np.zeros((nWeapons, nThreats))
    pkMat = np.zeros((nWeapons, nThreats))
    # threat loop
    for iThreat in range(0, nThreats):
        # weapon loop
        for iWeapon in range(0, nWeapons):
            # some weapons won't have access
            if np.random.rand(1, 1) < 0.35:
                continue
            
            # if weapon has access, build the divert
            requiredDivert[iWeapon, iThreat] = np.random.rand(1, 1) * MAX_DIVERT
            weaponCoverage[iWeapon, iThreat] = 1
            #         pkMat(iWeapon, iThreat) = min(randn(1,1) * SIG_PK + AVE_PK, MAX_PK)
            pkMat[iWeapon, iThreat] = AVE_PK

    return[nThreats, weaponCoverage, requiredDivert, pkMat, inventory, threatValues]


'''
Module Description:
    Util_UniformRandRange - generates a uniform random number between [low, high]
    Inputs:
        low  - lower limit of random number
        high - upper limit of random number
    Outputs:
        val - random number in range [low, high]
'''


def Util_UniformRandRange(low, high):
    val = low + (high - low) * np.random.rand(1, 1)

    return val
