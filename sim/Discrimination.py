"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*   ModuleName: DiscriminationGround                                                                        *
*   Author(s):  M. A. Lambrecht, P. B. McCoy                                                                *
*   Version:    1.0                                                                                         *
*   Date:       01/02/18                                                                                    *
*                                                                                                           *
*       Module          This function simulates the RF discrimination using                                 *
*       Description:    K-factors.                                                                          *
*                                                                                                           *
*       Algorithm:      1) Generate the instantaneous P(lethal) values                                      *
*                       2) Average the last 'nAve' values to generate the final                             *
*                          probabilities                                                                    *
*                                                                                                           *
*       Inputs:         nThreats - (Scalar) Number of threats                                               *
*                       rvID     - (Scalar) ID of threat that is the RV                                     *
*                       kFact    - (Scalar) K-factor for discrimination                                     *
*                                                                                                           *
*       Outputs:        pLethal - nThreats x 1 vector of probability of lethality                           *
*                                                                                                           *
*       Calls:                                                                                              *
*                                                                                                           *
*       History:        MAL 14 Dec 2017:  Initial version                                                   *
*                                                                                                           *
*************************************************************************************************************
"""
import os
import numpy as np

# RF discrimination
def DiscriminationGround(nThreats, rvID, kFact):
    # constants
    nAve = 25
    if not os.path.isdir('sim'):
        if os.getcwd().rsplit(sep='/', maxsplit=1)[-1] != 'sim':
            currentDir = os.path.dirname(os.path.realpath(os.getcwd()))
            filename = os.path.join(currentDir, 'sim', 'persistentVarsGround.npz')
        else:
            filename = 'persistentVarsGround.npz'
    else:
        filename = 'sim/persistentVarsGround.npz'

    if os.path.isfile(filename):
        currentData = np.load(filename)
        pLethalSave = currentData['pLethal']
        n = int(currentData['count'])
    else:
        pLethalSave = np.zeros([nThreats, nAve])
        plOnboard = np.zeros([nThreats, nAve])
        n = 0

    # compute the instantanqous lethality measurements
    pL = np.zeros((nThreats, 1))
    for iThreat in range(nThreats):
        if iThreat == rvID:
            randnum = np.random.randn()
        else:
            randnum = np.random.randn() + kFact

        pRV = np.exp(-0.5 * randnum**2) / np.sqrt(2*np.pi)
        pDecoy = np.exp(-0.5 * (randnum - kFact)**2) / np.sqrt(2*np.pi)
        pL[iThreat, 0] = pRV / (pRV + pDecoy)

    if n >= nAve:
        pLethalSave[:, 0:nAve-1] = pLethalSave[0:nThreats, 1:nAve]
        pLethalSave = np.append(pLethalSave, pL, axis=1)
    else:
        pLethalSave[:, n] = pL[:, 0]

    n += 1
    pLethal = np.mean(pLethalSave, axis=1)

    # save pLethalSave and n to file
    np.savez(filename, pLethal=pLethalSave, count=n)

    return pLethal
"""
TODO:   Look at additional distributions for lethality estimates, need to have a way to refine 
        lethality based on the TOMCorrelation information.
"""

# EOIR discrimination
def DiscriminationOnboard(nThreats, rvID, kFact):
    # constants
    nAve = 25
    if not os.path.isdir('sim'):
        if os.getcwd().rsplit(sep='/', maxsplit=1)[-1] != 'sim':
            currentDir = os.path.dirname(os.path.realpath(os.getcwd()))
            filename = os.path.join(currentDir, 'sim', 'persistentVarsOnboard.npz')
        else:
            filename = 'persistentVarsOnboard.npz'
    else:
        filename = 'sim/persistentVarsOnboard.npz'

    if os.path.isfile(filename):
        currentData = np.load(filename)
        pLethalSave = currentData['pLethal']
        n = int(currentData['count'])
    else:
        pLethalSave = np.zeros([nThreats, nAve])
        n = 0

    # compute the instantanqous lethality measurements
    pL = np.zeros((nThreats, 1))
    for iThreat in range(nThreats):
        if iThreat == rvID:
            randnum = np.random.randn()
        else:
            randnum = np.random.randn() + kFact

        pRV = np.exp(-0.5 * randnum**2) / np.sqrt(2*np.pi)
        pDecoy = np.exp(-0.5 * (randnum - kFact)**2) / np.sqrt(2*np.pi)
        pL[iThreat, 0] = pRV / (pRV + pDecoy)

    if n >= nAve:
        pLethalSave[:, 0:nAve-1] = pLethalSave[0:nThreats, 1:nAve]
        pLethalSave = np.append(pLethalSave, pL, axis=1)
    else:
        pLethalSave[:, n] = pL[:, 0]

    n += 1
    pLethal = np.mean(pLethalSave, axis=1)

    # save pLethalSave and n to file
    np.savez(filename, pLethal=pLethalSave, count=n)

    return pLethal
"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""
