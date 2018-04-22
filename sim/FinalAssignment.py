"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*   Module Name: Cluster                                                                                    *
*   Author(s): Mark Lambrecht, Brent McCoy                                                                  *
*   Version: 1.0                                                                                            *
*   Date: 01/26/18                                                                                          *
*                                                                                                           *
*       Module        Determine when it is time for final assignment and divert for                         *
*       Description:  each KV. For those that are deemed final, make the divert.                            *
*                                                                                                           *
*                                                                                                           *
*       Algorithm:    N/A                                                                                   *
*                                                                                                           *
*       Inputs:       planMat      - weapon-target pairing matrix                                           *
*                                    size of matrix is #KVS x #Threats                                      *
*                     kvStates     - #KVs x 7 KV state vectors, ECI                                         *
*                     kvFuel       - #KVs x 1 KV remaining divert, m/s                                      *
*                     threatStates - #Threats x 7 threat state vectors, ECI                                 *
*                     threatIds    - #Threats x 1 with unique threat IDs                                    *
*                     dT           - (Scalar) time frame for which to check divert,                         *
*                                    seconds                                                                *
*                                                                                                           *
*       Outputs:      finalAssign - #KVs x 1 final assignment vector. value > 0                             *
*                                   indicates a final assignment has been made                              *
*                     kvStates    - #KVs x 7 KV state vectors, ECI (updated for final                       *
*                                   assignment diverts)                                                     *
*                                                                                                           *
*       Calls:        SAPs_Null                                                                             *    
*                     Util_PropagateECI                                                                     *
*                     Util_FindPOCA                                                                         *
*                     Util_GaussProblem                                                                     *
*                                                                                                           *
*       OA:           Mark lambrecht                                                                        *
*                                                                                                           *
*       History:      MAL, 20 Dec 2017 - Initial version                                                    *
*                                                                                                           *
*************************************************************************************************************
"""
import os
import numpy as np
import importModules as mods

def FinalAssignment(planMatrix, kvStates, kvFuel, threatStates, threatIds, deltaT):
    # system adjustable parameters
    nKVs = mods.SAPs.KV_NUM_KVS

    # check each KV for a "need-to-divert now" condition
    finalAssign = -1*np.ones(nKVs)
    for iKV in range(nKVs):
        # find the assignment
        thrtIds = np.where(planMatrix[iKV])[0]
        if thrtIds.size > 0:
            idThrt = thrtIds[0]
            kvState = mods.PropagateECI(kvStates[iKV], 0.0, deltaT)
            threatState = mods.PropagateECI(threatStates[idThrt], 0.0, deltaT)
            tPOCA, _, _ = mods.FindPOCA(kvState, threatState)
            tNow = threatStates[idThrt, 6]
            tGo = tPOCA - (tNow + deltaT)
            if tGo.size > 1:
                print("something broke: tPOCA - {}, tNow - {}, deltaT - {}".format(tPOCA, tNow, deltaT))
            ptImpact = mods.PropagateECI(threatState, 0.0, tGo)
            if np.linalg.norm(kvStates[iKV, 0:3]) == 0 or np.linalg.norm(ptImpact[0:3]) == 0:
                kv = mods.PropagateECI(kvStates[iKV], 0.0, tPOCA - tNow)
            else:
                kv, _ = mods.GaussProblem(kvStates[iKV, 0:3], ptImpact[0:3], tPOCA - tNow)
            
            dVec = kv[3:6] - kvState[3:6]
            dV = np.linalg.norm(dVec)
            # see if we need to divert right now to make it. if so perform the divert
            mustDivert = ((kvFuel[iKV] - dV - mods.SAPs.KV_FUEL_RESERVE) <= 0)
            if mustDivert or tGo < 15.0:
                kvStates[iKV, 3:6] = kvStates[iKV, 3:6] + dVec
                kvFuel[iKV] = kvFuel[iKV] - dV
                finalAssign[iKV] = threatIds[idThrt]
    print("available threats: {}, \nassigned final threats: {}".format(threatIds, finalAssign.astype(int)))
    # return result
    return finalAssign.astype(int), kvStates

"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""
