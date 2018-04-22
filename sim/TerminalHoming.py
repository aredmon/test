"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*   Module Name: utilities.py                                                                               *
*   Author(s): Brent McCoy                                                                                  *
*   Version: 1.0                                                                                            *
*   Date: 02/23/18                                                                                          *
*                                                                                                           *
*       Module Description:     Fly KVs to their PIP                                                        *
*                                                                                                           *
*           Inputs:             assignedThreats   - threatIds that KVs have been assigned to                *
*                               kvStates          - current 7 sate of all onboard KVs                       *
*                               threatStates      - current 7 states of identified threats                  *
*                                                                                                           *
*           Outputs:            kvStates          - new 7 state created to guide KV to target               *
*                                                                                                           *
*           Calls:              mods.FindPOCA                                                               *
*                               mods.GaussProblem                                                           *
*                                                                                                           *
*           OA:                 Mark Lambrecht                                                              *
*                                                                                                           *
*           History:            MAL, 24 Jan 2018 - Initial Version                                          *
*                                                                                                           *
*************************************************************************************************************
"""
import numpy as np
import importModules as mods

def TerminalHoming(assignedThreats, kvStates, threatStates):
    nKVs = assignedThreats.size
    validAssignments = np.asarray(assignedThreats[ (assignedThreats >= 0) ], dtype=int)
    print("nKVs: {} \nassignedThreats: {} \nvalidAssignments: {}".format(nKVs, 
        assignedThreats, validAssignments))
    for index, threatId in enumerate(validAssignments):
        threatState = threatStates[ threatId ]
        kvState = kvStates[ index ]
        tNow = kvState[6]
        tPOCA, rPOCA, _ = mods.FindPOCA(kvState, threatState)
        tGo = tPOCA - tNow
        kv, _ = mods.GaussProblem(kvState[0:3], rPOCA, tGo)
        dV = kv[3:6] - kvState[3:6]
        kvStates[index, 3:6] = kvState[3:6] + dV
    # return final kvStates guidance solutions
    return kvStates

"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""
