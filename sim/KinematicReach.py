"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*   Module Name: KinematicReach                                                                             *
*   Author(s): Mark Lambrecht, Brent McCoy                                                                  *
*   Version: 1.0                                                                                            *
*   Date: 01/04/18                                                                                          *
*                                                                                                           *
*       Module        This function builds the kinematic reach matrix that specifies                        *
*       Description:  kinematic feasiblity of engagements between each pair of KVs and                      *
*                     theats.                                                                               *
*                                                                                                           *
*       Algorithm:    1) Build the CV coordinate system                                                     *
*                     2) Transform the KVs and the threats (at POCA) into the CV                            *
*                        coordinate system                                                                  *
*                     3) Determine the required divert for each pair                                        *
*                     4) If the required divert is < available divert, the engagement                       *
*                        is kinematically feasible                                                          *
*                                                                                                           *
*       Inputs:       cvState      - 7x1 CV state (t,x,y,z,vx,vy,vz), [s,m,m/s]                             *
*                     cvFuel       - 1x1 available delta-V on the CV [m/s]                                  *
*                     kvStates     - #KVsx7 KV states (t,x,y,z,vx,vy,vz), [s,m,m/s]                         *
*                     kvFuel       - #KVsx1 available delta-V on the KVs [m/s]                              *
*                     threatStates - #Threatsx7 threat states (t,x,y,z,vx,vy,vz),                           *
*                                    [s,m,m/s]                                                              *
*                     tPOCA        - (Scalar) Estimated time of intercept, s                                *
*                                                                                                           *
*       Outputs:      krMatrix     - #KVs x #Threats binary matrix (1 = feasible, 0 =                       *
*                                    infeasible)                                                            *
*                     dvMatrix     - #KVs x #Threats divert required matrix                                 *
*                                                                                                           *
*       Calls:        Util_FindPOCA                                                                         *
*                     Util_PropagateStates                                                                  *
*                     RequiredDivert (Local)                                                                *
*                                                                                                           *
*       Requires:     SAPs_Null.json                                                                        *
*                                                                                                           *
*       OA:           M. A. Lambrecht                                                                       *
*                                                                                                           *
*       History:      MAL 18 Dec 2017:  Initial version                                                     *
*                                                                                                           *
*************************************************************************************************************
"""
import os
import numpy as np
import importModules as mods
#from generalUtilities.Classes import jsonData
#from generalUtilities.FindPOCA import FindPOCA
#from generalUtilities.propagate import PropagateECI
#from generalUtilities.config import SAPs

def RequiredDivert(t0, tT, p0, v0, pT, aMax):
    ## read max acceleration from SAPsObj
    #aMax = SAPsObj.KV_MAX_ACC

    # Compute the time-to-go
    dT = tT - t0

    ## Adjust the pT to end at point KV would have travelled
    #pT = pT + p0 + v0*dT

    # Compute the required burn time
    tBurn = dT - np.sqrt( dT**2 - (2/aMax) * np.linalg.norm(pT - p0 - v0*dT) )

    # Compute the required delta-V
    dV = aMax * tBurn

    #a = pT - (p0 + v0*dT)
    #direction = a / np.linalg.norm(a)
    #dVvec = tCV2ECI * (direction * dV)

    ## Check to see if it actually hit
    #a = aMax * direction
    #p = p0 + v0*tBurn + 0.5*a*tBurn**2
    #v = v0 + a*tBurn
    #dT = (tT - t0) - tBurn
    #p = p + v*dT

    return dV, tBurn

def KinematicReach(cvState, kvStates, kvFuel, threatStates):
    # Numbers
    nKVs = kvStates.shape[0]
    nThreats = threatStates.shape[0]
    if nKVs != nThreats:
        print("number of KVs: {}".format(nKVs))
        print("number of threats: {}".format(nThreats))

    # Find the POCA between the CV and the center of the threat complex
    aveState = np.mean(threatStates, axis=0)
    tPOCA, rPOCA, _ = mods.FindPOCA(cvState, aveState)

    # Propagate all threats to tPOCA
    tGo = tPOCA - threatStates[0, 6]
    threatStatesPOCA = np.zeros(threatStates.shape)
    for iThrt in range(nThreats):
        threatStatesPOCA[iThrt] = mods.PropagateECI(threatStates[iThrt].reshape(7,1), 0.0, tGo)

    # Build the CV coordinate system to work in (and corresponding transformations)
    rCV_eci = cvState[0:3]
    vCV_eci = cvState[3:6]

    # c1 = vCV_eci'
    c1 = (rPOCA - rCV_eci) / np.linalg.norm( (rPOCA - rCV_eci) )
    c2 = np.cross(np.array([0, 0, 1]), c1) / np.linalg.norm( np.cross(np.array([0, 0, 1]), c1) )
    c3 = np.cross(c1, c2) / np.linalg.norm( np.cross(c1, c2) )

    tCV2ECI = np.hstack( (c1.reshape(3,1), c2.reshape(3,1), c3.reshape(3,1)) )
    #tECI2CV = np.transpose( tCV2ECI )  rotR*vec == vec*rotL  so I have made some modifications below
    
    # Put the KVs and threats in the CV reference frame
    rKV_cv = np.zeros((nKVs, 3))
    vKV_cv = np.zeros((nKVs, 3))
    for iKV in range(nKVs):
        rKV_cv[iKV] = np.dot( kvStates[iKV, 0:3] - rCV_eci, tCV2ECI)    # same as (tECI2CV*kvStates')'
        vKV_cv[iKV] = np.dot( kvStates[iKV, 3:6] - vCV_eci, tCV2ECI)    # same as (tECI2CV*kvStates')'

    rTht_cv = np.zeros((nThreats, 3))
    for iThrt in range(nThreats):
        rTht_cv[iThrt] = np.dot( threatStatesPOCA[iThrt, 0:3] - rPOCA, tCV2ECI )

    # Determine the Kinematic feasibility of each KV/Threat pair
    krMatrix = np.zeros((nKVs, nThreats))
    dvMatrix = np.zeros((nKVs, nThreats))
    for iKV in range(nKVs):
        for iThrt in range(nThreats):
            # New Method
            dV, tBurn = RequiredDivert( 
                    kvStates[iKV, 6],
                    tPOCA,
                    rKV_cv[iKV],
                    vKV_cv[iKV],
                    rTht_cv[iKV],
                    mods.SAPs.KV_MAX_ACC)
            if tBurn < tGo and (kvFuel[iKV] - dV - mods.SAPs.KV_FUEL_RESERVE) > 0:
                krMatrix[iKV, iThrt] = 1
                dvMatrix[iKV, iThrt] = dV

    return krMatrix, dvMatrix

"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""
