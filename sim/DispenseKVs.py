"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*   ModuleName: DispenseKVs                                                                                 *
*   Author(s):  M. A. Lambrecht, P. B. McCoy                                                                *
*   Version:    1.0                                                                                         *
*   Date:       01/02/18                                                                                    *
*                                                                                                           *
*       Module        This function performs and instantaneous dispense of the                              *
*       Description:  KVs.  Each KV is dispensed radially from the CV velocity                              *
*                     vector, at equidistant angular spacing.                                               *
*                                                                                                           *
*       Algorithm:    1) Generate a coordinate system with the x-axis being the CV                          *
*                        velocity vector (and the accompanying transformation matrix                        *
*                        between the new coordinate system and the ECI frame)                               *
*                     2) Compute the angles between the KVs on the CV body (evenly                          *
*                        dispersed)                                                                         *
*                     3) Generate the dispense offset and velocity in the new                               *
*                        coordinate system                                                                  *
*                     4) Transform the new offset and velocity back to ECI                                  *
*                     5) Add the new offset and velocity to the CV state to arrive at                       *
*                        a beginning state for each KV                                                      *
*                                                                                                           *
*       Inputs:       cvState - 7x1 CV state vector [x,y,z,Vx,Vy,Vz,t] (ECI)                                *
*                                                                                                           *
*       Outputs:      kvStates - Nx7 KV state vector [x,y,z,Vx,Vy,Vz,t] (ECI)                               *
*                                                                                                           *
*       Calls:        norm  (Matlab)                                                                        *
*                     cross (Matlab)                                                                        *
*                     sin   (Matlab)                                                                        *
*                     cos   (Matlab)                                                                        *
*                                                                                                           *
*       Requires:     SAPs_Null.json                                                                        *
*                                                                                                           *
*       Author:       M. A. Lambrecht                                                                       *
*                                                                                                           *
*       History:      MAL 10 Aug 2017:  Initial version                                                     *
*                                                                                                           *
*************************************************************************************************************
"""
import os
import numpy as np
import importModules as mods

def DispenseKVs(cvState):
    # number of KVs we are working with (assumed on the same ring)
    nKVs = mods.SAPs.KV_NUM_KVS

    # Build a coordinate system with the CV velocity as the x-axis (call it the G frame)
    xG = np.divide( cvState[3:6], np.linalg.norm(cvState[3:6]))
    zG = np.divide( np.cross(cvState[0:3], xG), 
        np.linalg.norm(np.cross(cvState[0:3], xG)))
    yG = np.divide( np.cross(zG, xG), np.linalg.norm(np.cross(zG, xG)))

    # find the direction cosine matrix from G->ECI
    t_G2ECI = np.hstack((xG.reshape(3,1), yG.reshape(3,1), zG.reshape(3,1)))

    # determine the KV angles around the CV body in order to determine the dispense direction
    theta = np.linspace(0, 2*np.pi*(nKVs-1)/nKVs, nKVs)

    # dispense the KVs
    dVDisp = mods.SAPs.KV_DISPENSE_DV
    aMax = mods.SAPs.KV_MAX_ACC
    r0 = mods.SAPs.CV_RADIUS
    kvStates = np.zeros((nKVs, 7))

    # position delta
    pGx = np.zeros((nKVs, 1))
    pGy = np.reshape( (r0+1*dVDisp)*np.cos(theta) + (0.5*aMax*(1**2)*np.cos(theta)), (nKVs, 1) )
    pGz = np.reshape( (r0+1*dVDisp)*np.sin(theta) + (0.5*aMax*(1**2)*np.sin(theta)), (nKVs, 1) )
    pG = np.hstack((pGx, pGy, pGz))

    # velocity delta
    vGx = np.zeros((nKVs,1))
    vGy = np.reshape( dVDisp*np.cos(theta) + aMax*1*np.cos(theta), (nKVs, 1) )
    vGz = np.reshape( dVDisp*np.sin(theta) + aMax*1*np.sin(theta), (nKVs, 1) )
    vG = np.hstack((vGx, vGy, vGz))

    for iKV in range(nKVs):
        # Transform back to ECI
        pECI = np.dot(t_G2ECI, pG[iKV])
        vECI = np.dot(t_G2ECI, vG[iKV])
        # Build the state for this KV
        kvStates[iKV, 6] = cvState[6]
        kvStates[iKV, 0:3] = cvState[0:3].flatten() + pECI
        kvStates[iKV, 3:6] = cvState[3:6].flatten() + vECI

    return kvStates

"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""
