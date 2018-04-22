"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*   Module Name: CVPointer                                                                                  *
*   Author(s): Mark Lambrecht, Brent McCoy                                                                  *
*   Version: 1.0                                                                                            *
*   Date: 04/19/18                                                                                          *
*                                                                                                           *
*       Description:  Exercises the sensor scheduler. Returns the observed objects
*
*       Algorithm:    Convert data to the sensor scheduler coordinate system (ECI).
*                     Execute the sensor scheduler.
*
*       Inputs:       cvState      - CV 6-state, r, v, t, ECEF, 7x1, m, m/s, s
*                     threatStates - threat 6-states, r, v, t, ECEF,
*                                    #Tracks (total) x 7, m, m/s, s
*                     threatIds    - 1 x # Tracks (total) threat IDs
*                     pLethal      - #Tracks (total) x 1 P(lethal)
*                     scpl         - 1 x # Tracks (total) statistical confidence of
*                                    P(lethal) values
*                     tomIds       - # tracks (in TOM) x 1 IDs of tracks in TOM
*                     features     - 1 x # Tracks (total) features (length, diameter,
*                                    temperature, emissivity, etc.)
*                     timeStep     - time step over which to exercise the SS, s
*                     tCurr        - current time, seconds
*
*       Outputs:      newTrackIds - list of newly observed track IDs
*
*       Calls:        ecef2eci     (local)
*                     convertForSS (local)
*                     exec_ss
*                                                                                                           *
*       OA:             M. A. Lambrecht                                                                     *
*                                                                                                           *
*       History:        MAL 22 Jan 2018:  Initial version                                                   *
*                                                                                                           *
*************************************************************************************************************
"""
import numpy as np
import importModules as mods

def run_ss(cvState, threatStates, ids, pLethalGround, scpl, tomIds, features, tStep, t):
    newTrackIds = {};
    return newTrackIds

def exec_ss(relevantSimStates, completeSimStates, ):

"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""
