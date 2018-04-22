"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*   Module Name: FindPOCA                                                                                   *
*   Author(s): Brent McCoy and Mark Lambrecht                                                               *
*   Version: 1.0                                                                                            *
*   Date: 12/20/17                                                                                          *
*                                                                                                           *
*
*       Module        This function finds the point of closest approach (POCA)
*       Description:  between two trajectories.  It assumes that the input states
*                     have the same time of validity.  It also assumes that the
*                     trajectories intersect within 750 seconds of that time of
*                     validity.
*
*       Algorithm:    1) Propagate both states 750 seconds in the future (coarse
*                        resolution)
*                     2) Find the POCA at coarse resolution
*                     3) Interpolate both trajectories to fine resolution around
*                        the coarse POCA
*                     4) Find the POCA at the fine resolution
*
*       Inputs:       kaState   - 7x1 state vector of Kill Asset trajectory
*                                 [x,y,z,Vx,Vy,Vz,t] (ECI)
*                     trgtState - 7x1 state vector of Target trajectory
*                                 [x,y,z,Vx,Vy,Vz,t] (ECI)
*
*       Outputs:      tPOCA  - (Scalar) Time of POCA [s]
*                     rPOCA  - Position of POCA (3x1) [m]
*                     dRPOCA - (Scalar) Distance of kaState from POCA at tPOCA, m
*
*       Calls:        Util_PropagateStatesToDivergence
*                     size  (Matlab)
*                     Util_InterpTraj
*                     min  (Matlab)
*                     sqrt (Matlab)
*
*       OA:           M. A. Lambrecht
*
*       History:      MAL 10 Aug 2017:  Initial version
*                                                                                                           *
*************************************************************************************************************
"""
import numpy as np
from generalUtilities.propagate import PropagateStatesToDivergence
from generalUtilities.utilities import vecNorm, InterpTraj

def FindPOCA(kaState, trgtState):
    #print("executing FindPOCA")
    #print("Propagate states at coarse resolution (5s)")
    # Propagate states at coarse resolution (5s)
    kaTraj, trgtTraj = PropagateStatesToDivergence(kaState, trgtState, 2.5)

    #print("interpolate states around the coarse resolution POCA - coarse resolution")
    # interpolate states around the coarse resolution POCA - coarse resolution
    # POCA is the 2nd to las trajectory point of the target
    rows = kaTraj.shape[0]
    tToInterp = np.arange(kaTraj[rows-4, 6], kaTraj[rows-1, 6], 0.01)
    interpKA = InterpTraj(kaTraj, tToInterp)
    interpTrgt = InterpTraj(trgtTraj, tToInterp)

    #print("find the fine resolution POCA (0.01s)")
    # find the fine resolution POCA (0.01s)
    rKA = interpKA[:, 0:3]
    rTrgt = interpTrgt[:, 0:3]
    rDiff = rKA - rTrgt
    rMag = vecNorm(rDiff, axis=1)
    
    #print("finding the maximum result, arbitrarily using the first hit in case there are multiple")
    # finding the maximum result, arbitrarily using the first hit in case there are multiple
    dRPOCA = rMag.min()
    ind = rMag.argmin()

    #print("return the values")
    # return the values
    tPOCA = interpTrgt[ind, 6]
    rPOCA = interpTrgt[ind, 0:3]

    return tPOCA, rPOCA, dRPOCA

"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""
