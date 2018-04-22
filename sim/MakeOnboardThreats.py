"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*   Module Name: MakeOnboardThreats                                                                         *
*   Author(s): Mark Lambrecht, Brent McCoy                                                                  *
*   Version: 1.0                                                                                            *
*   Date: 01/08/18                                                                                          *
*                                                                                                           *
*       Inputs:       threatStates - Nx7 array of threat states [x y z dx dy dz t],                         *
*                                    ECI (truth)                                                            *
*                     rvID         - (Scalar) index of RV in threatStates                                   *
*                     pLethal      - Nx1 P(RV) of each track                                                *
*                     fracObjsSeen - (Scalar) fraction of objects "seen" by CV/KVs                          *
*                                                                                                           *
*       Outputs:      tracks   - Mx9 array of threat states and lethality                                   *
*                                [x y z dx dy dz t P(RV) ID]                                                *
*                     trackCov - Mx6x6 position covariance of track states                                  *
*                     trackIds - Mx1 array of truth IDs in the TOM                                          *
*                                                                                                           *
*       Calls:        Util_UniformRandRange                                                                 *
*                     Util_PropagateECI                                                                     *
*                                                                                                           *
*       OA:           M. A. Lambrecht                                                                       *
*                                                                                                           *
*       History:      MAL 19 Dec 2017:  Initial version                                                     *
*                                                                                                           *
*************************************************************************************************************
"""
import os
import numpy as np
import importModules as mods

def MakeOnboardThreats(threatStates, rvID, pLethal, fracObjsSeen):
    # pick which threats are seen by the radar
    nThreats = threatStates.shape[0]
    nThreatsSeen = int( np.round( fracObjsSeen * nThreats ) )
    randIds = np.random.permutation( np.arange(nThreats) )
    trackIds = randIds[0:nThreatsSeen]

    # make sure the RV is in the TOM
    if not trackIds.__contains__(rvID):
        indx = np.random.randint(trackIds.shape[0])
        trackIds[indx] = rvID

    # now build the TOM
    tracks = np.zeros((nThreatsSeen, 9))
    trackCov = np.zeros((nThreatsSeen, 6, 6))
    biasR = np.random.randn(3) * ( mods.SAPs.CV_POS_BIAS / np.sqrt(3.0) )
    biasV = np.random.randn(3) * ( mods.SAPs.CV_VEL_BIAS / np.sqrt(3.0) )
    t = threatStates[0, 6]
    indx = 0
    for iThrt in trackIds:
        sigR = np.random.rand(1,3) * ( mods.SAPs.CV_POS_NOISE / np.sqrt(3.0) )
        sigV = np.random.rand(1,3) * ( mods.SAPs.CV_VEL_NOISE / np.sqrt(3.0) )
        r = sigR + biasR + threatStates[iThrt, 0:3]
        v = sigV + biasV + threatStates[iThrt, 3:6]
        tracks[indx, 0:6] = np.append(r, v)
        tracks[indx, 6] = t
        tracks[indx, 7] = pLethal[ iThrt ]
        tracks[indx, 8] = int( iThrt )

        covariance = np.append((sigR + biasR)**2, (sigV + biasV)**2)
        #print("calculated covariance shape: {}".format(covariance.shape))
        trackCov[indx] = np.diag(covariance)
        #print("trackCovariace: {}".format(trackCov[indx]))
        indx += 1

    # pass out tom information
    return tracks, trackCov, trackIds

"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""
