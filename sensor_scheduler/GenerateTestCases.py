"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*   Module Name: GenerateTestCases                                                                          *
*   Author(s): Mark Lambrecht, Brent McCoy                                                                  *
*   Version: 1.0                                                                                            *
*   Date: 03/30/18                                                                                          *
*                                                                                                           *
*       Module          Generates a simulated engagement with a CV state and a set                          *
*       Description:    of track states.                                                                    *
*                                                                                                           *
*       Algorithm:      N/A                                                                                 *
*                                                                                                           *
*       Inputs:         N/A                                                                                 *
*                                                                                                           *
*       Outputs:        rCV0        -   1x3 initial CV position in ECI                                      *
*                       vCV0        -   1x3 initial CV velocity in ECI                                      *
*                       rTgt0       -   #Threatsx3 initial threat position in ECI                           *
*                       vTgt0       -   #Threatsx3 initial threat velocity in ECI                           *
*                       tFinal      -   final time for simulation (time of flight)                          *
*                       pLethal     -   #Threats vector of initial threat lethality potentials              *
*                       statSigPL   -   #Threats vector of statstical confidence in pLethal assessment      *
*                       snr         -   #Threats vector of some statistical measurement (default 30)        *
*                       covTgt      -   #Threatsx6x6 matrix of state covariance for the threats             *
*                       tLastVisit  -   #Threats vector measuring time since last visited                   *
*                                                                                                           *
*       Calls:          mods.Kepler (propagation routine)                                                   *
*                                                                                                           *
*       Requires:       N/A                                                                                 *
*                                                                                                           *
*       OA:             Mark Lambrecht                                                                      *
*                                                                                                           *
*       History:        MAL 22 Jan 2018:    Initial version                                                 *
*                                                                                                           *
*************************************************************************************************************
"""
import numpy as np
import importModules as mods

def GenerateTestCase(unpack=False):
    # create a random number of tracks
    numTracks = np.random.randint(15, 21)
    # generate the final states of the CV and threat PDM
    rCVFinal = np.array([-7, 0, 0]) * 1e3       # final CV position [ECI, km]
    rMisl = np.zeros(3) * 1e3                   # missile vector [ECI, km]
    # terminal velocity of the track (km/s)
    vTrackTerm = np.array([0, -4.5, -5])
    # terminal velocity of the CV (km/s)
    vCVTerm = np.array([0, 7.5, 0])
    # time of flight, track and CV
    tFinal = 5 * 60
    # propagate CV track to start point
    rCV0, vCV0 = mods.Kepler(rCVFinal, -vCVTerm, tFinal)
    vCV0 = -vCV0
    # propagate target track (approx centroid)
    rTgt0, vTgt0 = mods.Kepler( rCVFinal+rMisl, -vTrackTerm, tFinal)
    vTgt0 = -vTgt0

    rDeviation = np.linalg.norm(rTgt0) * 0.01
    vDeviation = np.linalg.norm(vTgt0) * 0.01

    rTracks = np.tile(rTgt0, (numTracks, 1)) + rDeviation * np.random.randn(3, numTracks)
    vTracks = np.tile(vTgt0, (numTracks, 1)) + vDeviation * np.random.randn(3, numTracks)

    # generate pLethal values
    pLethal = np.random.rand(numTracks) * 0.5
    pLethal[np.random.randint(pLethal.size)] = mods.UniformRandRange(0.7, 0.9) 
    pLethal[np.random.randint(pLethal.size)] = mods.UniformRandRange(0.7, 0.9) 
    pLethal[-1] = 0.001

    # emulate statistical significance of lethality assessment
    statSigPL = np.random.rand(numTracks)
    statSigPL[-1] = 0.999

    # whatever SNR is? (range value of some sort?)
    snr = np.random.rand(numTracks) * 30
    snr[-1] = 30

    # target track covariance
    upperLeft = np.diag(np.ones(3)*600**2)
    lowerRight = np.diag(np.ones(3)*1.2**2)
    upperCov = np.hstack((upperLeft, np.zeros_like(upperLeft)))
    lowerCov = np.hstack((np.zeros_like(lowerRight), lowerRight))
    covBase = np.vstack((upperCov, lowerCov))

    covTgt = np.tile(covBase, (numTracks, 1, 1))

    # create the timeSinceLastVisit vector
    tLastVisit = np.ones(numTracks)
    tLastVisit[np.random.randint(numTracks)] = 35
    
    # build lethalityObject
    lethalObject = mods.lethalityMatrix(pLethal, statSigPL, snr)

    # build simState object
    buildDict = {"rCV": rCV0, "vCV": vCV0, "rTracks": rTracks, "vTracks": vTracks, 
            "lethalityObject": lethalObject, "timeArray": timeArray}

    simState = mods.stateCollection(*buildDict)

    # release results to the simulation
    if unpack:
        return rCV0, vCV0, rTgt0, vTgt0, tFinal, pLethal, statSigPL, snr, covTgt, tLastVisit
    else:
        return simState

"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""
