"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*   Module Name: ConvertForSS                                                                               *
*   Author(s): Mark Lambrecht, Keith Graves                                                                 *
*   Version: 1.0                                                                                            *
*   Date: 04/22/18                                                                                          *
*                                                                                                           *
*       Description:  Convert data to the sensor scheduler coordinate system (ECI).
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
*       Calls:        convertForSS (local)
*                                                                                                           *
*       OA:             M. A. Lambrecht                                                                     *
*                                                                                                           *
*       History:        MAL 22 Jan 2018:  Initial version                                                   *
*                                                                                                           *
*************************************************************************************************************
"""
import numpy as np

first_time = True

def convertForSS(cvState, threatStates, threatIds, pLethal, scpl, tomIds):
    # TODO: how to handle matlab persistent first_time flag

    nThreats = threatStates.shape[0]
    tracks = [None] * nThreats
    tomTracks = [None] * len(tomIds)
    cv = None
    simStateAll = {'tracks': tracks, 'cvState': cvState}
    simState = {'tomTracks': tomTracks}
    jj = 0;
    print(nThreats)
    print(threatIds)
    print(tomIds)
    for ii in range(nThreats):
        track = {'rCurr': np.divide(threatStates[ii][0:3], 1e3), 'vCurr': np.divide(threatStates[ii][3:6], 1e3)}
        track['id'] = threatIds[ii]
        track['active'] = True
        track['pLethal'] = pLethal[ii]
        track['snr'] = 10.0
        track['scpl'] = scpl[ii]
        simStateAll['tracks'][ii] = track
        # ind = np.where(tomIds == threatIds[ii])
        ind = list(tomIds).index(threatIds[ii])
        if ind is not None:
            simState['tomTracks'][jj] = simStateAll['tracks'][ii]
            jj += 1

    nTracks = jj - 1

    # update the cv
    cvTrack = {'rCurr': np.divide(cvState[0:3], 1e3), 'vCurr': np.divide(cvState[3:6], 1e3)}
    simStateAll['cvState'] = cvTrack
    simStateAll['cvState'] = cvTrack

    print(simStateAll)
    print (simState)

    global first_time
    if first_time:
        first_time = False
        simStateAll['cvState']['CVup'] = simStateAll['cvState']['rCurr']
        print("KLG", simState['tomTracks'][:nTracks])
        # cvPointing = np.mean(simState['tomTracks']['rCurr']) - simStateAll['cvState']['rCurr']
        # simStateAll['cvState']['CVpointing'] = cvPointing / np.norm(cvPointing)


if __name__ == '__main__':
    cvState = np.random.rand(7)
    threatStates = np.random.rand(12, 7)
    threatIds = np.arange(1, threatStates.shape[0]+1)
    pLethal = np.random.rand(12)
    scpl = np.random.rand(12)
    tomIds = np.arange(1, threatStates.shape[0]+1)
    convertForSS(cvState, threatStates, threatIds, pLethal, scpl, tomIds)
    convertForSS(cvState, threatStates, threatIds, pLethal, scpl, tomIds)
