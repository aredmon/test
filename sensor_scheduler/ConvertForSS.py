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


def convertForSS(cv_state, threat_states, threat_ids, p_lethal, scpl, tom_ids):
    n_threats = threat_states.shape[0]
    tracks = [None] * n_threats
    tom_tracks = [None] * len(tom_ids)
    sim_state_all = {'tracks': tracks, 'cvState': cv_state}
    sim_state = {'tomTracks': tom_tracks}
    jj = 0;
    for ii in range(n_threats):
        track = {
            'rCurr': np.divide(threat_states[ii][0:3], 1e3),
            'vCurr': np.divide(threat_states[ii][3:6], 1e3),
            'id': threat_ids[ii],
            'active': True,
            'pLethal': p_lethal[ii],
            'snr': 10.0,
            'scpl': scpl[ii]
        }
        sim_state_all['tracks'][ii] = track
        try:
            list(tom_ids).index(threat_ids[ii])
            sim_state['tomTracks'][jj] = sim_state_all['tracks'][ii]
            jj += 1
        except ValueError:
            pass

    n_tracks = jj - 1

    # update the cv
    cv_track = {
        'rCurr': np.divide(cv_state[0:3], 1e3),
        'vCurr': np.divide(cv_state[3:6], 1e3)
    }
    sim_state_all['cvState'] = cv_track
    sim_state['cvState'] = cv_track

    global first_time
    if first_time:
        positions = []
        sim_state_all['cvState']['CVup'] = sim_state_all['cvState']['rCurr']
        for position in sim_state['tomTracks']:
            positions.append(position['rCurr'])
        cv_pointing = np.mean(positions, axis=0) - sim_state_all['cvState']['rCurr']
        sim_state_all['cvState']['CVpointing'] = cv_pointing / np.linalg.norm(cv_pointing)
        sim_state['cvState'] = sim_state_all['cvState']
        first_time = False

    print(sim_state_all)
    print (sim_state)


if __name__ == '__main__':
    cvState = np.random.rand(7)
    threatStates = np.random.rand(12, 7)
    threatIds = np.arange(1, threatStates.shape[0]+1)
    pLethal = np.random.rand(12)
    scpl = np.random.rand(12)
    tomIds = np.arange(1, threatStates.shape[0])
    convertForSS(cvState, threatStates, threatIds, pLethal, scpl, tomIds)
