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

def convertForSS(cvState, threatStates, threatIds, pLethal, scpl, tomIds):
    # TODO: how to handle matlab persistent first_time flag

    nThreats = threatStates.shape[0]
    print nThreats

