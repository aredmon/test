"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*   Module Name: CVPointer                                                                                  *
*   Author(s): Mark Lambrecht, Brent McCoy                                                                  *
*   Version: 1.0                                                                                            *
*   Date: 04/05/18                                                                                          *
*                                                                                                           *
*       Module          Schedule the scensor given 5-states of the observer and 3-states                    *
*       Description:    of the targets of interest.                                                         *
*                                                                                                           *
*       Algorithm:      N/A                                                                                 *
*                                                                                                           *
*       Inputs:         cvState.rCurr - current observer position, ECI, km                                  *
*                       cvState.vCurr - current observer velocity, ECI, km/s                                *
*                       cvState.cvPointing - current pointing vector                                        *
*                       cvState.cvup - up/orientation vector for the CV                                     *
*                                                                                                           *
*                       tracks(i).rCurr - current track position(s), ECI, km                                *
*                       tracks(i).vCurr - current track velocity(ies), ECI, km/s                            *
*                                                                                                           *
*                       parameterDict - observer characteristics (stored in python dictionary)              *
*                                       params.DCV_ACC_LIM - angular acceleration limit, radians/sec^2      *
*                                       params.DCV_VEL_LIM - angular rate limit, radians/sec                *
*                                       params.W           - FOV (square), radians                          *
*                                       params.tdwell      - dwell time in seconds                          *
*                                       params.upMode      - up/orientation vector:                         *
*                                                          '+x' for positive x (static)                     *
*                                                          '+y' for positive y (static)                     *
*                                                          '+z' for positive z (static)                     *
*                                                          'local' for radial (changes depending on ECI     *
*                                                          position)                                        *
*                                       params.dither      - true to enable dither (default)                *
*                                       params.nDither     - dither 'refinement' parameter                  *
*                                                                                                           *
*       Outputs:        cvPointingOut - pointing command      (passed back as an update to the cvStateObj)  *
*                       cvUp          - orientation vector    (passed back as an update to the cvStateObj)  *
*                       slewTime      - slew time in seconds  (added to parameter dictionary)               *
*                       dwellTime     - dwell time in seconds (same as params.tdwell)                       *
*                                                                                                           *
*       Calls:          mods.ECI2Observer                                                                   *
*                       mods.Observer2AzEl                                                                  *
*                       mods.InitialGuess                                                                   *
*                       mods.Dither                                                                         *
*                       mods.CheckMarked                                                                    *
*                       mods.AzEl2Attitude                                                                  *
*                                                                                                           *
*       OA:             M. A. Lambrecht                                                                     *
*                                                                                                           *
*       History:        MAL 22 Jan 2018:  Initial version                                                   *
*                                                                                                           *
*************************************************************************************************************
"""
import numpy as np
import importModules as mods
"""
----------------------------------------------------------------------------------------------
    Inputs:       rCVfinal  - 1x2 target az/el, radians
                  rTracks   - #Tracksx2 az/el, radians
                  vTracks   - #tracksx2 azDot/elDot, radians/sec
                  rCVNow    - 1x2 cv pointing angles (az/el)
                  params    - CV (and other) parameters
                  getVecs   - boolean flag to select if you want the routine to output the 
                             reduced vectors instead of the boolean array of good points
                             default option is FALSE
    
    Outputs:      iMarked   - #Tracksx1 array of marked tracks (1 = seen, 0 = not)
                  rMarked   -
                  vMarked   -
----------------------------------------------------------------------------------------------
"""
def CheckMarked(rCVFinal, rTracks, vTracks, rCVNow, parameterDict, getVecs=False):
    # extract necessary info from the parameterDict
    accelLimit = parameterDict['DCV_ACC_LIM']
    velLimit = parameterDict['DCV_VEL_LIM']
    tolerance = parameterDict['FOV']
    tDwell = parameterDict['tDwell']

    tFinal = mods.Command(rCVNow, rCVFinal, accelLimit, velLimit)
    rNext = rTracks + vTracks * tFinal
    rNextDwell = rTracks + vTracks * (tFinal + tDwell)

    rNextCheck = np.array([np.allclose(rNextRow, rCVFinal, tolerance/2) for rNextRow in rNext])
    rDwellCheck = np.array([np.allclose(rDwellRow, rCVFinal, tolerance/2) for rDwellRow in rNextDwell])

    iMarked = np.logical_and(rNextCheck, rDwellCheck)

    #release results
    if getVecs:
        return iMarked, rTracks[~iMarked], vTracks[~iMarked]
    else:
        return iMarked

"""
-------------------------------------------------------------------------------------------------------------
    DEFAULT PointMe ROUTINE (runs once and provides an answer, does not keep track of information)
-------------------------------------------------------------------------------------------------------------
"""
def PointMe(cvStateObj, trackStateObj, parameterDict, record={}):
    # convert parameter dictionary into an object with fields:
    params = mods.SimpleNamespace(parameterDict)
    # extract the CVstate information from the cvStateObj
    RCV = cvStateObj.rCurr
    VCV = cvStateObj.vCurr
    cvPntg = cvStateObj.CVpointing
    cvUp = cvStateObj.CVup
    # extract the track state information from the trackStateObj
    rTracks = trackStateObj.rCurr
    vTracks = trackStateObj.vCurr

    # convert ECI states to CV frame
    dRangle, dVangle = mods.ECI2Observer(RCV, VCV, cvPntg, cvUp, rTracks, vTracks)

    # convert CV to track vectors in CV fram to az/el
    posAngle, velAngle = mods.Observer2AzEl(dRangle, dVangle)

    # initial guess setup:
    commGuess0, tIntercept, _, anglePrime, angleDotPrime = mods.InitialGuess(posAngle, velAngle, 
            np.zeros(2), parameterDict)

    # refine initial guess using dither routine
    if not 'dither' in parameterDict or parameterDict['dither']:
        pL = tracksStateObj.pLethal
        commDithered, tIntercept = mods.Dither(commGuess0, anglePrime, angleDotPrime, pL, 
                np.zeros(2), parameterDict)
        targetAngle = commDithered
    else:
        targetAngle = commGuess0

    # determine which tracks were just visited
    iMarked = CheckMarked(targetAngle, anglePrime, angleDotPrime, 
            np.zeros(2), parameterDict)

    indNotLethal = np.logical_or(iMarked, [trackStateObj.pLethal < params.lethalThreshold])
    record.update({'trackIdsVisitedThisTime': trackStateObj.Ids[indNotLethal})

    # convert az/el command back to pointing and up vector
    cvStateObj.CVpointing, cvStateObj.CVup = mods.AzEl2Attitude(RCV, targetAngle, cvPntg, cvUp)
    parameterDict.update("slewTime": tIntercept)

    # pass results back to the sim
    return cvStateObj, trackStateObj, parameterDict, record

"""
----------------------------------------------------------------------------------------------
Module        Computes a desired pointing/orientation pair given the CV and
Description:  target states, and the various CV sensor/kinematic capabilities.

    
    Inputs:       cvState.rCurr     - current CV position, ECI, m
                  cvState.vCurr     - current CV position, ECI, m
                  cvPointing        - current CV pointing vector, ECI
                  tracks.rCurr      - current track position(s), ECI, m
                  tracks.vCurr      - current track velocity(ies), ECI, m/s
                  tracks.id         - track identification number (must be unique)
                  tracks.active     - track active (boolean).  track will be ignored
                                      completely if this is false.

                  parameterDict         - observer characteristics (stored in python dictionary)
                  params.DCV_ACC_LIM    - angular acceleration limit, radians/sec^2
                  params.DCV_VEL_LIM    - angular rate limit, radians/sec
                  params.FOV            - FOV (square), radians
                  params.tdwell         - dwell time in seconds
                  params.upMode         - up/orientation vector:
                                            '+x' for positive x (static)
                                            '+y' for positive y (static)
                                            '+z' for positive z (static)
                                            'local' for radial (changes depending on ECI position)
                                            a 3x1 double array to manually specify the up vector

                  par.dither        - true to enable dither (default)
                  par.nDither       - dither 'refinement' parameter
                  par.verbose       - enable detailed output
                  par.graphics      - enable graphical output
    
    Outputs:      cvPointingCommand - commanded / desired ECI pointing vector
                  cvUpCommand       - commanded / desired ECI orientation vector
                                      to indicate up direction of sensor
                  slewTime          - slew time in seconds
                  dwelling          - flag to indicate that the scheduler is now
                                      commanding the CV to remain in attitude hold
                                      so that the sensor can dwell / make
                                      observations.
----------------------------------------------------------------------------------------------
"""
def PointMeTS(cvStateObj, trackStateObj, parameterDict, record={}):
    # constants 
    FOV = parameterDict['FOV']

    # extracr the CV state information from the cvState object
    RCV = cvState.rCurr
    VCV = cvState.vCurr

    # extract track state information from input structure
    rTracks = trackStateObj.rCurr
    vTracks = trackStateObj.vCurr
    nTracks = rTracks.shape[0]

    # convert ECI states to CV frame
    dRangle, dVangle = mods.ECI2Observer(RCV, VCV, cvPointing, cvUp, rTracks, vTracks)

    # convert the CV - track vectors in the CV frame to az/el
    posAngle, velAngle = mods.Observer2AzEl(dRangle, dVangle)
    trackAngles = posAngle
    trackAnglesDot = velAngle

    # cluster the available tracks based on the size of the FOV
    points = posAngle
    clusterList = mods.QTCluster(points, FOV/2.0)
    
    # determine the center point of each cluster
    ptsdVel = velAngle
    centerPos, centerVel = mods.CenterOfClusters(clusterList, points, ptsVel) 
    cities = np.vstack(( np.zeros(2), centerPos ))

    # distance matrix = euclidean for 1st attempt.
    distMat = mods.twoPointDistance(cities, 'manhattan')

    # traveling salesman problem
    userConfig = {'xy': cities, 'distMat': distMat, 'nSalesmen': 1}
    result = mods.mtspofs_ga(userConfig)
    
    # determine the pointing angle
    nPts = result['optRoute'].size
    
    availableCenter = np.delete(centerPos, 0, axis=0)
    availableVel = np.delete(centerVel, 0, axis=0)
    newAngle = availableCenter[ result['optRoute'] ]
    newAngleDot = availableVel[ result['optRoute'] ]

    commandInitGuess, tIntercept, _, _, _ = mods.InitialGuess(newAngle, newAngleDot)

    targetAngle = commandInitGuess
    # determine which tracks were just visited
    iMarked = CheckMarked(targetAngle, trackAngles, trackAnglesDot, np.zeros(2), parameterDict)

    indNotLethal = np.logical_or(iMarked, [trackStateObj.pLethal < params.lethalThreshold])
    record.update({'trackIdsVisitedThisTime': trackStateObj.Ids[indNotLethal]})

    # convert the az/el command back to pointing and up vector
    cvPointingOut, cvUpVector = mods.AzEl2Attitude(RCV, targetAngle, record['cvPointingCommand'], 
            record['cvUpCommand'])

    record.update({'lastSlewTime': tIntercept})
    record.update({'dwellTime': parameterDict['tDwell']})
    record.update({'cvPointingCommand': cvPointingOut, 'cvUpCommand': cvUpVector})
    
    #release results
    return record
    
"""
----------------------------------------------------------------------------------------------
Module        Computes a desired pointing/orientation pair given the CV and
Description:  target states, and the various CV sensor/kinematic capabilities.

    
    Inputs:       cvState.rCurr     - current CV position, ECI, m
                  cvState.vCurr     - current CV position, ECI, m
                  cvPointing        - current CV pointing vector, ECI
                  tracks.rCurr      - current track position(s), ECI, m
                  tracks.vCurr      - current track velocity(ies), ECI, m/s
                  tracks.id         - track identification number (must be unique)
                  tracks.active     - track active (boolean).  track will be ignored
                                      completely if this is false.

    **NEW**       time              - current time, s

                  parameterDict         - observer characteristics (stored in python dictionary)
                  params.DCV_ACC_LIM    - angular acceleration limit, radians/sec^2
                  params.DCV_VEL_LIM    - angular rate limit, radians/sec
                  params.FOV            - FOV (square), radians
                  params.tdwell         - dwell time in seconds
                  params.upMode         - up/orientation vector:
                                            '+x' for positive x (static)
                                            '+y' for positive y (static)
                                            '+z' for positive z (static)
                                            'local' for radial (changes depending on ECI position)
                                            a 3x1 double array to manually specify the up vector

                  par.dither        - true to enable dither (default)
                  par.nDither       - dither 'refinement' parameter
                  par.verbose       - enable detailed output
                  par.graphics      - enable graphical output
    
    Outputs:      cvPointingCommand - commanded / desired ECI pointing vector
                  cvUpCommand       - commanded / desired ECI orientation vector
                                      to indicate up direction of sensor
                  dwelling          - flag to indicate that the scheduler is now
                                      commanding the CV to remain in attitude hold
                                      so that the sensor can dwell / make
                                      observations.
----------------------------------------------------------------------------------------------
"""
def PointMeManager(cvStateObj, trackStateObj, parameterDict, time, record={}):
    # tracks that have been visited are kept track of via the record in order to minimize
    # slow-down do to system read/write
    tracksUpdateIds = []        # tracks that are in the FOV
    visitedTrackIds = []        # tracks that were visited

    # get orientation vector
    if not 'upMode' in parameterDict:
        parameterDict.update({'upMode': '+z'})

    if 'verbose' in parameterDict:
        verbose = parameterDict['verbose']
    else:
        verbose = False
        
    # get the orientation vector
    if parameterDict['upMode'] == '+x':
        cvUpCommand = np.array([1, 0, 0])
    #
    elif parameterDict['upMode'] == '+y':
        cvUpCommand = np.array([0, 1, 0])
    #
    elif parameterDict['upMode'] == '+z':
        cvUpCommand = np.array([0, 0, 1])
    #
    elif parameterDict['upMode'] == 'local':
        cvUpCommand = cvStateObj.rCurr / np.linalg.norm(cvStateObj.rCurr)
    #
    else:
        cvUpCommand = np.asarray(parameterDict['upMode'])
    # add the cvUpCommand variable to record
    record.update({'cvUpCommand': cvUpCommand})

    # get the persistent variables if they exist, if not, initialize them
    if len(record) <= 1:
        # initialize slew duration variable
        record.update({'lastSlewTime': -np.inf})
        # initialize slew start time variable
        record.update({'slewStartTime': time})
        # initialize the pointing vector variable
        record.update({'cvPointingCommand': cvStateObj.CVpointing})
        # initialize the dwelling boolean
        record.update({'dwelling': True})
        # initialize the trackIdsVisistedThisTime variable
        record.update({'trackIdsVisitedThisTime': []})
        # initialize the visited track IDs variable
        record.update({'visitedTrackIds': visitedTrackIds})
    else:
        if time - record['slewStartTime'] < record['lastSlewTime'] + parameterDict['tDwell']:
            if time - record['slewStartTime'] < record['lastSlewTime']:
                cvPointingCommandLast = record['cvPointingCommand']
                # sensor schedule believes the sensor is slewing
                if verbose:
                    print("pointMeManager: Slewing...")
                # set control variables
                record['dwelling'] = True
                record['cvPointingCommand'] = cvPointingCommandLast
                record['trackIdsVisitedThisTime'] = []
            else:
                if verbose:
                    if not record['dwelling']:
                        print("pointMeManager: Dwell starting (duration: {} s)".format(parameterDict['tDwell']))
                    #
                    print('pointMeManager: Dwelling')
                # set control variables
                record['dwelling'] = True
                record['cvPointingCommand'] = cvPointingCommandLast
                tracksUpdatedIds = record['trackIdsVisitedThisTime']
        else:
            tracks = trackStateObj.getActiveStates() 
            nTracks = len(tracks) 
            # if any tracks have been dropped, remove them from visited trackIds
            record['visitedTrackIds'] = np.intersect1d(record['visitedTrackIds'], tracks.Ids)
            # determine which set of trackIds still needs to be visited
            unvisitedTrackIds = np.setdiff1d(tracks.Ids, record['visitedTrackIds'])
            # if there are no more unvisited tracks, start the loop again by
            # declaring that all tracks are unvisited
            if unvisitedTrackIds.size > 0:
                record['visitedTrackIds'] = []
                unvisitedTrackIds = tracks.Ids

            # pick out the tracks that need visiting
            for goodId in unvisitedTrackIds:
                indices = np.where(trackStates.Ids == goodId)[0]
                unVisitedTracksIndices.append(indices[0]) 
            # get the selected trackStates as a new trackState object
            unvisitedTracks = tracks.getTracks(unVisitedTracksIndices)
            # call pointMe to determine the next attitude comman
            newRecord = PointMeTS(cvStateObj, unvisitedTracks, parameterDict, record)
            
            # record additional values for the next time around
            record.update(newRecord)
            record.update({'slewStartTime': time})

            if verbose:
                print('CVPointMeManager: Slew startes at {}'.format(record['slewStartTime']))
                print('CVPointMeManager: Slew duration {}'.format(slewTime))
                print('CVPointMeManager: Target Ids to visit at the end of slew {}'.format(
                    record['trackIdsVisitedThisTime']))

            # update the list of visited tracks
            record.update({'visitedTrackIds': np.union1d(record['visitedTrackIds'], trackIdsVisitedThisTime)})

            # sensor scheduler has just commanded a slew, so we are not dwelling right now
            record['dwelling'] = False

    # release the results
    return record


"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""
