"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*   Module Name: UnitConverter                                                                              *
*   Author(s): Mark Lambrecht, Brent McCoy                                                                  *
*   Version: 1.0                                                                                            *
*   Date: 04/05/18                                                                                          *
*                                                                                                           *
*       Module          This module contains a series of unit conversion routines necessary to run the      *
*       Description:    sensor scheduler algorithm suite.                                                   *
*                                                                                                           *
*       Algorithm:      N/A                                                                                 *
*                                                                                                           *
*       Modules:        ECI2Observer    - convert observable and multiple target states in ECI to           *
*                                       observer reference frame (x-forward, z-up, y-left).                 *
*                                                                                                           *
*                       AzEl2Attitude   - convert the observer frame (az/el) to attitude (pointing and      *
*                                       up vector in ECI). In this convention, az points right, el points   *
*                                       down                                                                *
*                                                                                                           *
*                       Observer2AzEl   - convert observer frame states to az/el and azDot/elDot.           *
*                                                                                                           *
*       OA:             M. A. Lambrecht                                                                     *
*                                                                                                           *
*       History:        MAL 22 Jan 2018:  Initial version                                                   *
*                                                                                                           *
*************************************************************************************************************
"""
import numpy as np
from generalUtilities.config import *
"""
----------------------------------------------------------------------------------------------
    Inputs:     time    -   time epoch in seconds

    Outputs:    Wex     -   
                Tex     -
----------------------------------------------------------------------------------------------
"""
def earthRotMat(time):
    # rotation matrices for ECI - ECR conversions
    omegaEarth = eParms.OMEGA_EARTH

    delLon = time * omegaEarth

    # Euler Rotation Matrix (CCW about z-axis)
    cosLon = np.cos(delLon)
    sinLon = np.sin(delLon)

    # generate Tex matrix
    Tex = np.array([
        [cosLon, sinLon, 0],
        [-sinLon, cosLon, 0],
        [0, 0, 1]])

    # generate Wex matrix
    Wex = np.array([
        [0, -omegaEarth, 0],
        [omegaEarth, 0, 0],
        [0, 0, 0]])
    # release results
    return Wex, Tex

"""
----------------------------------------------------------------------------------------------
    Inputs:     state       -   ECEF state vector [x, y,z, dx, dy, dz, t]

    Outputs:    newState    -   ECI state vector [x, y, z, dx, dy, dz, t]
----------------------------------------------------------------------------------------------
"""
def ECEF7toECI7(state):
    newState = state.copy()

    Wex, Tex = earthRotMat(state[6])

    # Rotate position vector
    newState[0:3] = np.dot(np.transpose(Tex), state[0:3])

    # Rotate velocity vector
    newState[3:6] = np.dot(np.transpose(Tex), state[3:6]) \
            + np.dot(Wex, np.dot(np.transpose(Tex), state[0:3]))

    # release results
    return newState

"""
----------------------------------------------------------------------------------------------
    Inputs:     state       -   ECI state vector [x, y,z, dx, dy, dz, t]

    Outputs:    newState    -   ECEF state vector [x, y, z, dx, dy, dz, t]
----------------------------------------------------------------------------------------------
"""
def ECI7toECEF7(state):
    newState = state.copy()

    Wex, Tex = earthRotMat(state[6])

    # Rotate position vector
    newState[0:3] = np.dot(Tex, state[0:3])

    # Rotate velocity vector
    newState[3:6] = np.dot(Tex, state[3:6]) + np.dot(Wex, np.dot(Tex, state[0:3]))

    # release results
    return newState

"""
----------------------------------------------------------------------------------------------
    Inputs:       rObserver - 3x1 CV position, ECI, km
                  vObserver - 3x1 CV velocity, ECI, km/s
                  ptVector  - 1x3 CV pointing vector (unit), ECI
                  upVector  - 3x1 CV up vector (unit), CV frame
                  rTarget   - 3x#Tracks target position, ECI, km
                  vTarget   - 3x#Tracks target velocity, ECI, km/s
    
    Outputs:      dRrot     - 3x#Tracks target position, observer frame, km
                  dVrot     - 3x#Tracks target velocity, observer frame, km/s
                  jacobian  - 3x3 transformation matrix (ECI->Observer)
----------------------------------------------------------------------------------------------
"""
def ECI2Observer(rObserver, vObserver, ptVector, upVector, rTarget, vTarget):
    M1 = ptVector / np.linalg.norm(ptVector)
    M2 = np.cross(upVector, ptVector)
    M2 = M2 / np.linalg.norm(M2)
    M3 = np.cross(M1, M2)

    jacobian = np.vstack((M1, M2, M3))

    # rotate to the Observer frame
    dRrot = np.dot( rTarget - rObserver, jacobian )
    dVrot = np.dot( vTarget - vObserver, jacobian )

    # release results
    return drRot, dVrot, jacobian

"""
----------------------------------------------------------------------------------------------
    Inputs:       rObserver       - 3x1 position of observer, ECI, km
                  azelVector      - 2x1 vector of azimuth/elevation angles,
                                    radians
                  currentPtVector - 1x3 pointing vector (unit) of observer
                  currentUpVector - 3x1 up vector (unit) of observer
    
    Outputs:      ptVector - 3x1 pointing vector (unit) of observer
                  upVector - 1x3 up vector (unit) of observer
----------------------------------------------------------------------------------------------
"""
def AzEl2Attitude(rObserver, azelVector, currentPtVector, currentUpVector):
    # utility function for rotation about an axis
    def axialRotate(vector, axis, angle):
        newVec = vector*np.cos(angle) + axis*( np.dot(axis, vector)*(1-np.cos(angle)) ) + \
                np.cross(vector, axis)*np.sin(angle)
        # release transformed vector
        return newVec

    # begin transformations:
    elevation = azelVector[1]
    azimuth = azelVector[0]

    # normalize the current pointing vector
    olPointingVector = currentPtVector / np.linalg.norm(currentPtVector)

    # up vector is just straight up from the center of the earth
    upVector = rObserver / np.linalg.norm(rObserver)

    # apply angle transformation to the current pointing vector
    ## elevation angle:
    elevationAxis = np.cross(currentPtVector, currentUpVector)
    elAxisNormalized = elevationAxis / np.linalg.norm(elevationAxis)
    elevatedPtVector = axialRotate(currentPtVector, elAxisNormalized, elevation)
    
    ## azimuth angle
    azimuthAxis = np.cross(elAxisNormalized, elevatedPtVector)
    ptVector = axialRotate(elevatedPtVector, azimuthAxis, azimuth)

    # release results
    return ptVector, upVector

"""
----------------------------------------------------------------------------------------------
    Inputs:       rObserved - 3x#Tracks position in observer's frame, km
                  vObserved - 3x#Tracks velocity in observer's frame, km/s
    
    Outputs:      angle    - 2x#Tracks az/el, radians
                  angledot - 2x#tracks azDot/elDot, radians/s
----------------------------------------------------------------------------------------------
"""
def Observer2AzEl(rObserved, vObserved):
    xVals = rObserved[:, 0]
    yVals = rObserved[:, 1]
    zVals = rObserved[:, 2]

    xDots = vObserved[:, 0]
    yDots = vObserved[:, 1]
    zDots = vObserved[:, 2]

    rHorizon2 = np.square(xVals) + np.square(yVals)
    rHorizon = np.sqrt(rHorizon2)
    vHorizon = (np.dot(xVals, xDots) + np.dot(yVals, yDots)) / rHorizon

    rTotal2 = rHorizon2 + np.square(zVals)
    rTotal = np.sqrt(rTotal2)
    
    elevation = -np.arctan2(zVals, rHorizon)
    azimuth = -np.arctan2(yVals, xVals)

    elDots = ( np.dot(zVals, vHorizon) - np.dot(zDots, rHorizon) ) / rTotal2
    azDots = ( np.dot(xDots, yVals) - np.dot(yDots, xVals) ) / rHorizon2

    # release the results
    return np.hstack((elevation, azimuth)), np.hstack((elDots, azDots))

"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""
