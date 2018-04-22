"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*   Module Name: Classes.py                                                                                 *
*   Author(s): Brent McCoy                                                                                  *
*   Version: 1.0                                                                                            *
*   Date: 12/21/17                                                                                          *
*                                                                                                           *
*       Module Description:                                                                                 *
*           Module that contains functions to handle the propagation of trajectories based on six state     *
*           information. This contains the PropagateECI and PropagateStatesToDivergence modules that can    *
*           be called separately. Descriptions of each routine in the context of inputs and outputs can be  *
*           found on the lines immediately preceding the routine in questions.                              *
*                                                                                                           *
*       Requires:   earthParams.json                                                                        *
*                                                                                                           *
*************************************************************************************************************
"""
import os
import numpy as np
import math
from generalUtilities.utilities import Gravity
from generalUtilities.config import eParms
"""
----------------------------------------------------------------------------------------------
D = drag(sv,beta)

Compute drag coefficient
----------------------------------------
    sv   = eci state vector (m)
    beta = Weight/(Cd * Area) (N/m^2)
    
    D    = drag vector per unit mass (vector in N/kg or m/s^2)
           acts opposite the velocity vector
    Drag = 0 if beta is <= 0
    
    Need Drag per unit mass (N/kg) -> so add to acceleration for
                                      gravity in runge-kutta prop
    
    First order model assumes a constant beta and drag coefficient
----------------------------------------------------------------------------------------------
"""
def drag(sv, beta):
    D = np.zeros(3)

    if beta <= 0:
        print "no drag coefficient specified"
        D = np.zeros((1,3))
    else:
        We = eParms.OMEGA_EARTH

        alt = EarthAltitude(sv[0], sv[1], sv[2])
        rho = AtmosphereQ(alt)

        rdota = np.zeros([3])
        rdota[0] = sv[3] + We*sv[1]
        rdota[1] = sv[4] - We*sv[0]
        rdota[2] = sv[5]
        Va = np.linalg.norm(rdota)

        D = -0.5 * rho * Va * rdota/beta

    return np.reshape(D, (3,1))

"""
----------------------------------------------------------------------------------------------
4th Order Runge Kutta Integration
------------------------------------------------------
    sv_old = initial state vector [X Y Z Vx Vy Vz t] in ECI
    delt   = time step in seconds
    beta   = optional drag coefficient, default valut is none
    
    sv_new = new ECI state vector [X Y Z Vx Vy Vz t] after one
             delt time step
----------------------------------------------------------------------------------------------
"""
def runge_kutta4(sv_old, delt, beta=None):
    f1 = np.zeros(6) 
    f2 = f1 
    f3 = f1 
    f4 = f1 
    sv_new = np.zeros(7) 

    if beta != None:
        # Determine f1
        G1 = Gravity(sv_old)
        D1 = drag(sv_old, beta)        
        f1 = np.append(sv_old[3:6], [G1 + D1]) 
        
        # Determine f2
        temp = sv_old[0:6] + 0.5*delt*f1 
        G2 = Gravity(temp) 
        D2 = drag(temp, beta)        
        f2 = np.append(temp[3:6], [G2 + D2])
        
        # Determine f3
        temp = sv_old[0:6] + 0.5*delt*f2 
        G3 = Gravity(temp)
        D3 = drag(temp, beta)        
        f3 = np.append(temp[3:6], [G3 + D3])
        
        # Determine f4
        temp = sv_old[0:6] + delt*f3 
        G4 = Gravity(temp) 
        D4 = drag(temp, beta)        
        f4 = np.append(temp[3:6], [G4 + D4])
    
    else:
        # Determine f1
        G1 = Gravity(sv_old)
        f1 = np.append(sv_old[3:6], [G1]) 
        
        # Determine f2
        temp = sv_old[0:6] + 0.5*delt*f1 
        G2 = Gravity(temp) 
        f2 = np.append(temp[3:6], [G2])
        
        # Determine f3
        temp = sv_old[0:6] + 0.5*delt*f2 
        G3 = Gravity(temp) 
        f3 = np.append(temp[3:6], [G3])
        
        # Determine f4
        temp = sv_old[0:6] + delt*f3 
        G4 = Gravity(temp) 
        f4 = np.append(temp[3:6], [G4])
        
    # Determine new state vector
    sv_new[0:6] = sv_old[0:6] + delt/6*(f1 + 2*f2 + 2*f3 + f4) 
    sv_new[6] = sv_old[6] + delt 

    return sv_new
  
"""
----------------------------------------------------------------------------------------------
    Inputs:       old_state - 7x1 ECI state vector [x y z dx dy dz t]'
                  beta      - (Scalar) ballistic coefficient, kg/m^2 (0 = no drag)
                  prop_time - (Scalar) time over which to propagate, seconds
                              note: for back-propagation, use negative prop_time
    
    Outputs:      new_state - 7x1 Propagated ECI state vector [x y z dx dy dz
                                                       t+prop_time]'
----------------------------------------------------------------------------------------------
"""
def PropagateECI(old_state, beta, prop_time):
    if beta == 0.0:
        integ_time_step_max = 25.0
    else:
        integ_time_step_max = 0.250

    new_state = np.reshape(old_state, (7))

    num_iterations = int( np.ceil( np.absolute(prop_time) / integ_time_step_max ) )
    
    if num_iterations != 0:
        integ_time_step = prop_time / num_iterations

        if beta == 0.0:
            for i in range(num_iterations):
                new_state = runge_kutta4(new_state, integ_time_step)
        else:
            for i in range(num_iterations):
                new_state = runge_kutta4(new_state, integ_time_step, beta)

    return new_state

"""
----------------------------------------------------------------------------------------------
    
    Inputs:       state         = initial ECI state vector [x y z vx vy vz validity_time]
                  covariance    = initial ECI 6x6 covariance matrix
                  propagateTime = total time to propagate in seconds
                  deltaT        = integration time step in seconds
                                  (delt > 0) ==> propagate forward
                                  (delt < 0) ==> propagate backward
    
    Outputs:      new_state     = new (propagated) ECI state vector
                  new_cov       = new (propagated) ECI covariance matrix(6x6)
    
----------------------------------------------------------------------------------------------
"""
def PropagateECICov(state, covariance, propagateTime, deltaT):
    muEarth = eParms.MU

    # intialize necessary variables
    newCov = covariance.copy()
    oldState = state.copy()
    oldCov = covariance
    propTime = np.absolute(propagateTime)

    # if propTime = 0 then do not propagate state
    if propTime != 0:
        # figure out how many iterations to make (// indicates integer division)
        numIterations = int( np.absolute(propTime // deltaT) )
        time_left = propTime - numIterations * np.absolute(deltaT)
        # if numIterations results in a partial step then increase it by 1
        if propTime % deltaT != 0:
            numIterations += 1

        # loop through iterations
        for ii in range(numIterations):
            # if partial step, change the delaT
            if ii == numIterations-1 and time_left > 0:
                if deltaT > 0:
                    deltaT = time_left
                else:
                    deltaT = -time_left
            # propagate the state:
            newState = PropagateECI(oldState, 0.0, deltaT)

            xOld = oldState[0:3].reshape(3,1)
            rOld = np.linalg.norm(xOld)
            #if rOld == 0:
            #    print("vector: \n{} \nmagnitude: {}".format(xOld, rOld))
            xNew = newState[0:3].reshape(3,1)
            rNew = np.linalg.norm(xNew)
            #if rNew == 0:
            #    print("vector: \n{} \nmagnitude: {}".format(xNew, rNew))
            
            # compute the parial gravity wrt position
            if rOld == 0:
                DgDx_old = np.zeros((3,3))
            else:
                DgDx_old = ( (3*muEarth) / (rOld**5) ) * np.dot(xOld, xOld.transpose())
                DgDx_old = ( muEarth / (rOld**3) ) * np.identity(3)

            if rNew == 0:
                DgDx_new = np.zeros((3,3))
            else:
                DgDx_new = ( (3*muEarth) / (rNew**5) ) * np.dot(xNew, xNew.transpose())
                DgDx_new = ( muEarth / (rNew**3) ) * np.identity(3)
            
            lowerLeft = (deltaT / 2) * (DgDx_old + DgDx_new)
            upperLeft = np.identity(3)
            
            upperRight = np.identity(3) * deltaT
            lowerRight = np.identity(3)

            scale = np.vstack(( np.hstack((upperLeft, upperRight)), 
                np.hstack((lowerLeft, lowerRight)) ))

            newCov = np.dot( np.dot(scale, covariance), scale.transpose() )

            # save the old state and covariance
            oldCov = newCov.copy()
            oldState = newState.copy()
    else:
        newState = oldState.copy()

    # return the newly calculated values
    return newState, newCov

"""
----------------------------------------------------------------------------------------------
    
    Inputs:       sECI1  - 7x1 Initial state for first object in ECI
                           [x y z vx vy vz t]
                  sECI2  - 7x1 Initial state for second object in ECI
                           [x y z vx vy vz t]
                  dT     - (Scalar) Time increment for intermediate points [sec]
    
    Outputs:      traj1 - Output trajectory (1st object) N x [x y z vx vy vz t]
                  traj2 - Output trajectory (2nd object) N x [x y z vx vy vz t]
    
----------------------------------------------------------------------------------------------
"""
def PropagateStatesToDivergence(sECI1, sECI2, dT):
    # iniialize memory for the trajectories
    traj1 = np.zeros((5000, 7))
    traj2 = np.zeros((5000, 7))

    # ensure input states are aligned in time
    t1 = sECI1[6]
    t2 = sECI2[6]

    if t1 < t2:
        sECI1 = PropagateECI(sECI1, 0.0, t2 - t1)
    elif t2 < t1:
        sECI2 = PropagateECI(sECI2, 0.0, t1 - t2)

    # propagate in increments of dT until we reach tFinal
    n = 0
    traj1[n, :] = sECI1
    traj2[n, :] = sECI2
    s1 = sECI1
    s2 = sECI2
    t = sECI1[6]
    diverging = False
    distMin = np.finfo(np.float64).max      # realmax in MatLab
    while not diverging:
        sProp1 = PropagateECI(s1, 0.0, dT)
        sProp2 = PropagateECI(s2, 0.0, dT)
        s1 = sProp1
        s2 = sProp2
        t = t + dT
        traj1[n, :] = sProp1
        traj2[n, :] = sProp2
        dist = np.linalg.norm( np.subtract(sProp1[0:3], sProp2[0:3]) )
        if dist > distMin:
            diverging = True
        else:
            distMin = dist
        n += 1

    # handle at or beyond the POCA case
    r = n - 1
    if r <= 2:
        for i in range(3-r):
            sProp1 = PropagateECI(traj1[n-1], 0.0, dT)
            traj1[n] = sProp1
            sProp2 = PropagateECI(traj2[n-1], 0.0, dT)
            traj2[n] = sProp2
            n += 1

    # output final trajectories
    return traj1[0:n], traj2[0:n]

"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""
