"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*   Module Name: GenerateTestCases                                                                          *
*   Author(s): David Vallado, Brent McCoy                                                                   *
*   Version: 1.0                                                                                            *
*   Date: 03/30/18                                                                                          *
*                                                                                                           *
*       Module          This function solves Keplers problem for orbit determination and returns a          *
*       Description:    future geocentric equatorial (i, j, k) position and velocity vector. The            *
*                       solution uses universal variables.                                                  *
*                                                                                                           *
*       Algorithm:      N/A                                                                                 *
*                                                                                                           *
*       Inputs:         r0          -   initial (i, j, k) position vector in km                             *
*                       v0          -   initial (i, j, k) velocity vector in km/s                           *
*                       propTime    -   length of time to propagate in seconds                              *
*                                                                                                           *
*       Outputs:        rFinal      -   (i, j, k) position vector in km                                     *
*                       vFinal      -   (i, j, k) velocity vector in km/s                                   *
*                       error       -   optional error for calculation                                      *
*                                                                                                           *
*       Calls:          N/A                                                                                 *
*                                                                                                           *
*       Requires:       N/A                                                                                 *
*                                                                                                           *
*       OA:             David Vallado                                                                       *
*                                                                                                           *
*       History:        DV 13 Apr 2004: Updates to Initial Version                                          *
*                                                                                                           *
*************************************************************************************************************
"""
import os
import numpy as np
from generalUtilities.config import eParms, physMath
    
# minimum value acceptable
minVal = 1e-8

class ConvergenceError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)

class ZeroResult(Exception):
    pass

def findC2C3Energies( zed ):
    if zed > minVal:
        zSqrt = np.sqrt( zed )
        #print("sqrt(z): {}".format(zSqrt))
        c2energy = (1.0 - np.cos(zSqrt)) / zed
        c3energy = (zSqrt - np.sin( zSqrt )) / ( zSqrt**3 )
    else:
        if zed < -minVal:
            zSqrt = np.sqrt( -zed )
            #print("sqrt(z): {}".format(zSqrt))
            c2energy = (1.0 - np.cosh(zSqrt)) / zed
            c3energy = (np.sinh( zSqrt ) - zSqrt) / ( zSqrt**3 )
        else:
            c2energy = 0.5
            c3energy = 1.0/6.0
    return c2energy, c3energy

def Kepler(r0, v0, propTime, maxIterations=35):
    rFinal = r0
    vFinal = v0

    if np.absolute(propTime) > minVal:
        rMag = np.linalg.norm(r0)
        vMag = np.linalg.norm(v0)

        dist = np.dot(r0, v0)
        #maxIterations = 35
        #--------------------------- find sME, aplha, and semiMajor axis ------------------------- 
        #   this section uses the following variable names for simplification:
        #       semiMajor(a)    - semi-major axis of an ellipse
        #       alpha           - 1/a (again a is the semiMajor axis)
        #       sME             - specific mechanical energy (total orbital energy per unit mass)
        #-----------------------------------------------------------------------------------------
        sME = ( (vMag**2)/2.0 ) - ( eParms.MU/rMag )  
        alpha = (-2.0*sME) / eParms.MU            

        if np.absolute( sME ) > minVal:
            semiMajor = 1 / alpha
        else:
            semiMajor = np.inf

        if np.absolute( alpha ) < minVal:
            alpha = 0.0
        #--------------------------------- setup initial guess for x -----------------------------
        #       period          - calculate period of the orbit
        #----------------------------- circular and ellisptical oribits --------------------------
        if alpha >= minVal:
            period = physMath.twopi * np.sqrt( np.absolute(semiMajor)**3 / eParms.MU )
            #---------------------------- if needed for 2body multi-rev --------------------------
            if np.absolute( propTime ) > np.absolute(period):
                propTime = propTime % period
            if np.absolute( alpha-1.0 ) > minVal:
                xOld = propTime * alpha
            else:
                xOld = propTime * alpha * 0.97
        else:
            #-------------------------------- parabolic orbits -----------------------------------
            #   angMom          - specific relative angular momentum
            #   focus           - radius from a vertex to the focus or a focus of the orbit
            #   eccentricity    - eccentric annomally of the orbit
            #-------------------------------------------------------------------------------------
            if np.absolute( alpha ) < minVal:
                angMom = np.cross(r0, v0)
                momentum = np.linalg.norm(angMom)
                focus = momentum * momentum / eParms.MU
                eccentricity = 0.5 * (physMath.halfpi 
                        - np.arctan(3.0*propTime*np.sqrt(eParms.MU/(focus*focus*focus))))
                theta = np.arctan( np.tan(dist)**(1.0/3.0) )
                xOld = np.sqrt(focus) * ( 2.0/np.tan(2.0*theta) )
                alpha = 0.0
            else:
                #------------------------------ hyperbolic orbits ---------------------------------
                print("temp variables \nmu: {} \nsemiMajor: {} \nsME: {} \nrMag:{} \nalpha: {}".format(
                    eParms.MU, semiMajor, sME, rMag, alpha))
                temp = -2.0 * eParms.MU * propTime / ( semiMajor*( dist + 
                    np.sign(propTime)*np.sqrt(eParms.MU*semiMajor)*(1.0 - rMag*alpha) ) )
                xOld = np.sign(propTime) * np.sqrt(-dist) * np.log(temp)
        #------------------------------------------------------------------------------------------
        count = 0
        convergence = 0
        newTime = -10.0
        while np.absolute(newTime/np.sqrt(eParms.MU) - propTime) >= minVal and count < maxIterations:
            xOldSqrd = xOld*xOld
            zNew = xOldSqrd * alpha
            #------------------------ find c2 energies and c3 energies ----------------------------
            c2energy, c3energy = findC2C3Energies( zNew )

            #------------------- use newton's method for finding new values -----------------------
            rVal = xOldSqrd*c2energy + dist/np.sqrt(eParms.MU) * xOld * (1.0 - zNew*c3energy) + rMag*( 
                    1.0 - zNew*c3energy )
            newTime = xOldSqrd * xOld * c3energy + dist/np.sqrt(eParms.MU)*xOldSqrd*c2energy + rMag*xOld*(
                    1.0 - zNew*c3energy )

            #------------------------------ calculate new value for x -----------------------------
            xNew = xOld + ( propTime*np.sqrt(eParms.MU) - newTime ) / rVal

            try:
                convergence = np.absolute(newTime/np.sqrt(eParms.MU) - propTime)
            except ZeroDivisionError:
                convergence = np.inf

            count += 1
            xOld = xNew
        #-------------------------------------- error handling -------------------------------------
        if np.absolute(newTime/np.sqrt(eParms.MU) - propTime) >= minVal and count >= maxIterations:
            try:
                raise ConvergenceError(convergence)
            except ConvergenceError:
                print("ConvergenceError: exceeded specific number of iterations before convergence was achieved")
            vFinal = np.zeros(3)
            rFinal = np.zeros(3)
        else:
            #--------------------------- find new position and velocity ----------------------------
            xNewSqrd = xNew * xNew
            fCalc = 1.0 - (xNewSqrd*c2energy / rMag)
            gCalc = propTime - xNewSqrd*xNew*c3energy/np.sqrt(eParms.MU)

            rFinal = fCalc*r0 + gCalc*v0
            rMagFinal = np.linalg.norm( rFinal )

            gDot = 1.0 - ( xNewSqrd*c2energy / rMagFinal )
            fDot = ( np.sqrt(eParms.MU)*xNew / ( rMagFinal*rMag ) ) * ( zNew*c3energy - 1.0 )

            vFinal = fDot*r0 + gDot*v0
            vMagFinal = np.linalg.norm( vFinal )

            temp = fCalc*gDot - fDot*gCalc
            
            if np.absolute(temp-1.0) > 0.00001:
                raise ZeroResult("the final result is ~ 0, check the c2/c3 energy" + 
                "calculations: {} and {}".format(c2energy, c3energy))
    # return the final result
    return rFinal, vFinal

"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""
