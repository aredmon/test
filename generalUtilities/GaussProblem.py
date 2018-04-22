"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*   Module Name: GaussProblem.py                                                                            *
*   Author(s): Mark Lambrecht, Brent McCoy                                                                  *
*   Version: 1.0                                                                                            *
*   Date: 12/20/17                                                                                          *
*                                                                                                           *
*       Module Description: This function propagates the input state to a specified time                    *
*                           "tFinal" in increments of "dT" seconds.  The resulting                          *
*                           trajectory is saved and returned in "traj".                                     *
*                                                                                                           *
*           Algorithm:  See Bate, Mueller, White                                                            *
*                                                                                                           *
*           Inputs:     pos1 - 3-Position of first trajectory                                               *
*                       pos2 - 3-Position of second trajectory                                              *
*                       tof  - (Scalar) Time to POCA, s                                                     *
*                                                                                                           *
*           Outputs:    sv1 - Output state vector of first trajectory (1x6)                                 *
*                       sv2 - Output state vector of second trajectory (1x6)                                *
*                                                                                                           *
*           Calls:                                                                                          *
*                                                                                                           *
*           Requires:   earthParams.json                                                                    *
*                                                                                                           *
*           OA:         Mark Lambrecht                                                                      *
*                                                                                                           *
*           History:    MAL 13 Dec 2017:  Initial version                                                   *
*                                                                                                           *
*************************************************************************************************************
"""
import os
import json
import numpy as np
from generalUtilities.config import eParms

def GaussProblem(pos1, pos2, tof):
    # returns an object with the json keys mapped to fields
    # to access the data use object.field syntax. For information 
    # on the available fields you can use the jsonInfo class to pull 
    # out descritions of each field.
    #if not os.path.isdir('sim'):
    #    if os.getcwd().rsplit('/', 1)[-1] != 'sim':
    #        currentDir = os.path.dirname(os.path.realpath(os.getcwd()))
    #        filename = os.path.join(currentDir, 'sim', 'earthParams.json')
    #    else:
    #        filename = 'earthParams.json'
    #else:
    #    filename = 'sim/earthParams.json'
    ## load file
    #eParms = jsonData(filename)
    mu = eParms.MU;
    recSqMu = 1/np.sqrt(mu);
    eps1 = 1.0e-8;
    #eps1 = 0.0001;
    eps2 = 1.0e-8;
    
    # Main code
    count = 0;
    #r1 = np.sqrt(sum(pos1.^2));
    #r2 = np.sqrt(sum(pos2.^2));
    r1 = np.linalg.norm(pos1)
    r2 = np.linalg.norm(pos2)
    cosNu = np.dot(pos1,pos2)/(r1*r2)
    #if r1 != 0 and r2 != 0:
    #    cosNu = np.dot(pos1,pos2)/(r1*r2)
    #else:
    #    cosNu = 1.0
    A = np.sqrt(r1*r2*(1+cosNu))
    zz = 0;
    t = -1;
    while abs(tof - t) > eps1 and count < 1000:
        if abs(zz) > eps2:
            if zz > eps2:
                cc = (1 - np.cos(np.sqrt(zz))) / zz
                ss = (np.sqrt(zz) - np.sin(np.sqrt(zz))) / np.sqrt(zz*zz*zz)
            else:
                cc = (1 - np.cosh(np.sqrt(-zz))) / zz
                ss = (np.sinh(np.sqrt(-zz)) - np.sqrt(-zz)) / np.sqrt((-zz)*(-zz)*(-zz))
        else: 
            cc = 1.0/2.0 - zz/24.0 + zz*zz/720.0-zz*zz*zz/40320.0
            ss = 1.0/6.0 - zz/120.0 + zz*zz/5040.0 - zz*zz*zz/362880.0

        sqc = np.sqrt(cc)
        count += 1
        y = r1 + r2 - A*(1-zz*ss)/sqc
        
        if y < 0.0:
            sv1 = np.zeros(6)
            sv2 = np.zeros(6)
            break
        sqy = np.sqrt(y)  
        x = sqy / sqc

        if x == 0:
            sv1 = np.zeros(6)
            sv2 = np.zeros(6)
            break

        x3 = x*x*x
        t = (x3*ss + A*sqy)*recSqMu
        z2 = zz*zz
        z3 = z2*zz
        Cp = 1.0/24.0 + 2.0*zz/720.0 - 3.0*z2/40320.0 + 4.0*z3/3628800.0
        Sp = 1.0/120.0 + 2.0*zz/5040.0 - 3.0*z2/362880.0 + 4.0*z3/39916800.0
        dtdz = (x3*(Sp - 3.0*ss*Cp/(2.0*cc)) + A/8.0*(3.0*ss*sqy/cc + A/x))*recSqMu
        zz = zz + (tof - t)/dtdz

    f = 1 - y/r1
    g = A*sqy*recSqMu
    gd = 1 - y/r2
    sv1 = np.zeros(6)
    sv2 = np.zeros(6)
    sv1[0:3] = pos1
    sv2[0:3] = pos2
    sv1[3:6] = pos2/g - f/g*pos1
    sv2[3:6] = gd/g*pos2 - pos1/g

    return sv1, sv2
    
"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""

