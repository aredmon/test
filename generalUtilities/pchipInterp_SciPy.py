"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*   Module Name: pchipInterp.py                                                                             *
*   Author(s): Brent McCoy                                                                                  *
*   Version: 1.0                                                                                            *
*   Date: 01/22/18                                                                                          *
*                                                                                                           *
*       Module Description:                                                                                 *
*           Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) routine for python based on            *
*           Dr. Raymond J. Spiteir's MatLab pchip routine. Source code was adapted from the MatLab          *
*           program he provided to his MATH 211 class ad the University of Saskatchewan in 2013.            *
*           <https://www.cs.usask.ca/~spiteri/M211/notes/pchiptx.m>                                         *
*                                                                                                           *
*       Modifications: changes have been made to the original code design with respect to the way the       *
*                      dervatives and end-points are calculated. Agreement with original pchip theory       *
*                      has been confirmed and the new routine has been tested. `pchipInter.py` is now       *
*                      deprecated.                                                                          *
*                                                                                                           *
*       NOTE: This code has been designed to work with a y input of either a vector or array of vectors     *
*             if single vector functionality is not desired any cases involving y.ndim==1 can be safely     *
*             removed. The opposite is true if array of vectors support is not desired.                     *
*                                                                                                           *
*************************************************************************************************************
"""
import numpy as np

def findDerivatives(x, y):
    # non-centered, shape-preserving, three-point formula calculating slopes, d(k), from data
    # based loosely on the Fitsch-Carlson method where the weighted harmonic means are computed for 
    # the slopes between four points: 
    #       when sgn(slope1)=sgn(slope2) - 
    #               3(xSep1 + xSep2)*((2xSep1 - xsep2)/(slope2) - (xsep1+2xSep2)/(slope1)^(-1)
    #       when sng(slope1)!=sgn(slope2) -
    #                                           0
    # need to deal with end-point slopes:
    def edgeCase(slopeLeft, slopeRight, slopeOut):
        #print("need to treat end points separately for contiuity of the polynomial")
        # need to treat end points separately for contiuity of the polynomial
        # calculate the slope as the weighted average of the left and right slope
        # If the left and/or right slope = 0, however, set the slope = 0
        try:
            slope0 = np.atleast_1d(slopeLeft)
            slope1 = np.atleast_1d(slopeRight)
            boolMask = (slope1 != 0) & (slope0 != 0)
            slopeOut[boolMask] = (slope0[boolMask] * slope1[boolMask]) / (slope0[boolMask] + slope1[boolMask])
        except TypeError:
            if slopeLeft.any() == 0 or slopeRight.any() == 0:
                slopeOut = 0
            else:
                slopeOut = (slopeLeft * slopeRight) / (slopeLeft + slopeRight)

    #print("compute the slopes of the interior points according to a 3-point modified version of the")
    # compute the slopes of the interior points according to a 3-point modified version of the 
    # Fritsch-Carlson method highlighted above.
    # slope = (Yi+1 - Yi)/(Xi+1 - Xi) (estimated from delta(Y)/delta(X) in initial data
    # xSep = (Xi+1 - Xi)
    # let d(k) be the slope of the kth segment (between k and k+1)
    # If d(k)=0 or d(k-1)=0 or sgn(d(k)) != sgn(d(k-1)) then d(k) == 0
    # else use weighted harmonic mean highlighted above
    if y.ndim == 1:
        xSep = x[1:] - x[:-1]
        slope= np.divide((y[1:] - y[:-1]), xSep)   # same length as xSep (n-1)

        numPts = xSep.shape[0]
        signCheck = np.sign(slope)
        condition = ( (signCheck[1:] != signCheck[:-1]) | (slope[1:] == 0) | (slope[:-1] == 0) )

        weight1 = 2*xSep[1:] + xSep[:-1]
        weight2 = xSep[1:] + 2*xSep[:-1]

        #print("calculate slope based on formula")
        # calculate slope based on formula
        # values where divide by zero occurs should be excluded by condition
        with np.errstate(divide='ignore'):
            slopeFinalInv = 1.0/(weight1+weight2) 
            slopeFinalInv = slopeFinalInv * (np.divide(weight1,slope[1:]) + np.divide(weight2,slope[:-1]))
        #print("set up final slope")
        # set up final slope:
        slopeFinal = np.zeros_like(y)
        slopeFinal[1:-1][condition] = 0.0
        slopeFinal[1:-1][~condition] = (1.0 / slopeFinalInv[~condition])

    else:
        #print("else use weighted harmonic mean highlighted above")
        xSep = x[1:] - x[:-1]
        slope= np.divide((y[1:] - y[:-1]).transpose(), xSep).transpose()   # same length as xSep (n-1)

        numPts = xSep.shape[0]
        signCheck = np.sign(slope)
        condition = ( (signCheck[1:] != signCheck[:-1]) | (slope[1:] == 0) | (slope[:-1] == 0) )

        weight1 = 2*xSep[1:] + xSep[:-1]
        weight2 = xSep[1:] + 2*xSep[:-1]

        weightTotal = weight1 + weight2

        #print("calculate slope based on formula")
        # calculate slope based on formula
        # values where divide by zero occurs should be excluded by condition
        with np.errstate(divide='ignore'):
            slopeFinalInv = 1.0/(weightTotal) 
            slopeFinalInv = slopeFinalInv * (weight1/np.transpose(slope[1:]) + weight2/np.transpose(slope[:-1]))

        #print("set up final slope")
        # set up final slope:
        slopeFinal = np.zeros_like(y)
        slopeFinal[1:-1][condition] = 0.0
        newSlope = 1.0/np.transpose(slopeFinalInv)
        slopeFinal[1:-1][~condition] = (1.0 / np.transpose(slopeFinalInv)[~condition])

    #print("calculate the slopes at the endpoints")
    # calculate the slopes at the endpoints:
    edgeCase(slope[0], slope[1], slopeFinal[0])
    edgeCase(slope[-1], slope[-2], slopeFinal[1])

    return slopeFinal

def pchip(x, y, xi):
    # x is a vector of depent variable points for some function f
    # y is a vector or an array of the corresponding function values, f(x)
    # xi are the points of interest where you want to evaluate the resultant interpolating function
    # returns a vector of values for the Piecewise Cubic Polynomial Hermite Interpolating Polynomial, Pi(X)
    # for the data (X, Y) evaluate at the points of interest, xi
    #print("approximate the first derivatives and pass them to pchipSlopes for refinement:")
    # approximate the first derivatives and pass them to pchipSlopes for refinement:
    if y.ndim == 1:
        xSep = x[1:] - x[:-1]                           # if n=length(x) then length(xSep) = n-1
        evalData = np.zeros(xi.shape[0])
        slopeInit = np.divide((y[1:] - y[:-1]), xSep)   # same length as xSep (n-1)

        numPts = x.shape[0]
        slopeFinal = findDerivatives( x, y)             # same length as x (n)

        #print("calculate the piecewise polynomial coefficients for the polynomial")
        # calculate the piecewise polynomial coefficients for the polynomial of the following form
        # for x in [Xi, Xi+1]:    Pi(X) = Yi + Yi'(X-Xi) + Ci(X-Xi)^2 + Di(X-Xi)^3
        #        "slopeFinal[1:]): \n{}, {}, {}".format(slopeInit.shape, 
        #            slopeFinal[0:-1].shape, slopeFinal[1:].shape))
        Ci = np.divide( 3*slopeInit - 2*slopeFinal[0:-1] - slopeFinal[1:], xSep )
        Di = np.divide( slopeFinal[0:-1] - 2*slopeInit + slopeFinal[1:], np.square(xSep) )
        #print("find the subinterval indices, kk, where x(kk) <= xi < x(kk+1)")
        # find the subinterval indices, kk, where x(kk) <= xi < x(kk+1)
        # let XminusXi stand for (X - Xi)
        # evalData will be the vector of 
        tmpArray = np.zeros(x.shape[0])
        for kk, point in enumerate(xi):
            tmpArray = x / point 
            primeIndx = np.argmax((tmpArray > 1))-1

            XminusXi = point - x[primeIndx]
            evalData[kk] = y[primeIndx] + XminusXi * (slopeFinal[primeIndx] + 
                    XminusXi * (Ci[primeIndx] + XminusXi*Di[primeIndx]))

    else:
        xSep = x[1:] - x[:-1]                           # if n=length(x) then length(xSep) = n-1
        evalData = np.zeros((xi.shape[0], y.shape[1]))
        slopeInit = np.divide((y[1:] - y[:-1]).transpose(), xSep).transpose()   # same length as xSep (n-1)

        numPts = x.shape[0]
        slopeFinal = findDerivatives( x, y)             # same length as x (n)

        #print("calculate the piecewise polynomial coefficients for the polynomial")
        # calculate the piecewise polynomial coefficients for the polynomial of the following form
        # for x in [Xi, Xi+1]:    Pi(X) = Yi + Yi'(X-Xi) + Ci(X-Xi)^2 + Di(X-Xi)^3
        #        "slopeFinal[1:]): \n{}, {}, {}".format(slopeInit.shape, 
        #            slopeFinal[0:-1].shape, slopeFinal[1:].shape))
        Ci = np.zeros_like(slopeInit)
        Di = np.zeros_like(slopeInit)
        for column in range(slopeFinal.shape[1]):
            Ci[:,column] = np.divide( 3*slopeInit[:,column] - 2*slopeFinal[0:-1, column] - slopeFinal[1:, 
                column], 
                    xSep )
            Di[:,column] = np.divide( slopeFinal[0:-1, column] - 2*slopeInit[:, column] + slopeFinal[1:, 
                column], np.square(xSep) )

        #print("find the subinterval indices, kk, where x(kk) <= xi < x(kk+1)")
        # find the subinterval indices, kk, where x(kk) <= xi < x(kk+1)
        # let XminusXi stand for (X - Xi)
        # evalData will be the vector of 
        tmpArray = np.zeros(x.shape[0])
        for kk, point in enumerate(xi):
            tmpArray = x / point 
            primeIndx = np.argmax((tmpArray > 1))-1

            XminusXi = point - x[primeIndx]
            evalData[kk] = y[primeIndx] + XminusXi * (slopeFinal[primeIndx, :] + 
                    XminusXi * (Ci[primeIndx] + XminusXi*Di[primeIndx, :]))

    # return the evaluated Pi(X) at the points in the Xi vector
    return evalData

"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""
