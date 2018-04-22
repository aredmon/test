"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*   Module Name: pchipInterp.py                                                                             *
*   Author(s): Brent McCoy                                                                                  *
*   Version: 1.0                                                                                            *
*   Date: 01/22/18                                                                                          *
*                                                                                                           *
*      !!!WARNING!!! - Code is deprecated as of 01/26/18. If you want to use the pchip interpolator         *
*      !!!WARNING!!!   in place of SciPy's pchip_interpolate please use the pchipInterp_SciPy.py            *
*      !!!WARNING!!!   module.                                                                              *
*                                                                                                           *
*       Module Description:                                                                                 *
*           Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) routine for python based on            *
*           Dr. Raymond J. Spiteir's MatLab pchip routine. Source code was adapted from the MatLab          *
*           program he provided to his MATH 211 class ad the University of Saskatchewan in 2013.            *
*           <https://www.cs.usask.ca/~spiteri/M211/notes/pchiptx.m>                                         *
*************************************************************************************************************
"""
import numpy as np

def pchipSlopes(xSep, slope):
    # non-centered, shape-preserving, three-point formula calculating slopes, d(k), from data
    # based loosely on the Fitsch-Carlson method where the weighted harmonic means are computed for 
    # the slopes between four points: 
    #       when sgn(slope1)=sgn(slope2) - 
    #               3(xSep1 + xSep2)*((2xSep1 - xsep2)/(slope2) - (xsep1+2xSep2)/(slope1)^(-1)
    #       when sng(slope1)!=sgn(slope2) -
    #                                           0
    # need to deal with end-point slopes:
    def pchipEndPoints(xSep1, xSep2, slope1, slope2):
        # need to treat end points separately for contiuity of the polynomial
        # xSep1 = (Xi+1 - Xi) or (Xi - Xi-1) for initial and final end points
        # slope1 = (Yi+1 - Yi)/(Xi+1 - Xi) or (Yi - Yi-1)/(Xi - Xi-1) for initial and final end points

        slopeNumerator = 2 * (xSep1 + xSep2) * slope1 - xSep1 * slope2
        slope = slopeNumerator / ( xSep1 + xSep2 )

        # apply piecwise Fritsch-Carlson monotoniciy condition:
        if np.sign(slope) != np.sign(slope1):
            slope = 0
        elif np.sign(slope1) != np.sign(slope2) and np.absolute(slope) > np.absolute(3*slope1):
            slope = 3*slope1

        # return final slope result:
        return slope

    # compute the slopes of the interior points according to a 3-point modified version of the 
    # Fritsch-Carlson method highlighted above.
    # slope = (Yi+1 - Yi)/(Xi+1 - Xi) (estimated from delta(Y)/delta(X) in initial data
    # xSep = (Xi+1 - Xi)
    numPts = xSep.shape[0]
    slopeFinal = np.zeros(numPts+1)
    # find indices where the estimated slope is non-zero:
    xSepRed = xSep[1:numPts]
    slopeRed = slope[1:numPts]
    signChange = np.zeros(slopeRed.shape)
    indexArray = np.arange(signChange.shape[0])
    signChange[1:] = np.diff( np.sign(slopeRed) )
    # figure out the discontinuities and turning points in the slope vector 
    disConIndices = np.where(signChange != 0)[0]
    zeroIndices = np.where(slopeRed == 0)[0]
    # remove any repeats and collect the bad indices
    badIndices = np.unique( np.hstack((disConIndices, zeroIndices)) )
    #print("badIndices: {}".format(badIndices))
    # remove any of the badIndices from the total list of indices
    goodIndices = np.delete(indexArray, badIndices, 0)

    for indx in goodIndices:
        weight1 = 2*xSepRed[indx] + xSepRed[indx-1]
        weight2 = xSepRed[indx] + 2*xSepRed[indx-1]
        # calculate slope based on formula
        slopeFinal[indx] = (weight1 + weight2) / (weight1/slope[indx-1] - weight2/slope[indx])

    # calculate the slopes at the endpoints:
    slopeFinal[0] = pchipEndPoints(xSep[0], xSep[1], slope[0], slope[1])
    slopeFinal[-1] = pchipEndPoints(xSep[-1], xSep[-2], slope[-1], slope[-2])

    return slopeFinal

def pchip(x, y, xi):
    print("!!WARNING!! pchipInterp is deprecated as of 01/26/18. Please use pchipInter_SciPy instead.")
    # x is a vector of depent variable points for some function f
    # y is a vector of the corresponding function values, f(x)
    # xi are the points of interest where you want to evaluate the resultant interpolating function
    # returns a vector of values for the Piecewise Cubic Polynomial Hermite Interpolating Polynomial, Pi(X)
    # for the data (X, Y) evaluate at the points of interest, xi
    # approximate the first derivatives and pass them to pchipSlopes for refinement:
    numPts = x.shape[0]
    xSep = x[1:] - x[:-1]                               # if n=length(x) then length(xSep) = n-1
    slopeInit = np.divide( (y[:1] - y[:-1]), xSep )     # same length as xSep (n-1)
    slopeFinal = pchipSlopes( xSep, slopeInit)          # same length as x (n)

    #print("shape check - (x, y, xi, slopeFinal): \n{}, {}, {}, {}".format(x.shape, y.shape, 
    #    xi.shape, slopeFinal.shape))

    #print("slopeInit: {}, slope1: {}, slope2: {}, xSep: {}, slopeFinal: {}".format(slopeInit.shape, 
    #    slopeFinal[0:numPts-1].shape, slopeFinal[1:numPts+1].shape, xSep.shape, slopeFinal.shape))
    #
    # calculate the piecewise polynomial coefficients for the polynomial of the following form
    # for x in [Xi, Xi+1]:    Pi(X) = Yi + Yi'(X-Xi) + Ci(X-Xi)^2 + Di(X-Xi)^3
    #print("calculation shape check: (slopeInit, slopeFinal[0:-1], "+
    #        "slopeFinal[1:]): \n{}, {}, {}".format(slopeInit.shape, 
    #            slopeFinal[0:-1].shape, slopeFinal[1:].shape))
    Ci = np.divide( 3*slopeInit - 2*slopeFinal[0:-1] - slopeFinal[1:], xSep )
    Di = np.divide( slopeFinal[0:-1] - 2*slopeInit + slopeFinal[1:], np.square(xSep) )

    # find the subinterval indices, kk, where x(kk) <= xi < x(kk+1)
    # let XminusXi stand for (X - Xi)
    # evalData will be the vector of 
    evalData = np.zeros(xi.shape[0])
    tmpArray = np.zeros(x.shape[0])
    for kk, point in enumerate(xi):
        tmpArray = x / point 
        primeIndx = np.argmax((tmpArray > 1))-1

        XminusXi = point - x[primeIndx]
        evalData[kk] = y[primeIndx] + XminusXi * (slopeFinal[primeIndx] + 
                XminusXi * (Ci[primeIndx] + XminusXi*Di[primeIndx]))

    # return the evaluated Pi(X) at the points in the Xi vector
    return evalData

"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""
