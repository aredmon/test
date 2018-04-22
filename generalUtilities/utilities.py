"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*   Module Name: utilities.py                                                                               *
*   Author(s): Brent McCoy                                                                                  *
*   Version: 1.0                                                                                            *
*   Date: 12/20/17                                                                                          *
*                                                                                                           *
*       Module Description:                                                                                 *
*           Python adaptation of the necessary Utils functions defined originally in MatLab. Each function  *
*           is callable and all dependencies are either defined here or explicitly pointed to in the code.  *
*************************************************************************************************************
"""
import numpy as np
import math
from generalUtilities.config import *
from generalUtilities.bhattacharyya import bhattacharyya
from generalUtilities.GaussProblem import GaussProblem
from generalUtilities.ModMunkres import ModMunkres2

"""
----------------------------------------------------------------------------------------------
    Inputs:     vector      -   1 dimensional numpy array that you want to reverse
                indices     -   1 dimensional numpy array with 1 or 2 entries
                direction   -   'head' or 'tail' conditional option to be used when
                                subSet.size == 1, this flag will control whether you
                                want to flip all entries up to the stop index ('head')
                                or flip all entris after the stop index ('tail')

    Outputs:    rotcev      -   as the name might imply this is a mirror image of the
                                original vector input 
                                example: (vector -> rotcev)
----------------------------------------------------------------------------------------------
"""
def flipVector(vector, indices=np.array([]), direction='head'):
    # handle mis-typed indices argument
    subSet = np.asarray(indices, dtype=int)
    # apply vector transformation
    if subSet.size > 1:
        sortedSubSet = np.squeeze( np.sort(subSet) )
        stop1 = sortedSubSet[0]
        stop2 = sortedSubSet[1]
        # flip the section between stop1 and stop2 and stitch back together
        rotcev = np.concatenate( 
                (vector[:stop1], vector[stop1:stop2][::-1], vector[stop2:]) 
                )
    
    elif subSet.size == 1:
        stop = np.asscalar(subSet)
        # check which part of the vector to flip
        if direction == 'head':
            # flip the part of the vector up to stop
            rotcev = np.concatenate( 
                    (vector[:stop][::-1], vector[stop:]) 
                    )
        elif direction == 'tail':
            # flip the part of the vector after stop
            rotcev = np.concatenate( 
                    (vector[:stop], vector[stop:][::-1]) 
                    )
        else:
            raise ValueError("invalid direction flag detected, please specify 'head' or 'tail'")
   
    else:
        rotcev = vector[::-1]
    
    # return the transformed vector
    return rotcev

"""
----------------------------------------------------------------------------------------------
    Inputs:     vector      -   1 dimensional numpy array that you want to reverse
                indices     -   1 dimensional numpy array with 1 or 2 entries

    Outputs:    vocter      -   as the name might imply this is a copy of the original vector
                                where two of the elements have been swapped
                                example: (vector -> vocter) swapping the 2nd and 5th element
----------------------------------------------------------------------------------------------
"""
def swapElements(vector, indices=np.array([])):
    # handle mis-typed indices agument
    subSet = np.squeeze( np.sort(np.asarray(indices, dtype=int)) )
    if subSet.size == 2:
        pos1 = subSet[0]
        pos2 = subSet[1]
        
        vocter = np.concatenate( 
                (vector[:pos1], [vector[pos2]], vector[pos1+1:pos2], 
                    [vector[pos1]], vector[pos2+1:]) 
                )
    else:
        # by default swapping just first and last element
        vocter = np.concatenate( 
                ([vector[-1]], vector[1:-1], [vector[0]]) 
                )

    # return transformed vector
    return vocter

"""
----------------------------------------------------------------------------------------------
    Inputs:     vector      -   1 dimensional numpy array that you want to reverse
                indices     -   1 dimensional numpy array with 1 or 2 entries
                shift       -   positive or negative integer indicating how many places and 
                                in which direction vector should be rolled
                                   - shift < 0 means shift left amount = |shift|
                                   - shift > 0 means shift right amount = |shift|

    Outputs:    vercto      -   as the name might imply this is a copy of the original vector
                                where the area before, after, or between the indices has been
                                rolled by an amount specified by the 'shift' input
                                example: (vector -> vercto) pos 1 shift starting at 3rd elem
----------------------------------------------------------------------------------------------
"""
def slideSet(vector, indices=np.array([]), shift=1):
    # handle mis-typed indices argument
    subSet = np.squeeze( np.sort(np.asarray(indices, dtype=int)) )
    shift = int( shift )
    if subSet.size == 2:
        stop1 = subSet[0]
        stop2 = subSet[1]

        vercto = np.concatenate( 
                (vector[:stop1], np.roll(vector[stop1:stop2], shift), vector[stop2:]) 
                )

    elif subSet.size == 1:
        stop = np.asscalar(subSet)
        vercto = np.concatenate( 
                (np.roll(vector[:stop], shift), np.roll(vector[stop:], shift)) 
                )

    else:
        vercto = np.roll( vector, shift )

    # return transformed vector, vercto
    return vercto

"""
----------------------------------------------------------------------------------------------
    Inputs:     array       -   numpy array you wish to take the norm of
                axis        -   optional argument to take norm along a particular axis 
                                of the array

    Outputs:    norm        -   norm of output, either a scalar (if axis was not specified or 
                                default of none was applied) or 1xN vector of norms along 
                                desired axis
----------------------------------------------------------------------------------------------
"""
def vecNorm(array, axis=None):
    if axis != None:
        norm = np.apply_along_axis(np.linalg.norm, axis, array)
    else:
        norm = np.linalg.norm(array)    # scalar result that is the frobenius norm of entire
                                        # array (default np.linalg.norm behavior)
    return norm

"""
----------------------------------------------------------------------------------------------
    Inputs:     arrays      -   n >= 2 1 dimensional array (must all have the same length)

    Outputs:    new_object  -   concatenated array along associated dimension
    
    axis:       invert      -   flag that indicates vertical or horizontal zipping
                            -   0 = vertical (default)  array elements are treated as rows
                            -   1 = horizontal  array elements are treated as columns
----------------------------------------------------------------------------------------------
"""
def zipArray(arrays, axis=0):
    if isinstance(axis, int):
        try:
            shapes = [array.shape[0] for array in arrays]
            ndims  = [array.ndim for array in arrays]
            # zip will by default only combine up to the shortest array
            # currently this is unwanted so the ValueError exception is thrown
            if len(set(shapes)) > 1 or not np.array_equal(ndims, np.ones_like(ndims)):
                if not len(set(shapes)) > 1:
                    raise ValueError("all arrays must be 1 dimensional: {}".format(ndims))
                else:
                    raise ValueError("all arrays must have the same length: {}".format(shapes))
            if axis == 0:
                combinedArray = np.asarray( zip(*arrays) )
            else:
                combinedArray = np.squeeze( np.asarray( zip(arrays) ) )
            return combinedArray

        except IndexError:
            print("'arrays' argument must be a tuple or a list")
            return {"arrays": locals()["arrays"], "axis": locals()["axis"]}      
    else:
        raise TypeError("axis argument must be of type integer")

"""
----------------------------------------------------------------------------------------------
    Inputs:     angle       -   angle in radians
                objects     -   nx3 matrix object positions to be yaw rotated

    Outputs:    new_object  -   nx3 matrix of rotated position vectors for new object
    
    IvertFlag:  invert      -   boolean flag that computes the inverse yaw rotation
----------------------------------------------------------------------------------------------
"""
def yaw_scene(angle, objects, invert=False):
    # Perform yaw rotation on the object scene
    # looking from negative Z to the origin, the rotation is counter-clockwise
    # aka the positive x-axis rotates toward negative y for positive rotation angles
    def yaw(angle):
        yaw_matrix = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle),  np.cos(angle), 0],
                [0, 0, 1]])
        return yaw_matrix

    if invert:
        new_object = np.dot(objects, np.transpose(yaw(angle)))
    else:
        new_object = np.transpose( np.dot( yaw(angle), np.transpose(objects) ) )

    return new_object

"""
----------------------------------------------------------------------------------------------
    Inputs:     angle       -   angle in radians
                objects     -   nx3 matrix object positions to be pitch rotated

    Outputs:    new_object  -   nx3 matrix of rotated position vectors for new object
----------------------------------------------------------------------------------------------
"""
def pitch_scene(angle, objects, invert=False):
    # Perform pitch rotation on the object scene
    # looking from positive Y to the origin, the rotation is counter-clockwise
    # aka the positive x-axis rotates toward negative z for positive rotation angles
    def pitch(angle):
        pitch_matrix = np.array([
                [ np.cos(angle), 0, np.sin(angle)],
                [ 0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]])
        return pitch_matrix

    if invert:
        new_object = np.dot(objects, np.transpose(pitch(angle)))
    else:
        new_object = np.transpose( np.dot( pitch(angle), np.transpose(objects) ) )

    return new_object

"""
----------------------------------------------------------------------------------------------
    Inputs:     array       -   NxM array of values
                axis        -   option axis agument finds mode of each vector along that axis

    Outputs:    mode        -   most common value, either an L or LxAxis.size array of modes
                freq        -   number of times the associated mode was found in the 
                                vector/array
                mult        -   list of all modes along designated axis, may not be different
                                from mode
----------------------------------------------------------------------------------------------
"""
def findMode(array, axis=0):
    values = np.unique(array)
    multiplicity = []   # mult array

    # handle vector cases or single modes for an entire array
    if array.ndim == 1 or axis == None:
        #do stuff for single vector mode
        result = np.zeros((values.size, 2),dtype=int)
        result[:,0] = values.transpose()
        for indx, val in enumerate(values):
            count = np.where(array == val)[0].size
            result[indx,1] = count
        # construct the multiplicity list, find mode and frequency
        index = np.argmax(result[:,1])
        mode = np.array([values[index]])
        frequency = np.array([result[index,1]])
        indices = np.where(result[:,1] == result[index, 1])[0]
        multiplicity.append(values[indices])

    # handle axis specific modes for a multi-dimensional array
    else:
        mode = np.zeros(array.shape[axis], dtype=int)
        frequency = np.zeros(array.shape[axis], dtype=int)
        # handle axis option
        if axis == 0:
            array = array.transpose()   # original construction is column-wise mode

        # create storage array for number of possible values in array and their count
        result = np.zeros((values.size, array.shape[1]+1), dtype=int)
        result[:,0] = values.transpose()
        # fill result matrix with counts for each number found in each column
        for col in range(array.shape[1]):
            for indx, val in enumerate(values):
                count = np.where(array[:, col] == val)[0].size
                result[indx, col+1] = count
        # construct the multiplicity list, mode array and frequencey array
        for col in range(1,result.shape[1]):
            indx = np.argmax(result[:,col])
            mode[col-1] = values[indx]
            indices = np.where(result[:,col] == result[indx, col])[0]
            multiplicity.append(values[indices])
            frequency[col-1] = np.where(array[:, col-1] == mode[col-1])[0].size

    # return the discovered values
    return mode, frequency, multiplicity

"""
----------------------------------------------------------------------------------------------
     Inputs:    x1,x2		= sets of data, should both be Nx2

     Outputs:   bias		= bias between two sets, should be 1x2
----------------------------------------------------------------------------------------------
"""
def removeBias(x1, x2, option=1):
    #x1 = x1[ ( (~np.isnan(x1)) & (~np.isinf(x1)) ) ]
    #x2 = x2[ ( (~np.isnan(x2)) & (~np.isinf(x2)) ) ]

    if option == 1:
        x1Mean = np.mean(x1, axis=0)
        x2Mean = np.mean(x2, axis=0)
        bias = x1Mean - x2Mean

    elif option == 2 or option == 3:
        x1Med = np.median(x1, axis=0)
        x2Med = np.median(x2, axis=0)
        bias = x2Med - x2Med

        if option == 3:
            spread1 = np.std(x1, axis=0)
            spread2 = np.std(x2, axis=0)
            limit = vecNorm(spread1) + vecNorm(spread2)
            if vecNorm(bias) < limit:
                bias = np.zeros(2)

    elif option == 5:
        bias = x2[1,:] - x1[1,:]

    elif option == 6:
        x1Len = x1.shape[0]
        x2Len = x2.shape[0]
        for row in range(x1Len):
            try: 
                dX = np.vstack( (dX, np.subtract(x2, x1[row])) )
            except:
                dX = np.subtract(x2, x1[row])
        # compute bias
        bias = np.median(dX, axis=0)
    else:
        bias = np.zeros(2)
    #return result:
    return bias

"""
----------------------------------------------------------------------------------------------
     Inputs:    x1,x2		= sets of data, should be Nx2 and Mx2 with covariance
                sig1, sig2      = covariance of selected data sets (x1 and x2)

     Outputs:   bias		= bias between two sets, should be 1x2
----------------------------------------------------------------------------------------------
"""
def removeBias2(x1, x2, sig1, sig2):
    #x1 = x1[ ( (~np.isnan(x1)) & (~np.isinf(x1)) ) ]
    #x2 = x2[ ( (~np.isnan(x2)) & (~np.isinf(x2)) ) ]
    print("shape of arrays we want to remove bias from: {}, {}".format(x1.shape, x2.shape))
    finalCost = np.infty
    testBias = []
    for ii in range(x1.shape[0]):
        for jj in range(x2.shape[0]):
            tmpBias = x1[ii] - x2[jj]
            xAdjusted = np.subtract( x1, tmpBias )
            # xAdjusted has the following three limiting cases:
            #   if x1[jj] << x2[ii] then xAdjusted ~ x2
            #   if x1[ii] >> x2[jj] then xAdjusted ~ 0
            #   if x1[ii] ~= x2[jj] then xAdjusted ~ x1
            # now we need to apply a similar adjustment to the covariance matrix
            # could potentially calcualte new estimated cov, based on new data
            # currently we are using a weighted average method which seems to make sense
            # given the limiting cases listed above
            if np.linalg.norm( x1[ii] + x2[jj] ) == 0:
                sig1Mod = np.zeros_like(sig1)
            else:
                sig1Mod = (sig1[ii]*x1[ii] + sig2[jj]*x2[jj]) / (np.linalg.norm(x1[ii] + x2[jj]))
            sig1Adj = np.add(sig1, sig1Mod)
            #print("sig1Adj: \n{} \nsig2Adj: \n{}".format(sig1Adj, sig2))
            bhattaDist, _ = bhattacharyya(xAdjusted, x2, sig1Adj, sig2)
            #print("bhattaDist shape: {}".format(bhattaDist))
            # the cost, or min sum, is now built in as a returnable from ModMunkres2
            _, assignMent, cost = ModMunkres2(bhattaDist)
            if assignMent.size > 0:
                avgCost = cost / assignMent.size
            else:
                avgCost = np.infty
            # check final output
            if avgCost < finalCost:
                finalCost = avgCost
                finalI = ii
                finalJ = jj
                testBias.append( x1[finalI] - x2[finalJ] )
    # create final Bias measurement
    try:
        finalBias = x1[finalI] - x2[finalJ]
    except NameError:
        finalBias = np.zeros(2)
    # return final result
    return finalBias, testBias

"""
----------------------------------------------------------------------------------------------
    Inputs:       low  - (Scalar) low range of random number
                  high - (Scalar) high range of random number

    Outputs:      val - (Scalar) uniform random deviate between [low, high]
----------------------------------------------------------------------------------------------
"""
def UniformRandRange(low, high):
    val = low + (high - low) * np.random.rand(1)
    return val

"""
----------------------------------------------------------------------------------------------
    Inputs:       vecA - First vector (vecA(:, 1:3))
                  vecB - Second vector (vecB(:, 1:3))
 
    Outputs:      theta - Angle between vectors (theta(:, 1)) [rad]
----------------------------------------------------------------------------------------------
"""
def AngleBetweenVecs(vecA, vecB):
    AdotB = np.dot(vecA, vecB)
    magA = vecNorm(vecA)
    magB = vecNorm(vecB)

    theta = math.acos( AdotB / (magA*magB) )
    return theta

"""
----------------------------------------------------------------------------------------------
    Inputs:       trajectory    - All points on the trajectory (pos, vel, t) (Nx7)
                  time          - Time(s) at which to interpolate a point (Mx1)
    
    Outputs:      trajTime      - Interpolated trajectory (pos, vel, t) (Mx7)
----------------------------------------------------------------------------------------------
"""
try:
    from scipy.interpolate import pchip_interpolate as pchip
    #print("using SciPy pchip_interpolate. Resultant matrix size: {}".format(trajTime.shape))

except ImportError:
    from generalUtilities.pchipInterp_SciPy import pchip
    #print("using in house pchip routine. Resultant matrix size: {}".format(trajTime.shape))

# actual code call using one of the above pchip interpolation routines
def InterpTraj(trajectory, time):
    #print("running pchipInterploation")
    columns = trajectory.shape[1]
    trajTime = pchip(trajectory[:, 6], trajectory[:,0:columns], time)
    return trajTime

"""
----------------------------------------------------------------------------------------------
    Inputs:       center - 1x3 position of ellipsoid center, ECI, m
                  rad    - (Scalar) Radius of ellipsoid, m
                  color  - 1x3 [r g b] color
                  aH     - axis handle for plotting
    
    Outputs:      N/A
----------------------------------------------------------------------------------------------
"""
#def ShowBubble(center, rad, color=np.array([0.5, 0.5, 0.5]), aH):
#    print("module currently under development")
#    # plot an ellipse
#    # [a,b,c] = ellipsoid(center(1), center(2), center(3), rad, rad, rad);
#    # surf(aH, a, b, c, 'edgecolor', 'none', 'facealpha', .05, 'facecolor', color);
#    # return;

"""
----------------------------------------------------------------------------------------------
    Inputs:     alt             - (Scalar) altitude, meters
    
    Outputs:    rho             - (Scalar) atmospheric density, kg/m^3

    Requires:   density_data    - density data table loaded from config.py
----------------------------------------------------------------------------------------------
"""
def AtmosphereQ(alt):
    delta_alt = 1000.0
    nAtmos = len(density_data)

    q = alt/delta_alt
    II = 1 + int( math.floor(q) )
    f = q - (II-1)

    if II < 1:
        rho = density_data[0]
    elif II >= nAtmos:
        rho = 0.0
    else:
        rho = (1.0 - f) * density_data[II] + f * density_data[II+1]

    return rho

"""
----------------------------------------------------------------------------------------------
    Inputs:       X     - (Scalar) ECEF (or ECI) X coordinate, m
                  Y     - (Scalar) ECEF (or ECI) Y coordinate, m
                  Z     - (Scalar) ECEF (or ECI) Z coordinate, m
    
    Outputs:      alt   - (Scalar) altitude in meters above the ellipsoid, m
    
    Requires:     eParms    - earthParameters.json data loaded from config.py
----------------------------------------------------------------------------------------------
"""
def EarthAltitude(X, Y, Z):
    Re = eParms.R_EARTHa
    e = eParms.eccearth

    if X == 0 and Y == 0 and Z == 0:
        alt = -5000000
    else:
        r = math.sqrt(X*X + Y*Y)
        phi = math.atan(Z/r)
        diff = 9999
        loopcnt = 0
        maxloopcnt = 50
        while diff > 0.00001 and loopcnt < maxloopcnt:
            phi1 = phi
            C = 1 / math.sqrt( 1 - (e*math.sin(phi1))**2 )
            phi = math.atan( Z + Re*C*e*e*math.sin(phi1)/r )
            diff = math.fabs(phi1 - phi)
            loopcnt += 1
        alt = r / math.cos(phi) - Re*C
    
    return alt

"""
----------------------------------------------------------------------------------------------
    Inputs:       p - input position [x y z] in meters
    
    Outputs:      g - acceleration due to gravity at the position p [gx gy gz] in
                      m/s^2 (Scalar)
    
    Requires:     eParms    - earthParameters.json data loaded from config.py
----------------------------------------------------------------------------------------------
"""
def Gravity(p):
    mu = eParms.MU

    # Earth Zonal Harmonics
    J = np.array([0.0, 0.108263e-2, -0.2532e-5, -0.1611e-5, -0.23578564879393e-6])
    
    # Earth Equatorial Radius
    Re = eParms.R_EARTHa
    
    x = p[0]
    y = p[1]
    z = p[2]
    r = vecNorm(p)
    
    if r != 0:
        r_inv = 1.0 / r
    else:
        r_inv = 1.0
    
    Redr = Re * r_inv 
    Redr_pow2 = Redr * Redr 
    Redr_pow3 = Redr_pow2 * Redr 
    Redr_pow4 = Redr_pow3 * Redr 
    
    
    zdr = z * r_inv 
    zdr_pow2 = zdr * zdr 
    zdr_pow3 = zdr_pow2 * zdr 
    zdr_pow4 = zdr_pow3 * zdr 
    
    # Terms for x and y
    xJ2 = J[1] *   1.5*Redr_pow2 * (5*zdr_pow2 - 1) 
    xJ3 = J[2] *   2.5*Redr_pow3 * (3*zdr - 7*zdr_pow3) 
    xJ4 = J[3] * 0.625*Redr_pow4 * (3 - 42*zdr_pow2 + 63*zdr_pow4) 
    
    # Term for z
    if z == 0:
        zJ2 = 0 
        zJ3 = 0 
        zJ4 = 0 
    else:
        zJ2 = J[1] *   1.5*Redr_pow2 * (3 - 5*zdr_pow2) 
        zJ3 = J[2] *   1.5*Redr_pow3 * (10*zdr - 11.66667*zdr_pow3 - r/z) 
        zJ4 = J[3] * 0.625*Redr_pow4 * (15 - 70*zdr_pow2 + 63*zdr_pow4) 
    
    g = np.zeros([3]) 
    
    # g[1] = -mu * x/r^3 * (1 - xJ2 + xJ3 - xJ4) 
    # g[2] = -mu * y/r^3 * (1 - xJ2 + xJ3 - xJ4) 
    # g[3] = -mu * z/r^3 * (1 + zJ2 + zJ3 - zJ4) 
    
    mmudr_pow3 = -mu * r_inv * r_inv * r_inv 
    CXY = mmudr_pow3 * (1 - xJ2 + xJ3 - xJ4) 
    
    g[0] = CXY * x 
    g[1] = CXY * y 
    g[2] = mmudr_pow3 * z * (1 + zJ2 + zJ3 - zJ4) 
    
    return g

"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""
