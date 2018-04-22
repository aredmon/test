"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*   Module Name: ModMunkres                                                                                 *
*   Author(s): Mark Lambrecht, Brent McCoy                                                                  *
*   Version: 1.0                                                                                            *
*   Date: 02/5/18                                                                                           *
*                                                                                                           *
*       Module          given the matrix M(m,n) of real numbers, find a permutation                         *
*       description:    perm(m) of the integers 1,2,3 ... mrows that minimizes                              *
*                       sum( M(m,perm(m)) ).                                                                *
*                                                                                                           *
*       Inputs:         source matrix: sourceMat   - MxN cost matrix                                        *
*                       output mask  : mask        - optional dtype argument that returns an MxN            *
*                                                    'boolean' matrix of the task assignments               *
*                                                                                                           *
*       Outputs:        vector permutation: finalPerm                                                       *
*                       permutation sum:    permSum                                                         *
*                                                                                                           *
*       References:     Bourgeois, F. and Lassalle, J.C., An Extension of the Munkres                       *
*                       Algorithm for the Assignment Problem to Rectangular Matrices,                       *
*                       Comm. ACM, Vol. 14, Number 12, (Dec 1971), 802-4.                                   *
*                                                                                                           *
*************************************************************************************************************
"""
import numpy as np

def buildStarsAndPrimes(sourceMat):
    # now lets do some fancy matrix math!
    # find the zeros of the matrix:
    rowZeros, colZeros = np.where(sourceMat == 0)
    rowUniques = np.unique(rowZeros)
    combinedZeros = np.asarray( zip(rowZeros, colZeros) )
    starsCon = np.zeros(rowZeros.size, dtype=bool)
    #------------------------- STEP 2 & STEP 3 --------------------------
    coveredCols = []
    coveredRows = []
    for val in rowUniques:
        subArray = np.where(combinedZeros[:,0] == val)[0]
        shortList = combinedZeros[subArray][:,1]
        for column in shortList:
            if np.in1d(column, coveredCols).any():
                # no good, skip to the next column
                continue
            else:
                subMatLow = ( (combinedZeros[:,0] == val) & (combinedZeros[:,1] == column) )
                coveredCols.append( int(column) )
                starsCon = np.logical_or(starsCon, subMatLow)
                break
    # do some minor array manipulation for clarity
    stars = combinedZeros[starsCon]                             # STEP 2
    primesCandidate = combinedZeros[~starsCon]                  # STEP 3
    primesConfirmed = []
    for index, pair in enumerate(primesCandidate):
        if np.in1d(int(pair[0]), coveredRows).any() or np.in1d(int(pair[1]), coveredCols).any():
            # this zero is already covered
            continue
        else:
            primesConfirmed.append( int(index) )
            coveredRows.append( int(pair[0]) )
            openRow = np.where(stars[:,0] == pair[0])[0]
            try:
                coveredCols.remove(stars[openRow[0]][1])
            except:
                # the column was not covered initially anyway
                continue
            
    # get final list of primes (uncovered zeros)
    try:
        primes = primesCandidate[primesConfirmed]
    except IndexError:
        primes = np.asarray([])
    
    # return the new stuff:
#     return {"stars": stars,"primes": primes,"combinedZeros": combinedZeros,
#             "coveredRows": coveredRows,"coveredCols": coveredCols}
    return stars, primes, {"coveredRows": coveredRows,"coveredCols": coveredCols}

def refactorPreliminaries(sourceMat, coveredRows, coveredCols):
    #------------------------- STEP 5 --------------------------
    originalMatrix = sourceMat.copy()
    adjustedMatrix = sourceMat.copy()
    # figure our what is uncovered
    unCoveredRows = np.setxor1d( coveredRows, np.arange(sourceMat.shape[0]) ).astype(int)
    unCoveredCols = np.setxor1d( coveredCols, np.arange(sourceMat.shape[1]) ).astype(int)
    # calculate what is left
    reducedMat = np.delete(adjustedMatrix, coveredRows, axis=0)
    reducedMat = np.delete(reducedMat, coveredCols, axis=1)
    if reducedMat.size == 0 or unCoveredCols.size == 0:
        pass
    else:
        # find the smallest remaining value (should be non-zero)
        absMin = np.amin(reducedMat)
        # adjust the preliminary matrix with the new information
        adjustedMatrix[coveredRows, :] = adjustedMatrix[coveredRows, :] + absMin
        try:
            adjustedMatrix[:, unCoveredCols] = adjustedMatrix[:, unCoveredCols] - absMin 
        except:
            print("unCoveredRows: {}".format(unCoveredRows))
            print("unCoveredCols: {}".format(unCoveredCols))
    # return the adjusted matrix and original matrix:
    return adjustedMatrix, originalMatrix

def calculateFinalResult(stars, sourceMat):
    # this will exectute when the stars matrix finally satisfies the desired min condition
    try:
        finalRows = stars[:,0]
        finalCols = stars[:,1]
    except:
        print("stars array is flat: {}".format(stars))
        finalRows = np.arange(sourceMat.shape[0])
        finalCols = np.arange(sourceMat.shape[1])
    permSum = 0.0
    for pair in stars:
        permSum += sourceMat[pair[0]][pair[1]]

    # return the results
    #return finalPerm, permSum
    return finalRows, finalCols, permSum

def ModMunkres2(sourceMat, mask=None):
    prelimMat = sourceMat.copy()
    incomplete = True
    flip = False
    maxCount = 100
    iteration = 0

    numRows, numCols = sourceMat.shape
    finalPerm = []
    colRange = np.arange( min(numRows, numCols) )
    permSum = 0.0

    if numRows >= numCols:
        minval = np.amin(prelimMat, 0)
        prelimMat = prelimMat - minval
    # NOTE: both if statements will execute in the event numCols = numRows
    if numRows <= numCols:
        minval = np.reshape( np.apply_along_axis(np.amin, 1, prelimMat), (numRows, 1) )
        prelimMat = prelimMat - minval
        if numRows < numCols:
            perlimMat = np.transpose(prelimMat)
            flip = True
    # now lets do some fancy matrix math!
    stars, primes, coverings = buildStarsAndPrimes(prelimMat)
    try:
        starsUnique = np.unique(stars[:,1])
    except IndexError:
        starsUnique = np.asarray([])
    # check for 0 matrix and run otherwise
    if np.array_equal(sourceMat, np.zeros_like(sourceMat)):
        print("zero matrix encountered, default permutation returned")
        finalRows = np.arange( numRows )
        finalCols = np.arange( numCols )
        #finalPerm = np.arange( numCols )
        permSum = 0.0
    elif np.array_equal(starsUnique, colRange):
        #finalPerm, permSum = calculateFinalResult(stars, sourceMat)
        finalRows, finalCols, permSum = calculateFinalResult(stars, sourceMat)
    else:
        while incomplete:     # STEP 2 check if complete
            prelimMat, originalMat = refactorPreliminaries(prelimMat, **coverings)

            stars, primes, coverings = buildStarsAndPrimes(prelimMat)
            try:
                starsUnique = np.unique(stars[:,1])
            except IndexError:
                starsUnique = np.asarray([])

            if np.array_equal(starsUnique, colRange) or iteration == maxCount:
                #finalPerm, permSum = calculateFinalResult(stars, sourceMat)
                finalRows, finalCols, permSum = calculateFinalResult(stars, sourceMat)
                if finalRows.size != colRange.size:
                    finalRows = np.arange(colRange.size)
                    finalCols = finalRows.copy()
                    permSum = 1000
                incomplete = False
            #elif iteration == maxCount:
            #    print("max iterations ({}) reached".format(maxCount))
            #    finalPerm, permSum = calculateFinalResult(stars, sourceMat)
            #    incomplete = False
            else:
                # one moe again
                iteration += 1
                continue
    if mask != None:
        outMat = np.zeros_like(sourceMat)
        for pair in zip(finalRows, finalCols):
            rowIndx = pair[0]
            colIndx = pair[1]
            outMat[rowIndx, colIndx] = 1
        # release the results
        return finalRows, finalCols, permSum, outMat.astype(mask)
    else:
        # release the results
        return finalRows, finalCols, permSum

"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""
