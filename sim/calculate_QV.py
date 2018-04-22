"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*   Module Name: Calculate_QV                                                                               *
*   Author(s): Mark Lambrecht, Brent McCoy                                                                  *
*   Version: 1.0                                                                                            *
*   Date: 01/25/18                                                                                          *
*                                                                                                           *
*************************************************************************************************************
"""
import numpy as np
import importModules as mods

def calculate_qv_single_asset2(algOption, biasMethod, rfObjects, irObjects, sigmaRF, sigmaIR, gatingK, 
        numRF, numIR, rfIndex, irIndex):
    # rfObjects and irObjects are transposed relative to the rest of the program
    # rfObjects is a 2xN set of az/el column vectors
    # irObjects is a 2xN set of az/el column vectors
    BIG_L = mods.TOM_SAPS.BIG_L

    # Need to filter out object array if they have empty values at the end
    rfObjs = rfObjects[0:numRF, :]
    irObjs = irObjects[0:numIR, :]
    rfUnBiased = np.zeros((numRF, 2))
    finalNumRF = 0

    # algorithm execution options:
    if algOption == 1:
        bias = mods.removeBias(rfObjs, irObjs, biasMethod) 
        #print("option 1 bias: {}".format(bias))
        rfUnBiased[:, 0] = rfObjs[:, 0] - bias[0]
        rfUnBiased[:, 1] = rfObjs[:, 1] - bias[1]

        mahalDist, _ = mods.mahalDist1( rfUnBiased, irObjs, sigmaRF, sigmaIR )
        rfPerm, irPerm, overallScore = mods.ModMunkres2(mahalDist)

        #print("initial RF Indices: {}".format(rfIndex))
        #print("initial IR Indices: {}".format(irIndex))
        finalRFIndex = rfIndex[rfPerm].astype(int)
        finalIRIndex = irIndex[irPerm].astype(int)
        # determine lethalObject ID
        #print("finalRFIndex, finalIRIndex: {}, {}".format(finalRFIndex, finalIRIndex))
        #irObjectIDSet = finalIRIndex[ finalRFIndex == irIndex[0] ]
        #if irObjectIDSet.size > 0:
        #    lethalObjectNum = irObjectIDSet[0]
        #else:
        #    lethalObjectNum = 0
        # create object pair tree
        pairedObjects = zip( finalRFIndex, finalIRIndex )
        # determine lethalObject ID
        pairedObjectsLocation = zip(rfPerm, irPerm)
        pairScore = np.zeros(len(pairedObjectsLocation))
        pairIndex = 0
        for pair in pairedObjectsLocation:
            rfId = pair[0]
            irId = pair[1]
            pairScore[pairIndex] = mahalDist[rfId][irId]
            pairIndex += 1
        finalNumRF = numRF

    elif algOption == 2:
        bias = mods.removeBias(rfObjs, irObjs, biasMethod) 
        #print("option 2 bias: {}".format(bias))
        rfUnBiased[:, 0] = rfObjs[0] - bias[0]
        rfUnBiased[:, 1] = rfObjs[1] - bias[1]

        mahalDist, _ = mods.mahalDist3( rfUnBiased, irObjs, sigmaRF, sigmaIR , gatingK)
        rfPerm, irPerm, overallScore = mods.ModMunkres2(mahalDist)

        #print("initial RF Indices: {}".format(rfIndex))
        #print("initial IR Indices: {}".format(irIndex))
        finalRFIndex = rfIndex[rfPerm].astype(int)
        finalIRIndex = irIndex[irPerm].astype(int)
        # Now break any associations that are too far:
        keepCondition = ( (irPerm > 0) & (irPerm < BIG_L) )
        finalIRIndex = finalIRIndex[ keepCondition ].astype(int)
        # end of association trimming
        #irObjectIDSet = finalIRIndex[ finalRFIndex == irIndex[0] ]
        #if irObjectIDSet.size > 0:
        #    lethalObjectNum = irObjectIDSet[0]
        #else:
        #    lethalObjectNum = 0
        # create object pair tree
        pairedObjects = zip( finalRFIndex, finalIRIndex )
        pairedObjectsLocation = zip(rfPerm, irPerm)
        # determine lethalObject ID
        pairScore = np.zeros(len(pairedObjects))
        pairIndex = 0
        for pair in pairedObjectsLocation:
            rfId = pair[0]
            irId = pair[1]
            pairIndex = pairIndex % pairScore.shape[0]
            pairScore[pairIndex] = mahalDist[rfId][irId]
            pairIndex += 1

    elif algOption == 3:
        if biasMethod == 4:
            # follow this procedure which is the 'correct' way to do it
            # specifically, finding the correct bias combination
            bias, testBias = mods.removeBias2(rfObjs, irObjs, sigmaRF, sigmaIR)
            print("\noption 4 bias: {}".format(bias))
        else:
            bias = mods.removeBias(rfObjs, irObjs, biasMethod)
            #bias = mods.removeBias(rfObjs, irObjs, 5)
            print("option 3 bias: {}".format(bias))

        rfUnBiased[:, 0] = rfObjs[:, 0] - bias[0]
        rfUnBiased[:, 1] = rfObjs[:, 1] - bias[1]
        #print("original rfObjects: \n{}".format(rfObjs))
        #print("unbiased rfObjects: \n{}".format(rfUnBiased))
        #print("original irObjects: \n{}".format(irObjs))

        bhattaDist, eucliDist = mods.bhattacharyya(rfUnBiased, irObjs, sigmaRF, sigmaIR, threshold=0.90)
        #bhattaDist, eucliDist = mods.bhattacharyya(rfObjs, irObjs, sigmaRF, sigmaIR, kFactor=1)
        if np.array_equal(bhattaDist, np.zeros_like(bhattaDist)):
            distanceMat = eucliDist
        else:
            distanceMat = bhattaDist
        # use the selected parameter
        condition = (np.amin(distanceMat, 1) < BIG_L)
        #if condition.any():
        #    print("boolean matrix for reducing final indices: {}".format(condition))
        #    pass
        #else:
        #    print("distanceMatrix: {}".format(np.amin(distanceMat, 1)))
        #    print("BIG_L value: {}".format(BIG_L))
        outIndx = np.where(condition)[0]
        probMat = distanceMat[ condition ]

        #print("initial RF Indices: {}".format(rfIndex))
        #print("initial IR Indices: {}".format(irIndex))
        finalRFIndex = rfIndex[outIndx].astype(int)
        finalNumRF = finalRFIndex.size
        #print("finalRFIndex before running Munkres: {}".format(finalRFIndex))

        if probMat.size > 0:
            finalProbCond = ( np.amin(probMat, 1) < BIG_L )
            finalProb = probMat[ finalProbCond ]
            # make sure finalPerm is non-empty
            if finalProb.ndim > 1:
                # break any permutations that are too far:
                rfPerm, irPerm, finalSum = mods.ModMunkres2(finalProb)

                if irPerm.size > 0:
                    finalPerm = irIndex[ irPerm ].astype(int)
                    finalCond = (finalPerm < BIG_L)
                    finalIRIndex = finalPerm[ finalCond ].astype(int)
                else:
                    outIndx = np.where(finalProbCond)[0]
                    finalIRIndex = irIndex[ outIndx ].astype(int)
                    finalNumIR = finalIRIndex.size
                # now do the rest of the stuff
                try:
                    # it was defined so we are good
                    assignments = finalIRIndex
                except NameError:
                    # finalIRindex was not defined
                    print("unable to create finalIRIndex")
                    print("initial IRIndex: {}".format(irIndex))
                    assignments = finalPerm
                    finalIRIndex = finalPerm

                pairedObjects = zip(finalRFIndex, finalIRIndex)
                pairedObjectsLocation = zip(rfPerm, irPerm)
                #print("pairedObjectsLocation: {}".format(pairedObjectsLocation))
                #print("finalRFIndex, finalIRIndex: {}, {}".format(finalRFIndex, finalIRIndex))
                # determine lethalObject ID
                pairScore = np.zeros(len(pairedObjects))
                pairIndex = 0
                for pair in pairedObjectsLocation:
                    #print("pairScore size: {}, pairIndex: {}".format(pairScore.size, pairIndex))
                    rfId = pair[0]
                    irId = pair[1]
                    pairIndex = pairIndex % pairScore.shape[0]
                    #print("finalProb shape: {}".format(finalProb.shape))
                    #print("rfId: {}, irId: {}".format(rfId, irId))
                    pairScore[pairIndex] = finalProb[rfId][irId]
                    pairIndex += 1
                #irObjectIDSet = finalIRIndex[ finalRFIndex == irIndex[0] ]
                #if irObjectIDSet.size > 0:
                #    lethalObjectNum = irObjectIDSet[0]
                #else:
                #    lethalObjectNum = 0
                try:
                    overallScore = finalSum
                except:
                    overallScore = finalProb
        else:
            #lethalObjectNum = 0
            assignments = np.zeros_like(finalRFIndex)
            pairedObjects = np.zeros(2)
            pairScore = np.array([100])
            overallScore = 100
            pass

    # return final results:
    #return finalIRIndex, overallScore, np.asarray(pairedObjects, dtype=int)
    return overallScore, pairScore, np.squeeze(np.asarray(pairedObjects, dtype=int))

"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""
