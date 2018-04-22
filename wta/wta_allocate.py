'''
WTA Allocation functions
'''
import numpy

def wta_allocate(doctrine, nWES, nTgt, pkMat, divertMat, wpnInv, threatValues):
    '''
    Allocate the WTA
    '''

    planMat = numpy.zeros((nWES, nTgt))
    weights = [1.0, 0.75, 0.25]  # p(Kill), Coverage, Divert
    nAlloc = nTgt * doctrine.maxShotsThreat
    valI = numpy.zeros((nAlloc, 1))
    valF = numpy.zeros((nAlloc, 1))
    alloc = numpy.zeros((nWES, nTgt))
    tgtVal = threatValues
    origPkMat = pkMat

    # ensure each viable candidate can fit in the timeline of the resources
    if not MoreShots(pkMat, nWES):
        return

    # compute global penalties - TODO:  threats w/ most coverage have highest score - is this correct?
    # choose weapons that can only cover a given threat for that threat - vice other
    # weapons that can cover multiple threats
    coverageScore = CoverageScore(nTgt, nWES, pkMat)

    nShotsWeapon = numpy.zeros((nWES, 1))
    tgtValTmp = tgtVal
    i_alloc = 0
    finished = False

    while not finished:
        # no shots for out of inventory weapons
        ind = wpnInv == 0

        # TODO Need to make sure this is equivalent in Matlab
        pkMat[ind.T[0], :] = 0

        shots = False
        while not shots:
            val = numpy.max(tgtValTmp)
            highValueThreatIndex = numpy.argmax(tgtValTmp)

            if val <= 0:
                return

            shots = MoreShotsThreat(pkMat, nWES, highValueThreatIndex)

            if not shots:
                tgtValTmp[highValueThreatIndex] = 0

        divertScore = DivertScore(highValueThreatIndex, divertMat)

        maxScore = 0

        for iWES in range(0, nWES):
            if nShotsWeapon[iWES] < doctrine.maxShotsWeapon:
                a = pkMat[iWES, highValueThreatIndex]
                b = coverageScore[iWES]
                c = divertScore[iWES]
                score = weights[0] * pkMat[iWES, highValueThreatIndex] + \
                weights[1] * coverageScore[iWES] + weights[2] * divertScore[iWES]

                if score > maxScore and pkMat[iWES, highValueThreatIndex] != 0:
                    maxIndScore = iWES
                    maxScore = score

        if maxScore <= 0:
            return

        nShotsWeapon[maxIndScore] = nShotsWeapon[maxIndScore] + 1

        # bookkeeping for chosen plan
        i_alloc = i_alloc + 1
        chosenPk = pkMat[maxIndScore, highValueThreatIndex]
        pkMat[maxIndScore, highValueThreatIndex] = 1
        wpnInv[maxIndScore] = wpnInv[maxIndScore]
        alloc[maxIndScore, highValueThreatIndex] = alloc[maxIndScore,
                                                         highValueThreatIndex] + 1
        valI[i_alloc] = tgtValTmp[highValueThreatIndex]
        tgtValTmp[highValueThreatIndex] = tgtValTmp[highValueThreatIndex] * \
            (1 - chosenPk)

        if tgtValTmp[highValueThreatIndex] <= doctrine.valueCutoff or \
                numpy.sum(alloc[:, highValueThreatIndex]) > doctrine.maxShotsThreat:
            pkMat[:, highValueThreatIndex] = 0

        ind = numpy.nonzero(tgtValTmp > doctrine.valueCutoff)
        if numpy.size(ind) == 0:
            finished = True
        if not MoreShots(pkMat, nWES):
            finished = True

        for ii in range(1, nWES):
            p = numpy.nonzero(planMat[ii, :] > 0)

            if numpy.size(p) == 0:
                thrtInd = divertMat[ii, :] > 0
                idx = numpy.argmax(valF[thrtInd])
                planMat[ii, idx] = 1
                valF[idx] = valF[idx] * (1.0 - origPkMat[ii, idx])

    return [planMat, valI, valF]


def MoreShots(pkMat, nWES):
    moreShots = False

    for iWES in range(0, nWES):
        ind = numpy.nonzero(pkMat[iWES, :])

        if numpy.size(ind) != 0:
            moreShots = True
            break
    return moreShots


def MoreShotsThreat(pkMat, nWES, iTgt):

    moreShots = False

    for iWES in range(0, nWES):
        ind = numpy.nonzero(pkMat[iWES, iTgt] > 0)[0]

        if numpy.size(ind) > 0:
            moreShots = True
            break

    return moreShots


def CoverageScore(nTgt, nWES, pkMat):

    # build the coverage table
    wesCoverage = numpy.zeros((nWES, nTgt))
    wesCoverage[pkMat > 0] = 1
    wesEngThreats = numpy.zeros((nWES, 1))

    # Compute total number of engageable threats
    totEngThreats = 0
    for ii in range(0, nTgt):
        coverageNotZero = numpy.nonzero(wesCoverage[:, ii] > 0)
        if numpy.size(coverageNotZero) > 0:
            totEngThreats = totEngThreats + 1

    # Compute the number of engageable threats per target
    for ii in range(0, nWES):
        wesEngThreats[ii] = numpy.size(numpy.nonzero(wesCoverage[ii, :]))

    # Basis element
    x = wesEngThreats / totEngThreats

    # Compute the score
    coverageScore = 1.0 - x

    return [coverageScore]


def DivertScore(trgtIndex, divertMat):
    xd = numpy.shape(divertMat)[0]
    divertScore = numpy.zeros((numpy.shape(divertMat)[0], 1))
    diverts = divertMat[:,trgtIndex]
    diverts = numpy.reshape(diverts, (diverts.shape[0], 1))
    goodInd = numpy.nonzero(diverts > 0)
    maxDivert = diverts[goodInd].max(0)
    divertScore[goodInd] = 1.0 - diverts[goodInd] / maxDivert

    return divertScore


def MoreShotsThreat2(pkMat, nWES, iTgt):
    
    moreShots = False

    for iWES in range(0,nWES):
        ind = numpy.nonzero(pkMat(iWES, iTgt) > 0)
        if numpy.size(ind) == 0:
            moreShots = True
            break

    return moreShots
