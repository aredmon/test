'''
Created on Nov 21, 2017

@author: mark.lambrecht

Module Description:
    run_sim - Simulates multiple Monte Carlo runs to exercise multiple weapon-target pairing algorithms
              for assigning multiple kill assets to multiple threat objects.
Class Descriptions:
    Doctrine    - contains the firing doctrine and associated constants for used in the algorithms
    KillVehicle - contains performance constants for individual kill assets
Important Function Descriptions:
    run_sim - Generates a specified number of iteration. In each iteration, the method does the following:
        1) Generate a scenario (kill asset-on-threat coverage, divert required, P(Kill) matrix, threat lethality)
        2) Exercise one or more WTA algorithms
        3) Accumulate statistics for each algorithm
        4) Create plots showing metrics
    Inputs:
        -N/A
    Outputs:
        -N/A
'''
import math
from matplotlib.lines import Line2D
import time

from assign_opt import assign_opt
from column import column
from fraction_threats_enagaged import fraction_threats_engaged
from fraction_threats_enagaged import inventory_expended
from greedy import greedy
from lethality_leakage import lethality_leakage
import matplotlib.pyplot as plt
from munkres_wrapper import munkres_wrapper
import numpy as np
import numpy.matlib as matlib
from sim_scenario import sim_scenario
from indices import indices


class Doctrine(object):
    maxShotsPerThreat = 2  # maximum shots per threat
    maxshotsPerWeapon = 1  # maximum shots per weapon
    valueCutoff = 0.099  # acceptable remaining lethality in task
    minPk = 0.4  # minimum pK before taking a shot
    pKWeight = 1.0  # weighting factor for Pk (used in Greedy algorithm)
    covWeight = 0.75  # global coverage penalty weight (used in Greedy algorithm)
    divWeight = 0.25  # divert penalty weight (used in Greedy algorithm)

    
class KillVehicle(object):
    avePk = 0.8  # average kill vehicle probability of kill
    inventory = 1  # inventory per kill vehicle
    maxDivert = 35  # maximum divert available, m/s

    
def run_sim():
    # constants
    doctrine = Doctrine()
    killVehicle = KillVehicle()
    nKVs = 6
    minThreats = 3
    maxThreats = 16
    nRuns = 2500
    nAlgorithms = 2
    
    # statistic for metrics
    numThreats = np.zeros((nRuns, 1))
    numMissiles = np.zeros((nRuns, 1))
    invExpended = np.zeros((nRuns, nAlgorithms))
    runTime = np.zeros((nRuns, nAlgorithms))
    fracLethNegated = np.zeros((nRuns, nAlgorithms))
    fracThrtsEngd = np.zeros((nRuns, nAlgorithms))
    divertUsed = np.zeros((nRuns, nAlgorithms))
    
    for iRun in range(0, nRuns):
        print('Run = ', iRun)
        algoId = 0
        
        # generate the scenario
        [nThreats, coverageMat, divertMat, pkMat, inventory, threatValues] = \
            sim_scenario(nKVs, minThreats, maxThreats, killVehicle)
        origPkMat = np.copy(pkMat)
        numThreats[iRun] = nThreats
        numMissiles[iRun] = nKVs
            
        # Munkres' Optimal Assignment algorithm
        lethality = matlib.repmat(threatValues.transpose(), nKVs, 1);
        costMat = np.multiply(lethality, pkMat)
        onesMat = np.ones(costMat.shape)
        costMat = np.subtract(onesMat, costMat)
        costMat = np.multiply(costMat, coverageMat)
        ndx = np.where(costMat <= 0)
        costMat[ndx] = 1
#         print("Cost Matrix:")
#         print(costMat)
        t0 = time.time()
        planMat = munkres_wrapper(costMat, doctrine.valueCutoff, divertMat, threatValues, killVehicle.avePk, doctrine.maxShotsPerThreat)
        t1 = time.time()
        runTime[iRun, algoId] = t1 - t0
        divertUsed[iRun, algoId] = np.sum(np.multiply(planMat, divertMat))
        fracLethNegated[iRun, algoId] = 1.0 - lethality_leakage(origPkMat, planMat, threatValues)
        fracThrtsEngd[iRun, algoId] = fraction_threats_engaged(planMat)
        invExpended[iRun, algoId] = inventory_expended(planMat)
#         print("MOA assignments:")
#         print(planMat)
        algoId = algoId + 1
        
        # Greedy Assignment algorithm
        t0 = time.time()
        planMat = greedy(doctrine, nKVs, nThreats, pkMat, divertMat, inventory, threatValues)
        t1 = time.time()
        runTime[iRun, algoId] = t1 - t0
        divertUsed[iRun, algoId] = np.sum(np.multiply(planMat, divertMat))
        fracLethNegated[iRun, algoId] = 1.0 - lethality_leakage(origPkMat, planMat, threatValues)
        fracThrtsEngd[iRun, algoId] = fraction_threats_engaged(planMat)
        invExpended[iRun, algoId] = inventory_expended(planMat)
#         print("Greedy assignments:")
#         print(planMat)
        algoId = algoId + 1
        
        # Brute Force Assignment algorithm
        planMat = np.zeros(coverageMat.shape, dtype=bool)
        wAvailable = np.where(np.sum(coverageMat, axis=1) > 0)
        tValAdj = np.fabs(threatValues - (np.ones(threatValues.shape) * doctrine.valueCutoff))
        tIgnore = np.where(np.logical_not(np.floor(tValAdj)) > 0)
        redPkMatTmp = origPkMat[~np.all(origPkMat == 0, axis=1)] # remove rows with all zeros
        redPkMat = redPkMatTmp[:, ~np.all(redPkMatTmp == 0, axis=0)] # remove cols with all zeros
        redCovTmp = coverageMat[~np.all(coverageMat == 0, axis=1)] # remove rows with all zeros
        redCov = redCovTmp[:, ~np.all(redCovTmp == 0, axis=0)] # remove cols with all zeros
        nT = np.sum(redCov, axis=1)
        assmt = np.zeros(redPkMat.shape, dtype=bool)
        nShot = doctrine.maxShotsPerThreat
        tVals = threatValues.copy()
        cvr = np.copy(redCov.T)
        poss = np.nonzero(cvr)
        [assign, best] = assign_opt(assmt.T, nT, nShot, poss, redPkMat, tVals)
        planMat[wAvailable, tIgnore] = assign.T
        algoId = algoId + 1
        
    # compute final metrics
    meanRT = np.zeros((maxThreats, nAlgorithms))
    sigmaRT = np.zeros((maxThreats, nAlgorithms))
    meanFracEng = np.zeros((maxThreats, nAlgorithms))
    sigmaFracEng = np.zeros((maxThreats, nAlgorithms))
    meanFracLethNeg = np.zeros((maxThreats, nAlgorithms))
    sigmaFracLethNeg = np.zeros((maxThreats, nAlgorithms))
    meanInvExp = np.zeros((maxThreats, nAlgorithms))
    sigmaInvExp = np.zeros((maxThreats, nAlgorithms))
    meanDU = np.zeros((maxThreats, nAlgorithms))
    sigmaDU = np.zeros((maxThreats, nAlgorithms))
    nObjs = np.zeros((maxThreats, nAlgorithms))
    for i in range(minThreats, maxThreats):
        # find all runs with "i" threats
        ndx = np.where(numThreats == i)
        nRunsI = np.size(ndx)
        if nRunsI > 1:
            rt = runTime[ndx, :]
            invExp = invExpended[ndx, :]
            fracEng = fracThrtsEngd[ndx, :]
            fracLethNeg = fracLethNegated[ndx, :]
            divUsd = divertUsed[ndx, :]
            for j in range(0, nAlgorithms):
                meanRT[i, j] = np.mean(column(rt, j))
                sigmaRT[i, j] = math.sqrt(np.var(column(rt, j)))
                meanFracEng[i, j] = np.mean(column(fracEng, j))
                sigmaFracEng[i, j] = math.sqrt(np.var(column(fracEng, j)))
                meanFracLethNeg[i, j] = np.mean(column(fracLethNeg, j))
                sigmaFracLethNeg[i, j] = math.sqrt(np.var(column(fracLethNeg, j)))
                meanInvExp[i, j] = np.mean(column(invExp, j))
                sigmaInvExp[i, j] = math.sqrt(np.var(column(invExp, j)))
                meanDU[i, j] = np.mean(column(divUsd, j))
                sigmaDU[i, j] = math.sqrt(np.var(column(divUsd, j)))
                nObjs[i, j] = i
    
    # plot metrics
    linestyles = ['_', '-', '--', ':']
    algorithms = ['MOA', 'Greedy']
    markers = []
    for m in Line2D.markers:
        try:
            if len(m) == 1 and m != ' ':
                markers.append(m)
        except TypeError:
            pass
    styles = markers + [
        r'$\lambda$',
        r'$\bowtie$',
        r'$\circlearrowleft$',
        r'$\clubsuit$',
        r'$\checkmark$']
    colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')

    dx = np.zeros((1, 2))
    dx[0, 0] = -0.1
    dx[0, 1] = 0.1
    # run time
    figNum = 1
    plt.figure(figNum)
    for i in range(0, nAlgorithms):
        color = colors[i % len(colors)]
        style = styles[(i - len(linestyles)) % len(styles)]
        x = column(nObjs, i)
        delX = dx[0, i]
        x[0:np.size(x)] += delX
        y = column(meanRT, i)
        for j in range(0, len(y)):
            y[j] = y[j] * 1000
        err = column(sigmaRT, i)
        for j in range(0, len(err)):
            err[j] = err[j] * 1000
        plt.errorbar(x, y, yerr=err, linestyle='None', marker=style, color=color, markersize=8, label=algorithms[i])
        plt.xlabel('# Threat Objects')
        plt.ylabel('Run Time, ms')
        plt.grid(True)
        plt.show()
        plt.hold(True)
    plt.legend()
    
    # fraction threats engaged
    figNum = figNum + 1
    plt.figure(figNum)
    for i in range(0, nAlgorithms):
        color = colors[i % len(colors)]
        style = styles[(i - len(linestyles)) % len(styles)]
        x = column(nObjs, i)
        delX = dx[0, i]
        x[0:np.size(x)] += delX
        y = column(meanFracEng, i)
        err = column(sigmaFracEng, i)
        plt.errorbar(x, y, yerr=err, linestyle='None', marker=style, color=color, markersize=8, label=algorithms[i])
        plt.xlabel('# Threat Objects')
        plt.ylabel('Fraction Threats Engaged')
        plt.grid(True)
        plt.show()
        plt.hold(True)
    plt.legend()

    # fraction lethality negated
    figNum = figNum + 1
    plt.figure(figNum)
    for i in range(0, nAlgorithms):
        color = colors[i % len(colors)]
        style = styles[(i - len(linestyles)) % len(styles)]
        x = column(nObjs, i)
        delX = dx[0, i]
        x[0:np.size(x)] += delX
        y = column(meanFracLethNeg, i)
        err = column(sigmaFracLethNeg, i)
        plt.errorbar(x, y, yerr=err, linestyle='None', marker=style, color=color, markersize=8, label=algorithms[i])
        plt.xlabel('# Threat Objects')
        plt.ylabel('Fraction Lethality Negated')
        plt.grid(True)
        plt.show()
        plt.hold(True)
    plt.legend()
    
    # divert used
    figNum = figNum + 1
    plt.figure(figNum)
    for i in range(0, nAlgorithms):
        color = colors[i % len(colors)]
        style = styles[(i - len(linestyles)) % len(styles)]
        x = column(nObjs, i)
        delX = dx[0, i]
        x[0:np.size(x)] += delX
        y = column(meanDU, i)
        err = column(sigmaDU, i)
        plt.errorbar(x, y, yerr=err, linestyle='None', marker=style, color=color, markersize=8, label=algorithms[i])
        plt.xlabel('# Threat Objects')
        plt.ylabel('Divert Expended, m/s')
        plt.grid(True)
        plt.show()
        plt.hold(True)
    plt.legend()


run_sim()
