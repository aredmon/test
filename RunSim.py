"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*   Module Name: RunSim                                                                                     *
*   Author(s): Mark Lambrecht, Brent McCoy                                                                  *
*   Version: 1.0                                                                                            *
*   Date: 01/04/18                                                                                          *
*                                                                                                           *
*       Module        This is the simulation driver. It sets up the initial threat and                      *
*       Description:  carrier vehicle (CV) doncitions. It performs the initial CV                           *
*                     divert toward the center of the threat complex. It then runs                          *
*                     through the scenario by time steps.                                                   *
*                                                                                                           *
*       Algorithm:    N/A                                                                                   *
*                                                                                                           *
*       Inputs:       N/A                                                                                   *
*                                                                                                           *
*       Outputs:      N/A                                                                                   *
*                                                                                                           *
*       Calls:        sim.importModules                                                                     *
*                                                                                                           *
*       Requires:     SAPs_Null.json                                                                        *
*                     Doctrine_Null.json                                                                    *
*                                                                                                           *
*       Author:       M. A. Lambrecht                                                                       *
*                                                                                                           *
*       History:      MAL 12 Dec 2017:  Initial version                                                     *
*                                                                                                           *
*************************************************************************************************************
"""
import os, sys
import time
import numpy as np
import json
import argparse
import sim.importModules as mods
import ScenarioBuilder

def runSim(cvState, cvFuel, threatStates, rvID, tankID, tPOCA, rPOCA, vClose, aClose, 
        thtRadius, tCurr, tFinal, tomStates, tomCov, tomIds, onboardThreats, onboardThreatsCov, 
        onboardThreatsIds, pLethalGround, pLethalOnboard, controlFlags):
    # set up the scenario
    kvStates = np.zeros((mods.SAPs.KV_NUM_KVS, 7))
    kvFuel = np.ones(mods.SAPs.KV_NUM_KVS) * mods.SAPs.KV_MAX_DIVERT
    # initialize controls
    nThreats = threatStates.shape[0]
    assignedKVs = -1*np.ones(mods.SAPs.KV_NUM_KVS)
    assignedThreats = -1*np.ones(mods.SAPs.KV_NUM_KVS)
    
    # initialize kvStates to current cvState
    kvStates = mods.DispenseKVs(cvState)

    scpl = mods.UniformRandRange(0.1, 0.99);

    controls = mods.SimpleNamespace(controlFlags)
    #exceptionCounter = 0
    startTime = time.clock()
    tStep = mods.SAPs.MDL_TIMESTEP
    t = tCurr
    tDispense = t
    while t < tFinal:
        # increment the time
        t += tStep
        elapsedTime = time.clock() - startTime
        print("Processing time step t = {}".format(t))
        print("current elapsed time = {}s".format(elapsedTime))
    
        # propagate the CV and KVs to the nearest time
        cvState = mods.PropagateECI(cvState, 0.0, tStep)
        if controls.kvsDispensed:
            for iKV in range(kvStates.shape[0]):
                kvStates[iKV] = mods.PropagateECI(kvStates[iKV], 0.0, tStep)
        
        # propagate threats to the nearest time
        for iThrt in range(nThreats):
            threatStates[iThrt] = mods.PropagateECI(threatStates[iThrt], 0.0, tStep)
    
        # create tracks local to the CV/KV complex and perform TOM matching
        #if tFinal - t <= mods.SAPs.CV_TGO_SENSOR_ON and correlateTOM == True:
        if controls.correlateTOM == True:
            tomCorrData, winner = mods.CorrelateTOM(tomStates, tomCov, tomIds, onboardThreats, 
                    onboardThreatsCov, onboardThreatsIds, mods.SAPs.KV_NUM_KVS, rvID, kvStates)

            # update lethality measurement with correlated ground and TOM threats 
            for pair in tomCorrData[0]:
                print("selected pair: {}".format(pair))
                rfIndex = int( pair[0] )
                irIndex = int( pair[1] )
                print("original lethality estimate: {}".format(pLethalOnboard[irIndex]))
                pLethalOnboard[irIndex] = np.average([ pLethalOnboard[irIndex], pLethalGround[rfIndex] ])
                print(" updated lethality estimate: {}".format(pLethalOnboard[irIndex]))

            controls.kvsDeployed = True
            controls.correlateTOM = False
            
    
        # dispense the KVs (when its time)
        # TODO investigate multiple dispenses (e.g. 1st for furthest threat
        # clusters)
        # TODO give time for start shot    
        if tFinal - t <= mods.SAPs.KV_BATTERY_LIFE and not controls.kvsDispensed:
            kvStates = mods.DispenseKVs(cvState)
            controls.kvsDispensed = True
            tDispense = t
    
        # deploy KVs (initial for zone defense)
        if controls.kvsDispensed and not controls.kvsDeployed and t > tDispense + 10.0:
            ## use the TOM to arrange KVs for intercept
            try:
                ind = int( np.where(tomIds == rvID) ) 
                rvIdTmp = int( tomIds[ind] )
            except TypeError:
                rvIdTmp = 0
    
            tomIds = tomStates[:,8].astype(int)
            # use tomIds to refine tomStates variable
            tomSelectionCondition = [np.asscalar(np.where(tomStates[:,8] == selectedId)[0]) for selectedId in tomIds]
            redTOM = tomStates[ tomSelectionCondition ]
            tomStates = np.zeros((redTOM.shape[0], 7))
            for iTomThrt in range(redTOM.shape[0]):
                tomStates[iTomThrt] = mods.PropagateECI(redTOM[iTomThrt, 0:7], 0.0, t - redTOM[iTomThrt, 6])
            # DeployKVs
            cvState, cvFuel, kvStates, kvFuel, clusters = mods.DeployKVsTOM(cvState, cvFuel, kvStates, kvFuel, 
                    tomStates, tomIds, pLethalGround, tPOCA, rvIdTmp, 'aH5')
            #controls.kvsDeployed = True
    
        #monitor the kinematic reach of the KVs
        if controls.kvsDeployed:
            # apply result to kinematic reach info
            #krMatrix, dVMatrix = mods.KinematicReach(cvState, kvStates, kvFuel, correlatedThreats)
            krMatrix, dVMatrix = mods.KinematicReach(cvState, kvStates, kvFuel, onboardThreats[:,0:7])
    
        # perform weapon-target pairing
        if controls.kvsDeployed and not controls.lastAssign:
            pkMat = krMatrix * mods.SAPs.WTA_AVE_KV_PK
            pkCond = (pkMat <= 0)       # boolean matrix - True if pkMat[i,j] <= 0 (it shouldn't)
            pkMat[pkCond] = 0.0         # replaces pkMat[i, j] <= 0 with 0 (unneccessary?)
            #lethality = np.tile( pLethalGround, (mods.SAPs.KV_NUM_KVS, 1) )
            lethality = np.tile( pLethalOnboard[onboardThreatsIds], (mods.SAPs.KV_NUM_KVS, 1) )
            costMat = lethality * pkMat
            costMat = (1 - costMat) * krMatrix
            costCond = (costMat == 0)   # boolean matrix - True if costMat[i,j] == 0.0
            costMat[costCond] = 1.0     # replaces costMat[i,j] == 0.0 with 1.0
            # refine costMatrix based on assignedKVs to avoid assigning KVs that have already been assigned
            costMat[ (assignedKVs >= 0) ] = 1
            dVMatrix[ (assignedKVs >= 0) ] = 0
            print("assignedThreats: {}".format(assignedThreats))
            ##pLethalRed = pLethalGround[(assignedThreats != 0)] * (1.0 - mods.doctrine.avePk)
            #targetedThreats = assignedThreats[(assignedThreats > -1)]
            #print("targeted threats: {}".format(targetedThreats))
            #if targetedThreats.size > 0:
            #    pLethalRed = pLethalOnboard[targetedThreats]
            # use Munkres alg to make assignment
            #print("costMat: \n{}".format(costMat))
            planMat, boolMat, pLethalOnboard = mods.MunkresPairing(costMat, mods.doctrine.valueCutoff, dVMatrix, 
                    pLethalOnboard, mods.doctrine.avePk)
            #print("planMat: \n{}".format(planMat))
            # final assignment determination (kv-by-kv basis)
            inThreatIds = np.arange(threatStates.shape[0])

            finalAssignment, kvStates = mods.FinalAssignment(planMat, kvStates, kvFuel, 
                    onboardThreats[:,0:7], onboardThreatsIds, 5.0)
            #finalAssignment, kvStates = mods.FinalAssignment(planMat, kvStates, kvFuel, 
            #        threatStates, inThreatIds, 5.0)
            #finalAssignment = finalAssignment + 1

            print("sanity check: \nfinalAssignment - {}, assignedThreats - {}"\
                    .format(finalAssignment, assignedThreats))

            # incprporate finalAssign results
            assignedKVs = np.logical_or(assignedKVs, finalAssignment != -1)
            # finalize assignment solutions
            reassignmentCondition = ( (assignedThreats == -1) & (assignedKVs != 0) )
            assignedThreats[ reassignmentCondition ] = finalAssignment[ reassignmentCondition ]
    
            # perform terminal homing:
            kvStates = mods.TerminalHoming(assignedThreats, kvStates, threatStates)
            #kvStates = mods.TerminalHoming(finalAssignment, kvStates, threatStates)

    return kvStates

def main(_):
    # if rmdToolFile is input
    if parser.parse_args().rmdFile != "":
        with open(FLAGS.rmdFile, 'r') as json_file:
            rmd_input = json.load(json_file, encoding='utf8')
            for field in rmd_input:
                if isinstance(rmd_input[field], list):
                    rmd_input.update({field: np.asarray(rmd_input[field])})
            scenarioParameters = ScenarioBuilder.buildSim(rmd_input)
    # if scenarioParameters file is input
    elif parser.parse_args().scenarioFile != "":
        with open(FLAGS.scenarioFile, 'r') as jsonFile:
            scenarioParameters = json.load(jsonFile, encoding='utf8')
    # if no file provided as input
    else:
        scenarioParameters = ScenarioBuilder.buildSim()

    scenarioData = scenarioParameters["dataStore"]
    scenarioControl = scenarioParameters["controlFlags"]

    for field in scenarioData:
        if isinstance(scenarioData[field], list):
            scenarioData.update({field: np.asarray(scenarioData[field])})

    # everything should be converted now, feed to runSim
    kvStates = runSim(controlFlags=scenarioControl, **scenarioData)
    print("final kvStates: \n{}".format(kvStates))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--scenarioFile', type=str, default="",
            help='input JSON string that contains controlFlags and scenario setup data')
    parser.add_argument('-r', '--rmdFile', required=False, default="",
            help='input JSON string that contains RMDTool output data')
    FLAGS, unparsed = parser.parse_known_args()
    main([sys.argv[0], unparsed])


"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""
