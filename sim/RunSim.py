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
parentDir = os.path.dirname(os.path.realpath(os.getcwd()))
sys.path.append(parentDir)
import importModules as mods

def runSim():
    # set up the scenario
    kvStates = np.zeros((mods.SAPs.KV_NUM_KVS, 7))
    kvFuel = np.ones(mods.SAPs.KV_NUM_KVS) * mods.SAPs.KV_MAX_DIVERT
    [cvState, cvFuel, threatStates, rvID, tankID, 
            tPOCA, rPOCA, vClose, aClose, thtRadius] = mods.InitializeScenario()
    nThreats = threatStates.shape[0]
    assignedKVs = -1*np.zeros(mods.SAPs.KV_NUM_KVS)
    assignedThreats = -1*np.ones(mods.SAPs.KV_NUM_KVS)
    
    # begin the simulation
    kvsDispensed = False
    kvsDeployed = False
    tomRecvd = False
    irThreatsTracked = False
    t = mods.SAPs.MDL_T_START
    tFinal = t + mods.SAPs.MDL_T_DURATION + 1.0
    tStep = mods.SAPs.MDL_TIMESTEP
    lastAssign = False
    exceptionCounter = 0
    startTime = time.clock()
    while t < tFinal:
        # increment the time
        t += tStep
        elapsedTime = time.clock() - startTime
        print("Processing time step t = {}".format(t))
        print("current elapsed time = {}s".format(elapsedTime))
    
        # propagate the CV and KVs to the nearest time
        cvState = mods.PropagateECI(cvState, 0.0, tStep)
        if kvsDispensed:
            for iKV in range(kvStates.shape[0]):
                kvStates[iKV] = mods.PropagateECI(kvStates[iKV], 0.0, tStep)
        
        # propagate threats to the nearest time
        for iThrt in range(nThreats):
            threatStates[iThrt] = mods.PropagateECI(threatStates[iThrt], 0.0, tStep)
    
        # create the TOM at the appropriate time (one time only)
        if tFinal - t >= mods.SAPs.RDR_TOM_TGO:
            pLethalGround = mods.DiscriminationGround(nThreats, rvID, mods.SAPs.RDR_KFACTOR)
        
        if not tomRecvd and tFinal - t <= mods.SAPs.RDR_TOM_TGO:
            tomRecvd = True
            # Nx9, [x y z dx dy dz t P(RV) ID]
            tom, tomCov, tomIds = mods.MakeTOM(threatStates, rvID, t, pLethalGround, 
                    mods.SAPs.RDR_FRAC_TRACKS_DETECTED, unpack=True)
    
        # create tracks local to the CV/KV complex and perform TOM matching
        if tFinal - t <= mods.SAPs.CV_TGO_SENSOR_ON:
            pLethalOnboard = mods.DiscriminationOnboard(nThreats, rvID, mods.SAPs.CV_KFACTOR)
            onboardThreats, onboardThreatsCov, onboardThreatIds = mods.MakeOnboardThreats(threatStates, rvID, 
                    pLethalOnboard, mods.SAPs.CV_FRAC_TRACKS_DETECTED)
            # update the onboard threats with results of TOM matching TODO
            tomCorrData, winner = mods.CorrelateTOM(tom, tomCov, tomIds, onboardThreats, 
                    onboardThreatsCov, onboardThreatIds, mods.SAPs.KV_NUM_KVS, rvID, kvStates)
            print("tomCorrData shape: {}".format(np.asarray(tomCorrData).shape))
            print("winner from tomCorrelation: {}".format(winner))
            if not irThreatsTracked:
                irThreatTracker = mods.zipArray( (onboardThreatIds, np.ones_like(onboardThreatIds)) )
                irThreatsTracked = True
            else:
                pass
            # update lethality measurement with correlated ground and TOM threats 
            if winner.size > 0:
                rfIndex = int( winner[0] )
                irIndex = int( winner[1] )
                realIndex = np.where(irThreatTracker[:,0] == irIndex)[0]
                try:
                    irLocation = np.amin( realIndex )
                except ValueError:
                    exceptionCounter += 1
                    print("\nEXCEPTION OCCURRED {}: \nirIndex: {}, rfIndex: {}".format(
                        exceptionCounter, irIndex, rfIndex))
                    print("available ifIndices: {}\n".format(irThreatTracker[:,0]))
                    irLocation = 0
                weights = np.array([ irThreatTracker[ irLocation ][1], 1])
                irThreatTracker[ irLocation, 1 ] += 1
                print("original lethality estimate: {}".format(pLethalOnboard[irIndex]))
                pLethalTmp = np.array([pLethalOnboard[irIndex], pLethalGround[rfIndex]])
                pLethalOnboard[irIndex] = np.average(pLethalTmp, weights=weights)
                print(" updated lethality estimate: {}".format(pLethalOnboard[irIndex]))
            
    
        # dispense the KVs (when its time)
        # TODO investigate multiple dispenses (e.g. 1st for furthest threat
        # clusters)
        # TODO give time for start shot    
        if tFinal - t <= mods.SAPs.KV_BATTERY_LIFE and not kvsDispensed:
            kvStates = mods.DispenseKVs(cvState)
            kvsDispensed = True
            tDispense = t
    
        # deploy KVs (initial for zone defense)
        if kvsDispensed and not kvsDeployed and t > tDispense + 10.0:
            ## use the TOM to arrange KVs for intercept
            try:
                ind = int( np.where(tomIds == rvID) ) 
                rvIdTmp = int( tomIds[ind] )
            except TypeError:
                rvIdTmp = 0
    
            #pRVs = tom[:, 7]
            # need to grab the tomStates that are correlated with onboardThreat states
            # currently just taking the array for the 1st KV, need to figure out if this
            # is supposed to be coordinated with the winner index returned from correlateTOM
            #try:
            #    tomIds = tomCorrData[0][:,0]
            #except UnboundLocalError:
            #    tomIds = tom[:,8].astype(int)
            tomIds = tom[:,8].astype(int)
            # use tomIds to create tomStates variable
            tomSelectionCondition = [np.asscalar(np.where(tom[:,8] == selectedId)[0]) for selectedId in tomIds]
            redTOM = tom[ tomSelectionCondition ]
            tomStates = np.zeros((redTOM.shape[0], 7))
            for iTomThrt in range(redTOM.shape[0]):
                tomStates[iTomThrt] = mods.PropagateECI(redTOM[iTomThrt, 0:7], 0.0, t - redTOM[iTomThrt, 6])
            # DeployKVs
            cvState, cvFuel, kvStates, kvFuel, clusters = mods.DeployKVsTOM(cvState, cvFuel, kvStates, kvFuel, 
                    tomStates, tomIds, pLethalGround, tPOCA, rvIdTmp, 'aH5')
            # spread KVs around the threat radius for flexibility
            #         [cvState, cvFuel, kvStates, kvFuel] = mods.DeployKVsNoTOM(cvState, cvFuel, ...
            #             kvStates, kvFuel, threatStates, rPOCA, tPOCA, rvID, thtRadius, aH4);
            kvsDeployed = True
    
        #monitor the kinematic reach of the KVs
        if kvsDeployed:
            # TODO: check this with Larry
            # I think this is where tomCorrData comes in, it maps tomStates to the states seen by
            # the CV. So I think the threatStates here really need to be the subset of paired threats
            # that both the TOM and CV have information about. If this is incorrect let me know.
            #try:
            #    correlatedThreats = threatStates[ tomCorrData[0][:,1].astype(int) ]
            #except UnboundLocalError:
            #    correlatedThreats = threatStates
            correlatedThreats = threatStates.copy()
            # apply result to kinematic reach info
            krMatrix, dVMatrix = mods.KinematicReach(cvState, kvStates, kvFuel, correlatedThreats)
    
        # perform weapon-target pairing
        if kvsDeployed and not lastAssign:
            pkMat = krMatrix * mods.SAPs.WTA_AVE_KV_PK
            pkCond = (pkMat <= 0)       # boolean matrix - True if pkMat[i,j] <= 0 (it shouldn't)
            pkMat[pkCond] = 0.0         # replaces pkMat[i, j] <= 0 with 0 (unneccessary?)
            lethality = np.tile( pLethalGround, (mods.SAPs.KV_NUM_KVS, 1) )
            costMat = lethality * pkMat
            costMat = (1 - costMat) * krMatrix
            costCond = (costMat == 0)   # boolean matrix - True if costMat[i,j] == 0.0
            costMat[costCond] = 1.0     # replaces costMat[i,j] == 0.0 with 1.0
            # refine costMatrix based on assignedKVs to avoid assigning KVs that have already been assigned
            costMat[ (assignedKVs >= 0) ] = 1
            dVMatrix[ (assignedKVs >= 0) ] = 0
            pLethalRed = pLethalGround[(assignedThreats != 0)] * (1.0 - mods.doctrine.avePk)
            # use Munkres alg to make assignment
            planMat = mods.munkres(costMat)
            #VisualizeWTA(planMat, ~krMatrix, pLethalGround, [], [], [], [], aH6)
            # final assignment determination (kv-by-kv basis)
            inThreatIds = np.arange(threatStates.shape[0])
            finalAssignment, kvStates = mods.FinalAssignment(planMat, kvStates, kvFuel, 
                    threatStates, inThreatIds, 5.0)
    
            # incprporate finalAssign results
            assignedKVs = np.logical_or(assignedKVs, finalAssignment)
            # finalize assignment solutions
            reassignmentCondition = ( (assignedThreats == -1) & (assignedKVs != 0) )
            assignedThreats[ reassignmentCondition ] = finalAssignment[ reassignmentCondition ]
    
            # perform terminal homing:
            kvStates = mods.TerminalHoming(assignedThreats, kvStates, threatStates)

    return kvStates

if __name__ == "__main__":
    print("Running full simulation....")
    kvStatesFinal = runSim()
    print("final kvStates: \n{}".format(kvStatesFinal))
"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""
