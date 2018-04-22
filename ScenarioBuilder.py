"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*   Module Name: RunSim                                                                                     *
*   Author(s): Mark Lambrecht, Brent McCoy                                                                  *
*   Version: 1.0                                                                                            *
*   Date: 01/04/18                                                                                          *
*                                                                                                           *
*       Module        This is the scenario builder. It sets up the initial threat and                       *
*       Description:  carrier vehicle (CV) ponsitions. It calls the simulation driver                       *
*                     which then performs the initial CV divert and runs through the                        *
*                     scenario by time steps.                                                               *
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
*                                                                                                           *
*       Generates:    scenatioParameters.json                                                               *
*                                                                                                           *
*       History:      PBM 18 Apr 2018:  Initial version                                                     *
*                                                                                                           *
*************************************************************************************************************
"""
import os, json, sys
import numpy as np
import sim.importModules as mods
import argparse

def np2jsonDefault(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj.__dict__

def buildSim(inputDictionary={}):
    # set up the scenario
    #[cvState, cvFuel, threatStates, rvID, tankID, 
    #        tPOCA, rPOCA, vClose, aClose, thtRadius] = mods.InitializeScenario()
    dataStore = mods.InitializeScenario(inputDictionary)

    nThreats = dataStore["threatStates"].shape[0]
    
    # begin the simulation
    t = mods.SAPs.MDL_T_START
    tFinal = t + mods.SAPs.MDL_T_DURATION + 1.0
    tStep = mods.SAPs.MDL_TIMESTEP
    controlFlags = mods.SimpleNamespace(
            {
                "irThreatsTracked" : False,
                "kvsDispensed" : False,
                "kvsDeployed" : False,
                "lastAssign" : False,
                "tomRecvd" : False,
                "buildTom": False,
                "computeLethality": False,
                "terminateSim": True,
                "correlateMeasurements": False,
                "runSim": False,
                "exceptionCounter": 0
                }
            )

    while not controlFlags.runSim:
        # increment the time
        t += tStep
        print("Processing time step t = {}".format(t))
        #print("current elapsed time = {}s".format(elapsedTime))

        # propagate the CV and KVs to the nearest time
        dataStore["cvState"] = mods.PropagateECI(dataStore["cvState"], 0.0, tStep)
        
        # propagate threats to the nearest time
        for iThrt in range(nThreats):
            dataStore["threatStates"][iThrt] = mods.PropagateECI(dataStore["threatStates"][iThrt], 0.0, tStep)
    
        # create the TOM at the appropriate time (one time only)
        if tFinal - t >= mods.SAPs.RDR_TOM_TGO:
            pLethalGround = mods.DiscriminationGround(nThreats, dataStore["rvID"], mods.SAPs.RDR_KFACTOR)
        
        if not controlFlags.tomRecvd and tFinal - t <= mods.SAPs.RDR_TOM_TGO:
            controlFlags.tomRecvd = True
            # Nx9, [x y z dx dy dz t P(RV) ID]
            tom, tomCov, tomIds = mods.MakeTOM(dataStore["threatStates"], dataStore["rvID"], t, pLethalGround, 
                    mods.SAPs.RDR_FRAC_TRACKS_DETECTED, unpack=True)
    
        # create tracks local to the CV/KV complex and perform TOM matching
        if tFinal - t <= mods.SAPs.CV_TGO_SENSOR_ON and controlFlags.tomRecvd == True:
            controlFlags.correlateTOM = True
            pLethalOnboard = mods.DiscriminationOnboard(nThreats, dataStore["rvID"], mods.SAPs.CV_KFACTOR)
            onboardThreats, onboardThreatsCov, onboardThreatsIds = mods.MakeOnboardThreats(
                    dataStore["threatStates"], dataStore["rvID"], 
                    pLethalOnboard, mods.SAPs.CV_FRAC_TRACKS_DETECTED)
            # flag the system to call to runSim
            controlFlags.runSim = True
    
        ## dispense the KVs (when its time)
        ## TODO investigate multiple dispenses (e.g. 1st for furthest threat
        ## clusters)
        ## TODO give time for start shot    
        #if tFinal - t <= mods.SAPs.KV_BATTERY_LIFE and not controlFlags.kvsDispensed:
        #    kvStates = mods.DispenseKVs(dataStore["cvState"])
        #    controlFlags.kvsDispensed = True
        #    tDispense = t

    # set up call to RunSim
    dataStore.update(
            {
                "tCurr": t,
                "tFinal": tFinal,
                "tomStates": tom,
                "tomCov": tomCov,
                "tomIds": tomIds,
                "pLethalGround": pLethalGround,
                "onboardThreats": onboardThreats,
                "onboardThreatsCov": onboardThreatsCov,
                "onboardThreatsIds": onboardThreatsIds,
                "pLethalOnboard": pLethalOnboard
                }
            )
    # package controlFlags and data output 
    outputScenario = {"controlFlags": controlFlags.toDict(), "dataStore": dataStore}
    print(outputScenario)
    # release results
    return outputScenario

def serializeJSON(jsonData, scenarioFile="scenarioParameters.json"):
    try:
        print("cleaning up old scenario data file")
        os.remove(scenarioFile)
    except OSError:
        pass

    # output serialized json object
    with open(scenarioFile, 'w') as outfile:
        json.dump(jsonData, outfile, ensure_ascii=True, separators=(',',': '), encoding='utf8', 
                default=np2jsonDefault, indent=0)

def main(_):
    rmd_input = {}
    if parser.parse_args().rmdFile != "":
        with open(FLAGS.rmdFile, 'r') as json_file:
            rmd_input = json.load(json_file, encoding='utf8')

    for field in rmd_input:
        if isinstance(rmd_input[field], list):
            rmd_input.update({field: np.asarray(rmd_input[field])})

    # everything should be converted now, feed to buildSim
    serializeJSON(buildSim(rmd_input))

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rmdFile', required = False, default = "", help='input JSON string that contains RMDTool output data')
    FLAGS, unparsed = parser.parse_known_args()
    main([sys.argv[0], unparsed])

"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""