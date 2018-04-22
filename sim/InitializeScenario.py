"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*   Module Name: InitializeScenario                                                                         *
*   Author(s): Mark Lambrecht, Brent McCoy                                                                  *
*   Version: 1.0                                                                                            *
*   Date: 01/03/18                                                                                          *
*                                                                                                           *
*       Module        Build the initial CV state and the initial threat states. If                          *
*       Description:  input files are available, use those.  Else generate a random                         *
*                     threat complex (properties are SAP controlled), as well as a CV                       *
*                     trajectory.                                                                           *
*                                                                                                           *
*       Algorithm:    N/A                                                                                   *
*                                                                                                           *
*       Inputs:       N/A                                                                                   *
*                                                                                                           *
*       Outputs:      cvState      - 7x1 CV state [x y z dx dy dz t], ECI                                   *
*                     cvFuel       - CV's fuel allotment                                                    *
*                     threatStates - #Threatsx7 states [x y z dx dy dz t], ECI                              *
*                     rvID         - (Scalar) ID of RV                                                      *
*                     tankID       - (Scalar) ID of Tank                                                    *
*                     tPOCA        - (Scalar) time of POCA, s                                               *
*                     rPOCA        - 3x1 position of POCA (CV/<threatState>), ECI                           *
*                     vClose       - (Scalar) closing speed (CV/<threatState>, m/s                          *
*                     aClose       - (Scalar) closing angle (CV/<threatState>, radians                      *
*                     thtRadius    - (Scalar) radius of threat complex, m                                   *
*                                                                                                           *
*       Calls:        generalUtilities.utilities.UniformRandRange                                           *
*                     generalUtilities.utilities.AngleBetweenVecs                                           *
*                     generalUtilities.propagate.PropagateECI                                               *
*                     generalUtilities.GaussProblem.GaussProblem                                            *
*                     generalUtilities.FindPOCA.FindPOCA                                                    *
*                                                                                                           *
*       Requires:     SAPs_Null.json                                                                        *
*                                                                                                           *
*       OA:           M. A. Lambrecht                                                                       *
*                                                                                                           *
*       History:      MAL 12 Dec 2017:  Initial version                                                     *
*                                                                                                           *
*************************************************************************************************************
"""
import os
import numpy as np
import importModules as mods

#tBegin = mods.SAPs.MDL_T_START
#tEnd = tBegin + mods.SAPs.MDL_T_DURATION + 1

def generateStates(rPdmFinal, vPdmFinal, rCVFinal, vCVFinal, tFinal):
    tmpState = np.append(rPdmFinal, vPdmFinal)
    tmpState = np.append(tmpState, [mods.SAPs.MDL_T_DURATION])
    #print("generate the initial threat PDM state")
    # generate the initial threat PDM state
    sPdmInitial = mods.PropagateECI(tmpState.reshape(7,1), 0.0, -mods.SAPs.MDL_T_DURATION)

    #print("generate the center point of each cloud. the PDM is in the center of the complex"
    #+ "while other clouds surround it.")
    # generate the center point of each cloud. the PDM is in the center of the complex
    # while other clouds surround it.
    nClouds = np.random.randint(mods.SAPs.THT_MIN_NUM_CLOUDS, mods.SAPs.THT_MAX_NUM_CLOUDS+1)
    rCloud = np.zeros((nClouds, 3))
    thtRadius = mods.UniformRandRange(mods.SAPs.THT_MIN_RADIUS, mods.SAPs.THT_MAX_RADIUS)[0]
    for iCloud in range(nClouds):
        # compute the position of the cloud w.r.t. the PDM at the final time
        delR = np.random.rand(1,3)
        delR = delR / np.linalg.norm(delR)
        delR = delR * np.absolute( np.random.randn() * thtRadius )
        rCloud[iCloud] = rPdmFinal + delR

    #print("generate the position of each threat at time = tFinal")
    # generate the position of each threat at time = tFinal
    rDeviation = mods.SAPs.THT_POS_DEV * np.random.rand()
    nThreats = np.random.randint(mods.SAPs.THT_MIN_NUM_OBJS, mods.SAPs.THT_MAX_NUM_OBJS+1)
    threatStates = np.zeros((nThreats, 7))
    for iThreat in range(nThreats):
        # determine which cloud this threat belongs to
        iCloud = np.random.randint(0, nClouds)
        # determine it's final state
        rFinal = rCloud[iCloud] + np.random.randn(1,3) * rDeviation
        sFinal, _ = mods.GaussProblem(rFinal, sPdmInitial[0:3], mods.SAPs.MDL_T_DURATION)
        vFinal = -sFinal[3:6]
        threatStates[iThreat, 0:3] = rFinal
        threatStates[iThreat, 3:6] = vFinal
        threatStates[iThreat, 6] = tFinal
    # make cvState
    tmpCVState = np.concatenate((rCVFinal.reshape(1,3), vCVFinal.reshape(1,3)), axis=1)
    tmpCVState = np.append(tmpCVState, [tFinal])
    cvState = mods.PropagateECI(tmpCVState.reshape(7,1), 0.0, -mods.SAPs.MDL_T_DURATION)

    # print statistics to the screen
    print("# Threat Objects     = {}".format(nThreats))
    print("# Threat Clouds      = {}".format(nClouds))
    print("Threat Divergence    = {} m/s".format(mods.SAPs.THT_VEL_DEV))
    print("EV Radius            = {} m".format(thtRadius))

    # release results
    return cvState, threatStates, thtRadius

"""
inputDictionary {
    cvState=[]
    threatStates=[]
    tStart=None
    tFinal=None
    tankID=None
    rvID=None
    thtRadius=None
    }
"""
def findClosingValues(inputDictionary={}):
    print("INPUT DICTIONARY", inputDictionary)

    tStart = inputDictionary.get('tStart')
    if tStart == None:
        tStart = mods.SAPs.MDL_T_START
    
    tFinal = inputDictionary.get('tFinal')
    if tFinal == None:
        tFinal = tStart + mods.SAPs.MDL_T_DURATION

    rCVFinal = np.array([-7, 0, 0]) * 1e6       # final CV position [ECI, m]
    cvSpeed = mods.UniformRandRange(mods.SAPs.CV_SPEED_MIN, mods.SAPs.CV_SPEED_MAX)
    vCVFinal = np.array([0, 1, 0]) * cvSpeed    # final CV velocity [ECI, m]
    rPdmFinal = np.array([-7, 0, 0]) * 1e6      # final Payload Deployment Module (PDM) position, [ECI, m]
    inBounds = False
    while not inBounds:
        v = np.array([0, np.random.randn(), np.random.rand()])
        v = v / np.linalg.norm(v)
        theta = mods.AngleBetweenVecs(vCVFinal, v)
        if theta > mods.SAPs.CV_CLOSE_ANGLE_MIN and theta <= mods.SAPs.CV_CLOSE_ANGLE_MAX:
            inBounds = True

    thtSpeed = mods.UniformRandRange(mods.SAPs.THT_SPEED_MIN, mods.SAPs.THT_SPEED_MAX)
    vPdmFinal = v * thtSpeed
    vClose = np.linalg.norm(vPdmFinal - vCVFinal)
    aClose = mods.AngleBetweenVecs(vPdmFinal, vCVFinal)

    cvState = inputDictionary.get('cvState')
    threatStates = inputDictionary.get('threatStates')
    thtRadius = inputDictionary.get('thtRadius')
    # perform initial CV divert
    if cvState is None or threatStates is None:
        print("CAUTION: CV State or ThreatStates is None. Generating States")
        cvState, threatStates, thtRadius = generateStates(rPdmFinal, vPdmFinal, rCVFinal, vCVFinal, tFinal)

    nThreats = threatStates.shape[0]

    if thtRadius == None:
        thtRadius = mods.UniformRandRange(mods.SAPs.THT_MIN_RADIUS, mods.SAPs.THT_MAX_RADIUS)[0]

    if inputDictionary.get('rvID') == None:
        rvID = np.random.randint(nThreats)

    if inputDictionary.get('tankID') == None:
        tankID = np.random.randint(nThreats)

    # compute the average state
    aveState = np.sum(threatStates, axis=0) / nThreats
    aveState = mods.PropagateECI(aveState.reshape(7,1), 0.0, -mods.SAPs.MDL_T_DURATION)

    for iThreat in range(nThreats):
        threatStates[iThreat] = mods.PropagateECI(threatStates[iThreat].reshape(7,1), 0.0, 
                -mods.SAPs.MDL_T_DURATION)

    tPOCA, rPOCA, _ = mods.FindPOCA(cvState, aveState)
    #print("findPOCA executed, passing on to GaussProblem...")
    cv, _ = mods.GaussProblem(cvState[0:3], rPOCA, tPOCA - cvState[6])
    dV = cv[3:6] - cvState[3:6]
    # tBurn1 = np.linalg.norm(dV) / mods.SAPs.CV_MAX_ACC
    cvState[3:6] = cvState[3:6] + dV
    cvFuel = mods.SAPs.CV_MAX_DIVERT - np.linalg.norm(dV)

    # print statistics to the screen
    print("Closing Velocity     = {} m/s".format(vClose))
    print("Closing Angle        = {} deg".format(aClose * 180/np.pi))
    print("rvID                 = {}".format(rvID))

    # return outputs
    return {
            "cvState":          cvState, 
            "cvFuel" :          cvFuel, 
            "threatStates" :    threatStates, 
            "rvID" :            rvID, 
            "tankID" :          tankID, 
            "tPOCA" :           tPOCA, 
            "rPOCA" :           rPOCA, 
            "vClose" :          vClose, 
            "aClose" :          aClose, 
            "thtRadius" :       thtRadius
            }

def InitializeScenario(inputDictionary={}, unpackOutput=False):
    #print("Setting up initial scenario...")
    if not os.path.isdir('sim'):
        if os.getcwd().rsplit('/', 1)[-1] != 'sim':
            currentDir = os.path.dirname(os.path.realpath(os.getcwd()))
            simDir = os.path.join(currentDir, 'sim')
            # clear out old pLethal files
            persistentGroundFile = simDir + '/persistentVarsGround.npz'
            persistentOnboardFile = simDir + '/persistentVarsOnboard.npz'
            # remove persistentGroundFile
            try:
                #print("cleaning out old lethality")
                os.remove(persistentGroundFile)
            except OSError:
                pass
            # remove persistentOnboardFile
            try:
                #print("cleaning out old lethality")
                os.remove(persistentOnboardFile)
            except OSError:
                pass

        else:
            persistentGroundFile = 'persistentVarsGround.npz'
            persistentOnboardFile = 'persistentVarsOnboard.npz'
            # remove persistentGroundFile
            try:
                #print("cleaning out old lethality")
                os.remove(persistentGroundFile)
            except OSError:
                pass
            # remove persistentOnboardFile
            try:
                #print("cleaning out old lethality")
                os.remove(persistentOnboardFile)
            except OSError:
                pass
    else:
        persistentGroundFile = 'sim/persistentVarsGround.npz'
        persistentOnboardFile = 'sim/persistentVarsOnboard.npz'
        # remove persistentGroundFile
        try:
            os.remove(persistentGroundFile)
        except OSError:
            pass
        # remove persistentOnboardFile
        try:
            os.remove(persistentOnboardFile)
        except OSError:
            pass

    # return outputs
    if unpackOutput:
        resultDictionary = findClosingValues(inputDictionary)
        results = mods.SimpleNamespace(resultDictionary)
        return results.cvState, results.cvFuel, results.threatStates, results.rvID, results.tankID, \
                results.tPOCA, results.rPOCA, results.vClose, results.aClose, results.thtRadius
    else:
        return findClosingValues(inputDictionary)
"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""
