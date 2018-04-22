"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*   Module Name: CorrelateTOM                                                                               *
*   Author(s): Larry Gariepy, Brent McCoy                                                                   *
*   Version: 1.0                                                                                            *
*   Date: 01/17/18                                                                                          *
*                                                                                                           *
*************************************************************************************************************
"""
import numpy as np
import importModules as mods
#from generalUtilities.utilities import yaw_scene, pitch_scene, findMode
#from generalUtilities.config import TOM_SAPS

def swapMat(matrix, indA, indB):
    tmpA = matrix[indA].copy()
    matrix[indA] = matrix[indB].copy()
    matrix[indB] = tmpA
    return matrix

def rotateMat2x2(angle, invert=False):
    rotMat = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]])
    if invert:
        rotMat = np.transpose( rotMat )
    # return final roation matrix
    return rotMat

def determine_winner(method, QV):
    # winner should be an nKV length array containing at least the 
    # irID of the RV that has been determined as the lethalObject
    # consider passing the paired mapping of the irID and the rfID
    # that satisfies the desired condition.
    BIG_L = mods.TOM_SAPS.BIG_L
    
    for QVSet in QV:
        # majority rules
        if method == 1:
           mode, freq, mult = mods.findMode(QV[:,0])
           num_modes = mult[0].size
           if num_modes > 1:
               index = np.random.randint(num_modes)
               winner = mult[0][index]
           else:
               winner = mode[0]

        # best score
        elif method == 2:
            best_score = np.argmin( QVSet[:,0] )
            winner = QVSet[ best_score, 1:]
            #try:
            #    QVred = QV[ QV[:,1:] < BIG_L ]
            #    best_score = np.argmin( QVRed[:,0] )
            #    winner = QV[ best_score, 1: ]
            #except ValueError:
            #    best_score = np.argmin( QVSet[:,0] )
            #    winner = QV[ best_score, 1:]
            ## select winner

    # return winning index
    return np.squeeze(np.asarray(winner, dtype=int))

"""
-----------------------------------------------------------------------------------------------------------
    Inputs:                                                                                              
       N_KVS        - number of kill assets present (carrier should be included if it is participating   
                      in TOM Correlation)                                                                
       N_total      - total number of distinct objects present in either RF or IR scenes                 
       RF_objects   - ECI positions of objects reported by RF ground sensors;  RV assumed to be first    
                      object in list; array of size N_RF x 4 [ RF_ID ECI_X  ECI_Y  ECI_Z ]               
       IR_objects   - ECI positions of objects reported by IR sensors;  array of size N_KVS x N_IR x 4   
                      [IR_ID  ECI_X  ECI_Y  ECI_Z ]                                                      
        NOTE: currently assumes that all IR sensors see all IR objects                                   
                                                                                                         
       RF_cov       - ECI covariance of RF objects;  array of size N_RF x 3 x 3                          
       IR_cov       - ECI covariance of IR objects;  array of size N_KVS x N_IR x 3 x 3                  
       KV_pos       - ECI positions of kill assets;  array of size N_KVS x 3                             
                                                                                                         
    Outputs:                                                                                             
       TOM_correlation_data     - N x 3 array with matched pairs of objects;  each row is                
                                  [KV_ID  RF_ID  IR_ID]  for pairs of objects that are correlated        
       winner                   - the IR object ID that is correlated with the RF RV                     
-----------------------------------------------------------------------------------------------------------
"""
def TomCorrelation(n_KVs, n_total, RF_objects, IR_objects, RF_cov, IR_cov, KV_pos):
    # set control variables
    ALG_OPTION = 3
    #ALG_OPTION = 2
    BIAS_METHOD = 4
    #BIAS_METHOD = 6
    CARRIER_SENSOR = False      # Default to using a carrier with a sensor aperture
    VOTING_METHOD = 2           # 1 = majority rules    2 = best score rules
    GatingK = 5                 # maximum sigmas of separation for an allowable correlation

    # set up index variables
    indRF = RF_objects[:, 0]
    indIR = IR_objects[:, :, 0]

    num_rf_objects = RF_objects.shape[0]
    num_ir_objects = IR_objects.shape[1]

    # Step 1: recenter coordinated on RV
    #--------------------------------- Step 1 ------------------------------------------
    RV_pos = RF_objects[0, 1:4]     # assume first object is RV
    RF_objects[:,1:4] = np.subtract( RF_objects[:,1:4], RV_pos )

    for ii in range(n_KVs):
        IR_objects[ii,:, 1:4] = np.subtract( IR_objects[ii,:, 1:4], RV_pos )

    KV_pos_adjusted = KV_pos - RV_pos

    #------------------------------ END OF STEP 1 --------------------------------------
    SigIR_2D = np.zeros((n_KVs, num_ir_objects, 2, 2))
    SigRF_2D = np.zeros((n_KVs, num_rf_objects, 2, 2))
    # the components per object per KV are [az el cov_major_axis cov_minor_axis cov_theta]
    KV_scene_2D_IR = np.zeros((n_KVs, num_ir_objects, 2))
    KV_scene_2D_RF = np.zeros((n_KVs, num_rf_objects, 2))

    # Inputs: TOM_truth_objects, KV_pos, N_total, object_flags, RF_cov_3D
    # Outputs: RF_projected_cov_2D, KV_scene_3D
    KV_scene_3D_IR = np.zeros((n_KVs, num_ir_objects, 3))
    KV_scene_3D_RF = np.zeros((n_KVs, num_rf_objects, 3))

    #QV = np.zeros( (n_KVs, 2) )
    QV = np.zeros( (n_KVs, 3) )
    QVpair = [None] * n_KVs

    counterTOM = 0
    TOM_correlation_data = []
    # Step 2: project the KV IR scene into the KV lineOfSight perspective, generating az/el truth
    #--------------------------------- Step 2 ------------------------------------------
    for ii in range(n_KVs):
        print("\nprocessing data for KV {}".format(ii))
        azKV = np.arctan2( KV_pos_adjusted[ii, 1], KV_pos_adjusted[ii, 0] )
        elKV = np.arctan2( KV_pos_adjusted[ii, 2], np.linalg.norm(KV_pos_adjusted[ii, 0:2]) )

        pitchMat = mods.pitch_scene(elKV, np.identity(3))
        yawMat = mods.yaw_scene(-azKV, np.identity(3))
        rotateLeft = np.dot( yawMat, pitchMat )
        rotateRight = np.dot( pitchMat.transpose(), yawMat.transpose())

        KV_scene_3D_IR[ii] = IR_objects[ii,:, 1:4]
        KV_scene_3D_IR[ii] = mods.pitch_scene(elKV, KV_scene_3D_IR[ii])
        KV_scene_3D_IR[ii] = mods.yaw_scene(-azKV, KV_scene_3D_IR[ii])

        KV_scene_3D_RF[ii] = RF_objects[:, 1:4]
        KV_scene_3D_RF[ii] = mods.pitch_scene(elKV, KV_scene_3D_RF[ii])
        KV_scene_3D_RF[ii] = mods.yaw_scene(-azKV, KV_scene_3D_RF[ii])

        # meters to rad scale factor?
        meters_to_rad_scale_factor = np.arctan2(1, mods.vecNorm(KV_pos_adjusted))

        # in the rotated frame, the KV lineOfSight coincides with the X-axis, but the KV_pos itself 
        # isn't rotated, it reflects the position relative to the original scene

        rngXY_IR = np.transpose( 
                np.vstack(
                    (
                        np.subtract( KV_scene_3D_IR[ii,:,0],mods.vecNorm(KV_pos_adjusted)),
                        KV_scene_3D_IR[ii,:,1], 
                        )
                    ) 
                )

        rngXY_RF = np.transpose( 
                np.vstack(
                    (
                        np.subtract( KV_scene_3D_RF[ii,:,0],mods.vecNorm(KV_pos_adjusted)),
                        KV_scene_3D_RF[ii,:,1], 
                        )
                    ) 
                )

        downRangeIR = mods.vecNorm(rngXY_IR, axis=1)
        downRangeRF = mods.vecNorm(rngXY_RF, axis=1)

        # generate azimuth angle based on the y-coordinate (opposite) and x-coord (adjacent)
        azIR = np.arctan2( rngXY_IR[:, 1], rngXY_IR[:, 0] )
        azRF = np.arctan2( rngXY_RF[:, 1], rngXY_RF[:, 0] )
        # generate elevation angle based on the z-coord (opposite) and downrange distance in 
        # XY plane (adjacent)
        elIR = np.arctan2( KV_scene_3D_IR[ii,:,2], downRangeIR )
        elRF = np.arctan2( KV_scene_3D_RF[ii,:,2], downRangeRF )

        KV_scene_2D_IR[ii] = np.transpose( np.vstack((azIR, elIR)) )
        KV_scene_2D_RF[ii] = np.transpose( np.vstack((azRF, elRF)) )

        projectedRFCov = np.zeros_like(RF_cov)
        for kk, covariance in enumerate(RF_cov):
            projectedRFCov[kk] = np.dot(rotateLeft, np.dot(covariance, rotateRight) )

        aIR, bIR, thetaIR = mods.project_covariance(np.zeros(3), KV_pos_adjusted[ii], 
                IR_cov[ii] * meters_to_rad_scale_factor**2)
        aRF, bRF, thetaRF = mods.project_covariance(np.zeros(3), KV_pos_adjusted[ii], 
                projectedRFCov * meters_to_rad_scale_factor**2)

        # MatLab code creates a 2x2 diagonal matrix with a^2 and b^2. It then multiplies this
        # matrix against a standard 2x2 rotation matrix based on the angle theta. The goal of
        # this piece is to do the same thing with the a, b, and theta 1xn arrays returned from 
        # project_covariance. Result is a covariance matrix assuming the original data-set follows
        # a multivariate normal distribution.
        for a2, b2, theta, jj in zip(np.square(aIR), np.square(bIR), thetaIR, np.arange(aIR.shape[0])):
            SigIR_2D[ii,jj] = np.dot( np.diag([a2,b2]), rotateMat2x2(theta) )

        for a2, b2, theta, jj in zip(np.square(aRF), np.square(bRF), thetaRF, np.arange(aRF.shape[0])):
            SigRF_2D[ii,jj] = np.dot( np.diag([a2,b2]), rotateMat2x2(theta) )

    # Step 6: Rotate object positions and covariances                                            
    #--------------------------------- Step 6 ------------------------------------------
        calculateQVinputs = {
                "algOption"  :   ALG_OPTION,
                "biasMethod" :   BIAS_METHOD,
                "rfObjects"  :   KV_scene_2D_RF[ii],
                "irObjects"  :   KV_scene_2D_IR[ii],
                "sigmaRF"    :   SigRF_2D[ii],
                "sigmaIR"    :   SigIR_2D[ii],
                "gatingK"    :   GatingK,
                "numRF"      :   num_rf_objects,
                "numIR"      :   num_ir_objects,
                "rfIndex"    :   indRF,
                "irIndex"    :   indIR[ii]}
        
        #QV[ii, 0], QV[ii, 1], pairedObjects = mods.calculate_qv_single_asset2(**calculateQVinputs)
        QV[ii, 0], pairScore, pairedObjects = mods.calculate_qv_single_asset2(**calculateQVinputs)
        if pairScore.shape[0] > 1:
            #print("pairScoreShape: {}".format(pairScore.shape))
            #print("pairedObjects shape: {}".format(pairedObjects.shape))
            #QVset = np.hstack( (pairScore.reshape(pairScore.size, 1), pairedObjects) )
            QVset = np.hstack( (np.reshape(pairScore, (pairScore.size, 1)), pairedObjects) )
        else:
            QVset = np.hstack( (pairScore, pairedObjects) )
        QVpair[ii] = QVset

        rows = pairedObjects.shape[0]
        print("object pairs: \n{} \nrfObjects: {} \nirObjects: {}".format(
            np.squeeze(pairedObjects), num_rf_objects, num_ir_objects))

        TOM_correlation_tmp = np.zeros( (rows, 2) )
        # since python can enumerate an array automatically we will suppress the column that
        # would normally contain the KV_ID as that will be the index of the list anyhow
        TOM_correlation_tmp[:] = np.squeeze( pairedObjects.astype(int) )
        TOM_correlation_data.append(TOM_correlation_tmp)
        
    # Step 7: Determine global winner
    #--------------------------------- Step 7 ------------------------------------------
    if CARRIER_SENSOR:
        # VOTING_METHOD = 1
        globalWinner = determine_winner(VOTING_METHOD, QVpair)
        #print("selected QV information: {}".format(np.squeeze(QVpair[winner])))
        #selectedQV = np.squeeze(QVpair[winner])
        #if selectedQV.ndim > 1:
        #    setWinner = determine_winner(VOTING_METHOD, selectedQV)
        #    globalWinner = np.asarray( selectedQV[setWinner][1:] , dtype=int ) 
        #else:
        #    globalWinner = np.asarray( selectedQV[1:], dtype=int )
    else:
        # VOTING_METHOD = 2
        globalWinner = determine_winner(VOTING_METHOD, QVpair)
        #print("selected QV information: {}".format(np.squeeze(QVpair[winner])))
        #selectedQV = np.squeeze(QVpair[winner])
        #if selectedQV.ndim > 1:
        #    print("selectedQV: {}".format(selectedQV))
        #    setWinner = determine_winner(VOTING_METHOD, selectedQV)
        #    globalWinner = np.asarray( selectedQV[setWinner][1:], dtype=int )
        #else:
        #    globalWinner = np.asarray( selectedQV[1:], dtype=int )
    print("winners from determine winner: {}".format(globalWinner)) 
    return TOM_correlation_data, globalWinner

"""
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
"""
def CorrelateTOM(rfStates, rfCov, rfIds, irStates, irCov, irIds, nKVs, rvID, kvStates):
    # set constants
    nRF = rfStates.shape[0]
    nIR = irStates.shape[0]

    # find the latest time in the states
    rfLatest = max(rfStates[:,6])
    irLatest = max(irStates[:,6])
    tLatest = max(rfLatest, irLatest)

    # propagate RF states to T = tLatest
    RF_objects = np.zeros((nRF, 4))
    RF_cov = np.zeros((nRF, 3, 3))
    for ii in range(nRF):
        propState, propCov = mods.PropagateECICov(rfStates[ii, 0:7], rfCov[ii], tLatest - rfStates[ii, 6], 2.0)
        RF_objects[ii, 0] = rfIds[ii]
        RF_objects[ii, 1:4] = propState[0:3]
        RF_cov[ii] = propCov[0:3, 0:3]

    # move the RV to top of the list in RF tracks
    rvInd = np.where(rfIds==rvID)[0]
    RF_objects = swapMat(RF_objects, 0, rvInd[0])
    RV_cov = swapMat(RF_cov, 0, rvInd[0])

    # propagate IR states to T = tLatest
    IR_objects = np.zeros((nKVs, nIR, 4))
    IR_cov = np.zeros((nKVs, nIR, 3, 3))
    for jj in range(nKVs):
        for ii in range(nIR):
            propState, propCov = mods.PropagateECICov(irStates[ii, 0:7], irCov[ii], 
                tLatest - irStates[ii, 6], 2.0)
            IR_objects[jj, ii, 0] = irIds[ii]
            IR_objects[jj, ii, 1:4] = propState[0:3]
            IR_cov[jj, ii] = propCov[0:3, 0:3]

    # propagate KV states to T = tLatest
    KV_pos = np.zeros((nKVs, 3))
    for ii in range(nKVs):
        propState = mods.PropagateECI(kvStates[ii, 0:7], 0.0, tLatest - kvStates[ii, 6])
        KV_pos[ii] = propState[0:3]

    # unique IDs
    uniqueIds = np.intersect1d(irIds, rfIds)

    # TOM correlation
    return TomCorrelation(nKVs, uniqueIds.size, RF_objects, IR_objects, RV_cov, IR_cov, KV_pos)

"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""
