"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*   ModuleName: DeployKVsNoTOM                                                                              *
*   Author(s):  M. A. Lambrecht, P. B. McCoy                                                                *
*   Version:    1.0                                                                                         *
*   Date:       01/02/18                                                                                    *
*                                                                                                           *
*       Module        This function performs the initial KV assignments in order to                         *
*       Description:  perform zone defense of the engagement volume.  It is called                          *
*                     once at or shortly after dispense.  This function assumes that                        *
*                     no TOM is available, and that acquisition has not yet been                            *
*                     performed.                                                                            *
*                                                                                                           *
*       Algorithm:    1) Compute the center of the EV at POCA (using the average                            *
*                        threat state)                                                                      *
*                     2) Build the CV coordinate system (x is relative velocity, y and                      *
*                        z are perpendicular to x and each other)                                           *
*                     3) Build the KV PIP states in the CV-frame                                            *
*                     4) Transform the PIP states to the ECI frame                                          *
*                     5) Divert the KVs to their PIP states                                                 *
*                                                                                                           *
*       Inputs:       cvState      - 7x1 CV state (t,x,y,z,vx,vy,vz), [s,m,m/s]                             *
*                     cvFuel       - 1x1 available delta-V on the CV [m/s]                                  *
*                     kvStates     - #KVsx7 KV states (t,x,y,z,vx,vy,vz), [s,m,m/s]                         *
*                     kvFuel       - #KVsx1 available delta-V on the KVs [m/s]                              *
*                     threatStates - #Threatsx7 threat states (t,x,y,z,vx,vy,vz),                           *
*                                    [s,m,m/s]                                                              *
*                     rPOCA        - 3x1 postion of CV POCA [m]                                             *
*                     tPOCA        - (Scalar) Estimated time of intercept, s                                *
*                     rvID         - (Scalar) index of RV in threatStates                                   *
*                     tgtRadius    - (Scalar) radius of threat complex, m                                   *
*                     aH1          - axis handle for plotting                                               *
*                                                                                                           *
*       Outputs:      cvState      - 7x1 CV state (t,x,y,z,vx,vy,vz), [s,m,m/s]                             *
*                     cvFuel       - (Scalar) available delta-V on the CV [m/s]                             *
*                     kvStates     - #KVsx7 KV states (t,x,y,z,vx,vy,vz), [s,m,m/s]                         *
*                     kvFuel       - #KVsx1 available delta-V on the KVs [m/s]                              *
*                                                                                                           *
*       Calls:        Util_GaussProblem                                                                     *
*                     Util_PropagateECI                                                                     *
*                     ShowBubble                                                                            *
*                                                                                                           *
*       Requires:     SAPs_Null.json                                                                        *
*                                                                                                           *
*       History:      M.A. Lambrecht, 13 Dec 2017 - Initial version                                         *
*                                                                                                           *
*************************************************************************************************************
"""
import os
import numpy as np
import importModules as mods

def DeployKVsNoTOM(cvState, cvFuel, kvStates, kvFuel, threatStates, rPOCA, tPOCA, rvID, thtRadius, aH1):
    # number of KVs we are working with (assumed on the same ring)
    nKVs = mods.SAPs.KV_NUM_KVS
    tNOW = float(threatStates[0])
    tGO = tPOCA - tNOW

    # build a coordinate system with the CV LOS to the POCA as the x-axis (call it the G frame)
    # and the direction cosine matrix between the G and the ECI frame
    xG = np.divide(rPOCA - cvState[0:3], mods.vecNorm(rPOCA - cvState[0:3]) )
    zG = np.divide( np.cross(cvState[0:3], xG, axis=0), 
            mods.vecNorm(np.cross(cvState[0:3], xG, axis=0)) )
    yG = np.divide( np.cross(zG, xG, axis=0), 
            mods.vecNorm(np.cross(zG, xG, axis=0)) )

    t_G2ECI = np.hstack((xG.reshape(3,1), yG.reshape(3,1), zG.reshape(3,1)))

    # build the KV PIP states in the G-frame
    theta = 2*np.pi / nKVs
    rotMat = np.array([[np.cos(theta), -np.sin(theta)], 
        [np.sin(theta), np.cos(theta)]])
    xxG = np.zeros((nKVs, 1))
    yyG = np.zeros((nKVs, 1))
    zzG = np.zeros((nKVs, 1))

    yyG[0, 0] = mods.SAPs.WTA_COV_RADIUS_FRAC * thtRadius

    for iKV in range(1, nKVs):
        tmp = np.array(((yyG[iKV-1, 0]), (zzG[iKV-1, 0])))
        v = np.dot( rotMat, tmp)
        yyG[iKV] = v[0]
        zzG[iKv] = v[1]

    rrG = np.hstack((xxG, yyG, zzG))

    # build the KV PIP states in the ECI-frame and divert the KVs to their positions
    kvPIP = np.zeros([nKVs, 3])
    kvKillRadius = np.zeros([nKVS, 1])
    kvPOCA = np.zeros([nKVs, 3])
    for iKV in range(nKVs):
        kvPIP[iKV] = np.reshape(rPOCA, (1,3)) + np.dot(t_G2ECI, rrG[iKV])
        kv, poca = mods.GaussProblem(kvStates[iKV, 0:3], kvPIP[iKV], tGo)
        dV = kv[3:6] - kvStates[iKV, 3:6]
        kvStates[iKV, 3:6] = kvStates[iKV, 3:6] + dV
        kvFuel[iKV] = kvFuel[iKV] - mods.vecNorm(dV)
        kvKillRadius[iKV] = ( kvFuel[iKV] - mods.SAPs.KV_FUEL_RESERVE ) * mods.SAPs.CV_FINAL_ASSGN_TGO
        kvPOCA[iKV] = poca

    return cvState, cvFuel, kvStates, kvFuel

##### MATLAB PLOTTING CODE #####
#    # plot results
#    if mods.SAPs.MDL_PLOT_ON
#        for iKV = 1 : size(kvStates, 1)
#            ShowBubble(kvPOCA(iKV, 1:3), kvKillRadius(iKV), [0 1 0]);
#            text(kvPOCA(iKV,1), kvPOCA(iKV,2), kvPOCA(iKV,3), num2str(iKV), 'color', 'k', 'Parent', aH1);
#            plot3(kvPOCA(iKV,1), kvPOCA(iKV,2), kvPOCA(iKV,3), 'g>', 'MarkerSize', 10, 
#                    'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'k', 'Parent', aH1);
#        end % for iKV = 1 : size(kvStates, 1)
#        threatStatesPOCA = zeros(size(threatStates));
#        for iTht = 1 : size(threatStates, 1)
#            threatStatesPOCA(iTht, :) = Util_mods.PropagateECI(threatStates(iTht,:)', 0.0, 
#            tPOCA - threatStates(iTht, 7));
#        end
#        com = sum(threatStatesPOCA, 1) ./ size(threatStatesPOCA, 1);
#        plot3(com(1), com(2), com(3), 'gs', 'MarkerSize', 10, 'MarkerFaceColor', 'g', 
#            'MarkerEdgeColor', 'k', 'Parent', aH1);
#        plot3(threatStatesPOCA(:,1), threatStatesPOCA(:,2), threatStatesPOCA(:,3), 'rx', 
#            'MarkerSize', 4, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'r', 'Parent', aH1);
#        plot3(threatStatesPOCA(rvID,1), threatStatesPOCA(rvID,2), threatStatesPOCA(rvID,3), 'r>', 
#            'MarkerSize', 8, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k', 'Parent', aH1);
#        axis(aH1, 'equal');
#        drawnow;
#    end % if mods.SAPs.PLOT_ON
"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""
