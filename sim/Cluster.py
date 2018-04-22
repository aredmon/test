"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*   Module Name: Cluster                                                                                    *
*   Author(s): Mark Lambrecht, Brent McCoy                                                                  *
*   Version: 1.0                                                                                            *
*   Date: 01/08/18                                                                                          *
*                                                                                                           *
*       Module        This function forms clusters from threat objects at the time of                       *
*       Description:  intercept.  Cluster size is determined by KV kinematic reach.                         *
*                                                                                                           *
*       Algorithm:    1) Propagate all threats to time = tPOCA                                              *
*                     3) Find the center-of-mass of all input points                                        *
*                     2) Loop over all input data points                                                    *
*                        a) Find the point furthest from the center                                         *
*                        b) Determine if this point is kinematically feasible for a KV                      *
*                           to reach                                                                        *
*                        c) Compute the kill radius for a KV reaching this point                            *
*                        d) Determine all points within the radius of this point                            *
*                        e) Refine the points by adjusting the center (keeping the far                      *
*                           point within reach                                                              *
*                        f) Remove all used objects from the list and go to step 2                          *
*                                                                                                           *
*       Inputs:       cvState      - 7x1 CV state (x,y,z,vx,vy,vz,t), [m,m/s,s]                             *
*                     kvFuel       - #KVsx1 available delta-V on the KVs [m/s]                              *
*                     threatStates - #Threatsx7 threat states (x,y,z,vx,vy,vz,t),                           *
*                                    [m,m/s,s]                                                              *
*                     threatIDs    - 1x#Threats vector of threat IDs                                        *
*                     tPOCA        - (Scalar) Approximate time of intercept, s                              *
*                     aH1          - axis handle for plotting                                               *
*                                                                                                           *
*       Outputs:      clusters     - 1x#clusters structure:                                                 *
*                                       ctr    - 1x3 cluster center position [m]                            *
*                                       pts    - #ptsx3 cluster points positions [m]                        *
*                                       states - #ptsx7 point states(x,y,z,vx,vy,vz,t)                      *
*                                                [m, m/s, s]                                                *
*                                       ids    - 1x#pts IDs of points in cluster                            *
*                                       tPOCA  - 1x3pts time of intercept for pts                           *
*                                       kr     - 1x1 delta-V to reach cluster                               *
*                                                                                                           *
*       Calls:        SAPs_Null                                                                             *
*                     Util_PropagateStates                                                                  *
*                     FarPoint (Local)                                                                      *
*                     Util_FindPOCA                                                                         *
*                     Util_GaussProblem                                                                     *
*                     ReCenter (Local)                                                                      *
*                                                                                                           *
*       OA:           Mark Lambrecht                                                                        *
*                                                                                                           *
*       History:      Mark Lambrecht, 10 Aug 2017 - Initial version                                         *
*                                                                                                           *
*************************************************************************************************************
"""
import os
import numpy as np
import importModules as mods

"""
----------------------------------------------------------------------------------------------
    Inputs:       distPts      - #ptsx3 list of point positions
                  distCtr      - 1x3 current center of points
                  distMax      - Maximum distance away from center
    
    Outputs:      newCtr       - 1x3 updated center of points
                  nearestId    - ID of point closest to center
----------------------------------------------------------------------------------------------
"""
def ReCenter(distPts, distCtr, distMax):
    # calculate distance between each point and desired center
    difference = np.subtract(distPts, distCtr)
    distance = mods.vecNorm(difference, axis=1)
    # get list of measurements that are within distance, dist, from center
    ids = np.where(distance < distMax)[0]
    if ids.size > 0:
        distRed = distPts[ ids ]
        # compute new center based on arithmetic mean of reduced values
        newCtr = np.mean(distRed, axis=0)
        # find the point closest to the new center from the reduced list
        newDist = mods.vecNorm( np.subtract(distRed, newCtr), axis=1 )
        nearestId = np.argmin(newDist)
    else:
        print("Empty list in recenter??")
        distRed = np.empty(0)
        nearestId = -1

    return newCtr, nearestId

"""
----------------------------------------------------------------------------------------------
    Inputs:       distPts       - #pointsx3 positions of each point considered
                  distCtr       - 1x3 position of center of points
                  distMax       - 1x1 Maximum distance used
    
    Outputs:      newCtr        - 1x3 coordinates of farthest point from center
                  farthestId    - 1x1 ID of point
                  farDist       - 1x1 distance of farthest point from center
----------------------------------------------------------------------------------------------
"""
def FarPoint(distPts, distCtr, distMax=1e17):
    # calculate distance between each point and desired center
    difference = np.subtract(distPts, distCtr)
    distance = mods.vecNorm(difference, axis=1)
    # get list of measurements that are within distance, distMax, from center
    ids = np.where(distance < distMax)[0]
    if ids.size > 0:
        distRed = distPts[ids]
        newDist = distance[ids]
        # find the point furthest from distCtr in the reduced list
        farthestId = np.argmax(newDist)
        newCtr = distRed[ farthestId ]
        farDist = newDist[ farthestId ]
    else:
        print("No points within distance {} of point {}".format(distMax, distCtr))
        farthestId = np.argmax(distance)
        newCtr = distPts[ farthestId ]
        farDist = distance[ farthestId ]

    return newCtr, farthestId, farDist

"""
----------------------------------------------------------------------------------------------
    Inputs:       cvState      - 7x1 CV state (x,y,z,vx,vy,vz,t), [m,m/s,s]
                  kvFuel       - #KVsx1 available delta-V on the KVs [m/s]
                  threatStates - #Threatsx7 threat states (x,y,z,vx,vy,vz,t),
                                 [m,m/s,s]
                  threatIDs    - 1x#Threats vector of global threat IDs
                  tPOCA        - (Scalar) Approximate time of intercept, s
                  aH1          - axis handle for plotting
    
    Outputs:      clusters     - 1x#clusters list of dictionaries:
                                    ctr    - 1x3 cluster center position [m]
                                    pts    - #ptsx3 cluster points positions [m]
                                    states - #ptsx7 point states(x,y,z,vx,vy,vz,t)
                                             [m, m/s, s]
                                    ids    - 1x#pts IDs of points in cluster
                                    tPOCA  - 1x3pts time of intercept for pts
                                    kr     - 1x1 delta-V to reach cluster
----------------------------------------------------------------------------------------------
"""
def Cluster(cvState, kvFuel, threatStates, threatIds, tPOCA, aH1):
    # propagate the threats to tPOCA
    nThreats = threatStates.shape[0]
    threatStatesPOCA = np.zeros(threatStates.shape)
    iPOCA = 0
    for thrtState in threatStates:
        threatStatesPOCA[iPOCA] = mods.PropagateECI(thrtState, 0.0, tPOCA - thrtState[-1])
        iPOCA += 1

    # find the cloud center and separate the positions at POCA for clustering
    dVec = threatStatesPOCA[:, 0:3]
    center = np.mean( dVec, axis=0 )
    centerSave = center.copy()

    # perform clustering
    tGo = tPOCA - cvState[6]
    redThreatStates = threatStates.copy()
    delIds = np.empty(0, dtype=int)
    clusterList = []
    while dVec.size > 0:
        cluster = {}
        # find the object furthest from centroid of object map
        dCnt, farId, _ = FarPoint(dVec, center)
        # compute the required delta-V for a "virgin" KV to fly there
        tPOCAt, rPOCAt, _ = mods.FindPOCA( cvState,  redThreatStates[farId])
        if mods.vecNorm(cvState[0:3]) == 0 or mods.vecNorm(rPOCAt) == 0:
            print("Trying to use GaussProblem on an origin vector.")
            print("pos1: {}, pos2: {}".format(cvState[0:3], rPOCAt))
            cv = mods.PropagateECI(cvState, 0.0, tPOCA - cvState[6])
        else:
            cv, _ = mods.GaussProblem(cvState[0:3], rPOCAt, tPOCAt - cvState[6])
        # compute the Delta-V required to get there for a KV
        dV = mods.vecNorm( cv[3:6] - cvState[3:6] )
        # if the required delta-V is too large, get rid of this threat from the list and start 
        # over. The assumption is that KVs have just been dispensed or are not yet dispensed. 
        # KVs have approximately the same state as the CV, and no fuel has been burned.
        dVAfterDivert = kvFuel[0] - dV - mods.SAPs.KV_FUEL_RESERVE
        tBurn = dV / mods.SAPs.KV_MAX_ACC
        # remove the threats from the threat list that are too far away
        if dVAfterDivert <= 0 or tBurn >= tGo:
            # keep track of which global ids were removed
            delIds = np.append(delIds, threatIds[ farId ])
            # remove non-viable threat from consideration
            redThreatStates = np.delete(redThreatStates, farId, 0)
            threatStatesPOCA = np.delete(threatStatesPOCA, farId, 0)
            threatIds = np.delete(threatIds, farId, 0)
            dVec = np.delete(dVec, farId, 0)
            # skip remaining code and re-evaluate loop on the next point
            continue
        # compute the killRadius on the qualified KV
        killRadius = dVAfterDivert * mods.SAPs.CV_FINAL_ASSGN_TGO
        # compute the center of all threat objects within 'killRadius' of this object
        # and iterate for a better fit
        dCnt, _ = ReCenter(dVec, dCnt, killRadius)
        diffCtr = np.zeros(3)
        while mods.vecNorm( dCnt - diffCtr ) > 0.1 and dVec.size > 1:
            # Re-center
            diffCtr = dCnt.copy()
            dCnt, _ = ReCenter(dVec, dCnt, killRadius)
        # Save as the cluster center location
        cluster.update({"ctr": dCnt})
        # Remove all the objects within 'killRadius' of this object from the object map
        killIds = np.empty(0, dtype=int)
        noKillIds = np.empty(0, dtype=int)
        # calculate distance between each point and the object location
        difference = np.subtract(dVec, dCnt)
        distance = mods.vecNorm(difference, axis=1)
        indx = 0
        for dist in distance:
            if dist >= killRadius:
                noKillIds = np.append(noKillIds, indx)
            else:
                killIds = np.append(killIds, indx)
            indx += 1

        # create clusters from the objects corresponding to killIds
        if killIds.size > 0:
            cluster.update({"pts": dVec[ killIds ]})
            cluster.update({"states": threatStatesPOCA[ killIds ]})
            cluster.update({"ids": threatIds[ killIds ]})
            cluster.update({"tPOCA": tPOCA, "kr": killRadius})
            # add current cluster as a SimpleNamespace object to clusterList
            clusterList.append( mods.SimpleNamespace(cluster) )

            # update dVec, threatIds, redThreatStates, threatStatesPOCA 
            # corresponding objects in noKillIds
            dVec = np.delete(dVec, killIds, 0)
            redThreatStates = np.delete(redThreatStates, killIds, 0)
            threatStatesPOCA = np.delete(threatStatesPOCA, killIds, 0)
            threatIds = np.delete(threatIds, killIds, 0)
        else:
            break
    # return clusters to be used outside of the code
    return clusterList

#
## Plot results
#if SAPs.MDL_PLOT_ON
#    nClst = size(clusters, 2);
#    clrs = ['g'; 'b'; 'c'; 'm'; 'k';];
#    mrks = ['+'; 'o'; '*'; 'x'; 's'; 'd'; '^'; 'v'; '>'; '<'; 'p'; 'h';];
#    iClr = 1;
#    iMrk = 1;
#    for iClst = 1 : nClst
#        str = sprintf('%s%s', clrs(iClr,:), mrks(iMrk,:));
#        iClr = iClr + 1; if iClr > length(clrs), iClr = 1; end
#        iMrk = iMrk + 1; if iMrk > length(mrks), iMrk = 1; end
#        plot3(clusters(iClst).pts(:,1),...
#            clusters(iClst).pts(:,2), clusters(iClst).pts(:,3), str, ...
#            'Parent', aH1);
#        ShowBubble(clusters(iClst).ctr, clusters(iClst).kr, [0 0 1], aH1);
#        text(clusters(iClst).ctr(1), clusters(iClst).ctr(2), ...
#            clusters(iClst).ctr(3), num2str(iClst), 'color', 'k', ...
#            'Parent', aH1);
#    end
#    plot3(threatStatesPOCA(delIDs,1), threatStatesPOCA(delIDs,2), ...
#        threatStatesPOCA(delIDs,3), 'r>', 'MarkerSize', 10, ...
#        'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k', ...
#        'Parent', aH1);
#    plot3(centerSave(1), centerSave(2), centerSave(3), 'gs', ...
#        'MarkerSize', 10, 'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'k', ...
#        'Parent', aH1);
#    axis(aH1, 'equal');
#    drawnow;

"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""
