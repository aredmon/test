"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*   Module Name: mtspofs_ga (Fixed Start Multiple Traveling Salesman Problem Genetic Algorithm)             *
*   Author(s): Joseph Kirk, Mark Lambrecht, Brent McCoy                                                     *
*   Version: 1.0                                                                                            *
*   Date: 01/25/18                                                                                          *
*                                                                                                           *
*       Module          Finds a (near) optimal solution to a variation of the "open" M-TSP by               *
*       Description:    setting up a GA to search for the shortest route (least distance needed             *
*                       for each salesman to travel from the start location to unique individual            *
*                       cities without returning to the starting location)                                  *
*                                                                                                           *
*       Input:                                                                                              *
*               USERCONFIG (dictionary) with zero or more of the following fields:                          *
*                   - XY (float) is an Nx2 matrix of city locations, where N is the number of cities        *
*                   - DMAT (float) is an NxN matrix of city-to-city distances or costs                      *
*                   - NSALESMEN (scalar integer) is the number of salesmen to visit the cities              *
*                   - MINTOUR (scalar integer) is the minimum tour length for any of the                    *
*                       salesmen, NOT including the start point                                             *
*                   - POPSIZE (scalar integer) is the size of the population (should be divisible by 8)     *
*                   - NUMITER (scalar integer) is the number of desired iterations for the algorithm to run *
*                   - SHOWPROG (scalar logical) shows the GA progress if true                               *
*                   - SHOWRESULT (scalar logical) shows the GA results if true                              *
*                   - SHOWWAITBAR (scalar logical) shows a waitbar if true                                  *
*                                                                                                           *
*       Output:                                                                                             *
*               RESULTS (dictionary) with the following fields:                                             *
*                   (in addition to a record of the algorithm configuration)                                *
*                   - OPTROUTE (integer array) is the best route found by the algorithm                     *
*                   - OPTBREAK (integer array) is the list of route break points (these specify the indices *
*                       into the route used to obtain the individual salesman routes)                       *
*                   - MINDIST (scalar float) is the total distance traveled by the salesmen                 *
*                                                                                                           *
*       OA:         Joseph Kirk                                                                             *
*                                                                                                           *
*       Email:      jdkirk630@gmail.com                                                                     *
*                                                                                                           *
*       History:    JK 01 May 2014: version 2.0 release                                                     *
*                                                                                                           *
*************************************************************************************************************
"""
import numpy as np
from generalUtilities.utilities import flipVector, swapElements, slideSet
from generalUtilities.Classes import SimpleNamespace

def rand_breaks(minTour, numPts, numBreaks, cumProb):
    if minTour == 1:     # No constraint on Breaks
        tmpBreaks = np.random.permutation( np.arange(numPts-1) )
        breaks = np.squeeze( np.sort(tmpBreaks[0:numBreaks]) )
    else:                # Force Breaks to be at Least the Minimum Tour Length
        testIndices = np.where(np.random.rand() < cumProb)[0]
        nAdjust = testIndices[0]
        spaces = np.ceil(numBreaks*np.random.rand(nAdjust))
        adjust = np.zeros(numBreaks)
        for kk in range(numBreaks):
            adjust[kk] = np.where(spaces == kk+1)[0].size
        # generate breaks array
        breaks = minTour*np.arange(numBreaks) + np.cumsum(adjust)
    # release results
    return breaks

def mtspofs_ga(pointSet=10*np.random.rand(40,2), distMat=[], nSalesmen=5, minTour=2, populationSize=80, 
        iterationMax=5e3, popGroupSize=8, userConfig={}):
    #----------------------------------------------------------------------------------------------------
    # arbitrarily create a default configuration from the standard inputs in case everything is passed
    # in using key-word arguments.
    #
    # NOTE: any value listed in the default dictionary that is also in the userConfig dictionary will be 
    #       overwritten by what is in the userConfig dictionary if supplied. Therefore, be careful mixing 
    #       function calls with key-word arguments AND a userConfig dictionary
    #----------------------------------------------------------------------------------------------------
    defaultConfig = {'pointSet': pointSet,
                    'distMat': distMat,
                    'nSalesmen': nSalesmen,
                    'minTour': minTour,
                    'populationSize': populationSize,
                    'iterationMax': iterationMax,
                    'popGroupSize': popGroupSize}

    if len(userConfig) != 0:
        configuration = SimpleNamespace(defaultConfig.update(userConfig))
    else:
        configuration = SimpleNamespace(defaultConfig)

    if configuration.populationSize % configuration.popGroupSize != 0:
        # adjust population size to be divisible by populationGroupSize
        newPopSize = configuration.populationSize + (configuration.popGroupSize - 
                configuration.populationSize % configuration.popGroupSize)
        # print warning message to alert operator of the change
        print("WARNING: populationSize {} is not divisible by popGroupSize {}. Padding populationSize to {}"\
                .format(configuration.populationSize, configuration.popGroupSize, newPopSize))
        # save the new populationSize to the configuration dictionary
        configuration.populationSize = newPopSize

    pointSet = configuration.pointSet
    distMat = configuration.distMat
    
    if len(distMat) == 0:
        nPoints = pointSet.shape[0]
        distMat = np.zeros([nPoints, nPoints])
        for rowIndex in range(nPoints):
            for colIndex in range(nPoints):
                distMat[rowIndex, colIndex] = np.linalg.norm( pointSet[rowIndex] - pointSet[colIndex] )
    
    # Verify inputs
    if distMat.shape[0] != pointSet.shape[0] or distMat.shape[1] != pointSet.shape[0]:
        raise IndexError("provided pointSet and distMat variables are incompatible, shape {} cannot"
                + "be combined with shape {}".format(pointSet.shape, distMat.shape))
    else:
        configuration.distMat = distMat

    nMax = pointSet.shape[0] - 1

    # Sanity checks
    nSalesmen = int(configuration.nSalesmen)
    minTour = int(configuration.minTour)
    maxIterations = int(configuration.iterationMax)

    # Initialization for the Route Break Point Selection
    nBreaks = nSalesmen - 1
    DoF = nMax - minTour*nSalesmen
    tmpVec = np.ones(DoF+1)
    for kk in range(1, nBreaks):
        tmpVec = np.cumsum(tmpVec)
    # calculate cumulative probability
    cumProb = np.cumsum(tmpVec)/sum(tmpVec)

    # Initialize the populations
    popRoute = np.zeros([configuration.populationSize, nMax], dtype=int)
    popBreak = np.zeros([configuration.populationSize, nBreaks], dtype=int)
    popRoute[0] = np.arange(nMax) + 1
    popBreak[0] = rand_breaks(minTour, nMax, nBreaks, cumProb)
    for kk in range(1, configuration.populationSize):
        popRoute[kk] = np.random.permutation( np.arange(nMax) ) + 1
        popBreak[kk] = rand_breaks(minTour, nMax, nBreaks, cumProb)

    # Run the genetic algorithm
    globalMin = np.inf
    totalDistance = np.zeros(configuration.populationSize)
    distanceHist = np.zeros(maxIterations)
    tmpPopRoute = np.zeros([configuration.popGroupSize, nMax], dtype=int)
    tmpPopBreak = np.zeros([configuration.popGroupSize, nBreaks], dtype=int)
    #newPopRoute = np.zeros([configuration.populationSize, nMax])
    #newPopBreak = np.zeros([configuration.populationSize, nBreaks])

    for iteration in range(maxIterations):
        if iteration % 100 == 0:
            print("beginning iteration: {}".format(iteration))
        # Evaluate members of the Population
        for member in range(configuration.populationSize):
            distance = 0
            memberRoute = popRoute[member].astype(int)
            memberBreak = popBreak[member].astype(int)
            memberRange = np.array([ np.concatenate(([0], memberBreak+1)), 
                np.concatenate((memberBreak, [nMax])) ], dtype=int)
            for salesman in range(configuration.nSalesmen):
                # find starting distance
                try:
                    distance = distance + configuration.distMat[0, memberRoute[memberRange[0, salesman]]]
                except:
                    print("working on citizen: {}".format(member))
                    print("current iteration: {}".format(iteration))
                    #print("distMat[0] values: {}".format(distMat[0]))
                    #print("memberRange vector: {}".format(memberRange))
                    #print("memberRange variable: {}".format(memberRange[0, salesman]))
                    print("full memberRoute variable: {}".format(memberRoute))
                    print("index from memberRoute: {}".format(memberRoute[memberRange[0, salesman]]))
                # find the distance for the rest of the locations
                for kk in range(memberRange[0, salesman], memberRange[1, salesman]-1):
                    distance = distance + configuration.distMat[memberRoute[kk], memberRoute[kk+1]]
            # store the distance information
            totalDistance[member] = distance

        # Find the best route to the population
        minIndex = totalDistance.argmin()
        minDist = totalDistance[ minIndex ]
        distanceHist[minIndex] = minDist
        if minDist < globalMin:
            globalMin = minDist
            optimalRoute = popRoute[minIndex]
            optimalBreak = popBreak[minIndex]
            popRange = np.array([np.concatenate(([0], optimalBreak+1)), 
                np.concatenate((optimalBreak, [nMax]))], dtype=int)
            for salesman in range(nSalesmen):
                route = np.concatenate(([1], optimalRoute[popRange[0, salesman]:popRange[1, salesman]]))

        # genetic algorithm
        randomOrder = np.random.permutation(np.arange(configuration.populationSize))
        for index in range(configuration.popGroupSize, 
                configuration.populationSize, configuration.popGroupSize):
            # setting up routes
            newRoutes = popRoute[randomOrder[index-configuration.popGroupSize:index]]
            newBreaks = popBreak[randomOrder[index-configuration.popGroupSize:index]]
            distances = totalDistance[randomOrder[index-configuration.popGroupSize:index]]
            try:
                bestIndex = distances.argmin()
            except ValueError:
                print("current population set: {} \ndistances vector: {}".format(index, distances))
                pass
            bestOfRoute = newRoutes[bestIndex]
            bestOfBreak = newBreaks[bestIndex]
            #routeInsertionPoints = np.squeeze( np.sort(np.round((nMax-1)*np.random.rand(1,2))) ).astype(int)
            routeInsertionPoints = np.squeeze( np.sort(np.random.randint(nMax, size=(2))) )
            # catch transient bug of a size 0 subset
            if np.diff(routeInsertionPoints)[0] == 0:
                try:
                    #print("routeInsertionPoints: {}".format(routeInsertionPoints))
                    routeInsertionPoints[0] = np.random.randint(routeInsertionPoints[0])
                    routeInsertionPoints[1] = np.random.randint(routeInsertionPoints[1], nMax)
                except ValueError:
                    seperation = int( nMax / 3 )
                    routeInsertionPoints[0] = seperation
                    routeInsertionPoints[1] = seperation*2

            for kk in range(configuration.popGroupSize):
                tmpPopRoute[kk] = bestOfRoute
                tmpPopBreak[kk] = bestOfBreak
                if np.any(np.isnan(tmpPopRoute[kk])):
                    print("tmpPopRoute[{}] contains nans before mutation: {}".format(kk))
                # mutate solution
                if kk == 1:
                    # flip subSet of route stops
                    tmpPopRoute[kk] = flipVector(tmpPopRoute[kk], routeInsertionPoints)
                elif kk == 2:
                    # swap to route stops for one another
                    try:
                        tmpPopRoute[kk] = swapElements(tmpPopRoute[kk], routeInsertionPoints)
                    except ValueError:
                        print("routeInsertionPoints: {}".format(routeInsertionPoints))
                elif kk == 3:
                    # roll a subset of route stops back 1 position (cyclic)
                    tmpPopRoute[kk] = slideSet(tmpPopRoute[kk], routeInsertionPoints, shift=-1)
                elif kk == 4:
                    # adjust the breaks
                    tmpPopBreak[kk] = rand_breaks(minTour, nMax, nBreaks, cumProb)
                elif kk == 5:
                    # flip subSet of route stops
                    tmpPopRoute[kk] = flipVector(tmpPopRoute[kk], routeInsertionPoints)
                    # adjust the breaks
                    tmpPopBreak[kk] = rand_breaks(minTour, nMax, nBreaks, cumProb)
                elif kk == 6:
                    # swap to route stops for one another
                    tmpPopRoute[kk] = swapElements(tmpPopRoute[kk], routeInsertionPoints)
                    # adjust the breaks
                    tmpPopBreak[kk] = rand_breaks(minTour, nMax, nBreaks, cumProb)
                elif kk == 7:
                    # roll a subset of route stops back 1 position (cyclic)
                    tmpPopRoute[kk] = slideSet(tmpPopRoute[kk], routeInsertionPoints, shift=-1)
                    # adjust the breaks
                    tmpPopBreak[kk] = rand_breaks(minTour, nMax, nBreaks, cumProb)
                else:
                    pass
                if np.any(np.isnan(tmpPopRoute[kk])):
                    print("tmpPopRoute[{}] contains nans after mutation".format(kk))
                    print("routeInsertionPoints: {}".format(routeInsertionPoints))
            # overwrite orifinal popRoutes with mutated routes
            popRoute[index-configuration.popGroupSize:index, :] = tmpPopRoute
            popBreak[index-configuration.popGroupSize:index, :] = tmpPopBreak
        # continue with the next iteration
    # return final output
    results = configuration.toDict()
    results.update({"optimalRoute": optimalRoute, "optimalBreak": optimalBreak, "minDistance": minDist})
    return results

"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""
