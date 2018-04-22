"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*   Module Name: importModules.py                                                                           *
*   Author(s): Brent McCoy                                                                                  *
*   Version: 1.0                                                                                            *
*   Date: 12/20/17                                                                                          *
*                                                                                                           *
*       Module Description:                                                                                 *
*           Universal import module used to organize and coordinate the other modules called in this        *
*           package.                                                                                        *
*************************************************************************************************************
"""
import os, sys
parentDir = os.path.dirname(os.path.realpath(os.getcwd()))
sys.path.append(parentDir)
"""
--------------------------------------------- UTILITY METHODS ---------------------------------------------
"""
from generalUtilities.utilities import yaw_scene, pitch_scene, UniformRandRange, AngleBetweenVecs
from generalUtilities.utilities import vecNorm, InterpTraj, AtmosphereQ, EarthAltitude, Gravity
from generalUtilities.utilities import findMode, removeBias, removeBias2, zipArray
from generalUtilities.utilities import flipVector, swapElements, slideSet
from generalUtilities.FindPOCA import FindPOCA
from generalUtilities.projection import project_covariance
"""
--------------------------------------------- GENETIC ALFORITHMS ------------------------------------------
"""
from generalUtilities.mtspofs_ga import mtspofs_ga
#"""
#--------------------------------------------- UTILITY CLASSES ---------------------------------------------
#"""
#from generalUtilities.Classes import SimpleNamespace, jsonData, jsonInfo
"""
--------------------------------------------- CLUSTER METHODS ---------------------------------------------
"""
from generalUtilities.Cluster import Cluster
from generalUtilities.kMeans import kMeans
"""
--------------------------------------------- CREATION METHODS --------------------------------------------
"""
from generalUtilities.MakeOnboardThreats import MakeOnboardThreats
from generalUtilities.MakeTOM import MakeTOM
"""
-------------------------------------------- PROPAGATE METHODS --------------------------------------------
"""
from generalUtilities.propagate import PropagateECI, PropagateStatesToDivergence, PropagateECICov
from generalUtilities.GaussProblem import GaussProblem
"""
-------------------------------------------- KINEMATIC METHODS --------------------------------------------
"""
from generalUtilities.KinematicReach import KinematicReach
"""
-------------------------------------- STATISTICAL DISTANCE METHODS ---------------------------------------
"""
from generalUtilities.MahalanobisDist import mahalDist1, mahalDist2, mahalDist3
from generalUtilities.bhattacharyya import bhattacharyya
"""
------------------------------------------ INITIALIZATION METHODS -----------------------------------------
"""
#from sensor_scehduler.InitializeScenario import InitializeScenario
"""
----------------------------------------------- CONSTANTS -------------------------------------------------
"""
from generalUtilities.config import *
"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""
