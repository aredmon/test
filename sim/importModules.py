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
from generalUtilities.FindPOCA import FindPOCA
from generalUtilities.projection import project_covariance
"""
--------------------------------------------- UTILITY CLASSES ---------------------------------------------
"""
from generalUtilities.Classes import CVObject, KVObjects, TOMObject, SimpleNamespace
"""
--------------------------------------------- CLUSTER METHODS ---------------------------------------------
"""
#from generalUtilities.Cluster import Cluster
from generalUtilities.kMeans import kMeans
from sim.Cluster import Cluster
"""
--------------------------------------------- CREATION METHODS --------------------------------------------
"""
#from generalUtilities.MakeOnboardThreats import MakeOnboardThreats
#from generalUtilities.MakeTOM import MakeTOM
from sim.MakeOnboardThreats import MakeOnboardThreats
from sim.MakeTOM import MakeTOM
"""
-------------------------------------------- PROPAGATE METHODS --------------------------------------------
"""
from generalUtilities.propagate import PropagateECI, PropagateStatesToDivergence, PropagateECICov
from generalUtilities.GaussProblem import GaussProblem
"""
-------------------------------------------- KINEMATIC METHODS --------------------------------------------
"""
#from generalUtilities.KinematicReach import KinematicReach
from sim.KinematicReach import KinematicReach
"""
-------------------------------------- STATISTICAL DISTANCE METHODS ---------------------------------------
"""
from generalUtilities.MahalanobisDist import mahalDist1, mahalDist2, mahalDist3
from generalUtilities.bhattacharyya import bhattacharyya
"""
-------------------------------------------- DEPLOYMENT METHODS -------------------------------------------
"""
from sim.DeployKVsNoTOM import DeployKVsNoTOM
from sim.DeployKVsTOM import DeployKVsTOM
from sim.DispenseKVs import DispenseKVs
"""
-------------------------------------------- ASSIGNMENT METHODS -------------------------------------------
"""
from sim.FinalAssignment import FinalAssignment
from sim.calculate_QV import calculate_qv_single_asset2
"""
------------------------------------------ INITIALIZATION METHODS -----------------------------------------
"""
from sim.InitializeScenario import InitializeScenario
"""
------------------------------------------ DISCRIMINATION METHODS -----------------------------------------
"""
from sim.Discrimination import DiscriminationGround, DiscriminationOnboard
"""
-------------------------------------- MUNKRES ALLOCATTION METHODS ----------------------------------------
"""
from sim.Munkres import munkres
from sim.MunkresPairing import MunkresPairing
from generalUtilities.ModMunkres import ModMunkres2
"""
-------------------------------------- CORRELATION/FUSION METHODS -----------------------------------------
"""
from sim.CorrelateTOM import CorrelateTOM
#from sim.FuseStates import FuseSixStates
"""
-------------------------------------------- GUIDANCE METHODS ---------------------------------------------
"""
from sim.TerminalHoming import TerminalHoming
"""
----------------------------------------------- CONSTANTS -------------------------------------------------
"""
from generalUtilities.config import *
from sim.config import *
"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""
