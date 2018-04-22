"""
****************************************************
Module Name: Cost_Matrix
Author(s): Larry Gariepy
Version: 1.0
Date: 12/23/2015

    Module Description:
    	A collection of functions that can be used to build cost matrices for input into the Munkres optimal assignment algorithm 
    	(see Munkres.py)
    Class Descriptions:
       -Class 1: Cost_Matrix
	Important Function Descriptions:
		-build_pos_cost_matrix_MT():
				Build a cost matrix for assigning measurements to tracks, using only position information
					inputs: 1) measurements - list of new measurements
							2) tracks - list of existing tracks
					outputs: cost matrix
		-build_pos_vel_cost_matrix_MT():
				Build a cost matrix for assigning measurements to tracks, using position and velocity information
					inputs: 1) measurements - list of new measurements
							2) tracks - list of existing tracks
					outputs: cost matrix
		-build_pos_cost_matrix_DT():
				Build a cost matrix for assigning detections to tracks, using position information only
					inputs: 1) measurements - list of new measurements
							2) tracks - list of existing tracks
					outputs: cost matrix

****************************************************
"""
import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG,datefmt='%m/%d/%Y %I:%M:%S %p')
import numpy as np
import SAPS
import math
from Detection import Detection
from Measurement import Measurement
import random as Rand	

"""
****************************************************
Cost_Matrix.py Module

The Cost_Matrix module implements a number of different schemes for calculating a cost matrix to be used in an assignment problem.
Since building cost matrices using Measurement objects is fundamentally different than using Detection objects, I have adopted a 
suffix naming convention to indicate which type of objects are being used in that particular cost function.  The cost functions
that only use position values are simpler, but less accurate than cost functions that incorporate velocity values.  However,
since velocity values are not always available, there must be position-only cost functions available.
"""

# The suffix "MT" stands for (M)easurements vs. (T)racks
def build_pos_cost_matrix_MT(measurements, tracks):
	"""build a cost matrix using only the position values of the measurements and tracks
       NOTE: this method should work for measurements vs. measurements as well"""
	
	logging.debug("Building cost matrix")
	
	## first, get the time stamps of all the measurements and tracks, so that a common time 
	## can be determined for association - in this case, the most current track or measurement time
	## is used
	mtimes = [measurement.get_tvalid() for measurement in measurements]
	ttimes = [track.get_tvalid() for track in tracks]
	common_time = max(max(mtimes), max(ttimes))
	
	# Next, get all the track states and all the measurement states at the common reference time
	track_id_list = []
	track_pos_list = []
	for track in tracks:
		track_id_list.append(track.get_id())
		pos,vel = track.get_state(common_time)
		track_pos_list.append(pos)
	
	meas_pos_list = []
	for measurement in measurements:
		pos,vel = measurement.get_state(common_time)
		meas_pos_list.append(pos)
	
	## Build the cost matrix using the squared distances between the corresponding pairs of states
	num_meas = len(measurements)
	num_tracks = len(tracks)
	cost_matrix = np.zeros( (num_meas, num_tracks) )
	for i in range(num_meas):
		for j in range(num_tracks):
			cost_matrix[i][j] = np.linalg.norm(meas_pos_list[i] - track_pos_list[j],2)**2

	return cost_matrix

# The suffix "MT" stands for (M)easurements vs. (T)racks
def build_pos_vel_cost_matrix_MT(measurements, tracks):
	## TODO: not finished
	raise Error("Not implemented yet")
	"""Build a cost matrix using the position and velocity values of the measurements and tracks.
       This cost function is tuneable, because the relative weight of the position vs. velocity
       component of the cost computation can be adjusted.
       NOTE: this method should work for measurements vs. measurements as well"""
       
	(weight_pos, weight_vel) = COST_MATRIX_WEIGHTS_POS_VEL
	cos_vel_threshold = math.cos(SAPS.VELOCITY_ANGLE_THRESHOLD * np.pi/180.0)
    
	## first, get the time stamps of all the measurements and tracks, so that a common time 
	## can be determined for association - in this case, the most current track or measurement time
	## is used
	mtimes = [measurement.get_tvalid() for measurement in measurements]
	ttimes = [track.get_tvalid() for track in tracks]
	common_time = max(max(mtimes), max(ttimes))
	
	# Next, get all the track states and all the measurement states at the common reference time
	track_id_list = []
	track_pos_list = []
	track_vel_list = []
	for track in tracks:
		track_id_list.append(track.get_id())
		pos,vel = track.get_state(common_time)
		track_pos_list.append(pos)
	
	meas_pos_list = []
	meas_vel_list = []
	for measurement in measurements:
		pos,vel = measurement.get_state(common_time)
		meas_pos_list.append(pos)
	
	num_meas = len(measurements)
	num_tracks = len(tracks)
	cost_matrix = np.zeros( (num_meas, num_tracks) )
	for i in range(num_meas):
		for j in range(num_tracks):
			cost_matrix[i][j] = np.linalg.norm(meas_pos_list[i] - track_pos_list[j],2)**2

	return cost_matrix

# the suffix "DT" stands for (D)etections vs. (T)racks
def build_pos_cost_matrix_DT(detections, tracks):
	"""Build a cost matrix using the position values of a set of detections
       vs. the current tracks.  """

	## TODO: need to propagate tracks to a common time to match detections
	## TODO: need to add a buffer based on the range_rate values of the detections if there is a "significant" time difference between the two lists       

	det_pos_list = [] # detection position list
	for detection in detections:
		det_pos_list.append(detection.get_position())
	
	trk_pos_list = [] # track position list
	for track in tracks:
		trk_pos_list.append(track.get_position())  ## TODO: this needs to be a state estimate, not the initialized position
	
	num_det = len(det_pos_list)
	num_trk = len(trk_pos_list)
	cost_matrix = np.zeros( (num_det, num_trk) )
	for i in range(num_det):
		for j in range(num_trk):
			data = tracks[j].get_state( np.array([detections[i].get_tvalid(), ]), use_measurements = True)
			trk_pos = data[0]
			#print("trk_pos = ", trk_pos)
			#print("comparing trk state %s: @ %0.1f [%0.2f,%0.2f,%0.2f]\n" % (tracks[j].get_id(), detections[i].get_tvalid(), trk_pos[0][0], trk_pos[0][1], trk_pos[0][2]))
			#print("TO detection %0.1f, %0.2f, %0.2f, %0.2f" % (detections[i].get_tvalid(), det_pos_list[i][0], det_pos_list[i][1], det_pos_list[i][2]))
			cost_matrix[i][j] = np.linalg.norm(det_pos_list[i] - trk_pos[0],2)**2

	return cost_matrix


if __name__ == "__main__":
	det_list1 = []
	det_list2 = []
	det_list1.append(Detection(position = np.array([100*Rand.random(), 100*Rand.random(), 100*Rand.random()]), range_rate = 1.0, tValid = 1.0, sensorID = "sensor1"))
	det_list1.append(Detection(position = np.array([100*Rand.random(), 100*Rand.random(), 100*Rand.random()]), range_rate = 5.0, tValid = 1.0, sensorID = "sensor1"))
	det_list1.append(Detection(position = np.array([100*Rand.random(), 100*Rand.random(), 100*Rand.random()]), range_rate = 10.0, tValid = 1.0, sensorID = "sensor1"))
	det_list2.append(Detection(position = np.array([100*Rand.random(), 100*Rand.random(), 100*Rand.random()]), range_rate = 1.0, tValid = 1.0, sensorID = "sensor2"))
	det_list2.append(Detection(position = np.array([100*Rand.random(), 100*Rand.random(), 100*Rand.random()]), range_rate = -5.0, tValid = 1.0, sensorID = "sensor2"))
	det_list2.append(Detection(position = np.array([100*Rand.random(), 100*Rand.random(), 100*Rand.random()]), range_rate = -20.0, tValid = 1.0, sensorID = "sensor2"))
	# This set of inputs doesn't make sense for the cost matrix any more...building a cost matrix always involves tracks now
# 	print("cost_matrix = ", build_pos_cost_matrix_DT(det_list1, det_list2))
	print("det_list1 = ",det_list1)
	print("det_list2 = ",det_list2)
	
	measurements = []
	measurements.append(Measurement(position = np.array([100*Rand.random(), 100*Rand.random(), 100*Rand.random()]),
	                    velocity = np.array([100*Rand.random(), 100*Rand.random(), 100*Rand.random()]),
	                    tValid = 0,
	                    sensorID = "sensor1"))
	measurements.append(Measurement(position = np.array([100*Rand.random(), 100*Rand.random(), 100*Rand.random()]),
	                    velocity = np.array([100*Rand.random(), 100*Rand.random(), 100*Rand.random()]),
	                    tValid = 0,
	                    sensorID = "sensor2"))
	measurements.append(Measurement(position = np.array([100*Rand.random(), 100*Rand.random(), 100*Rand.random()]),
	                    velocity = np.array([100*Rand.random(), 100*Rand.random(), 100*Rand.random()]),
	                    tValid = 0,
	                    sensorID = "sensor1"))
	measurements.append(Measurement(position = np.array([100*Rand.random(), 100*Rand.random(), 100*Rand.random()]),
	                    velocity = np.array([100*Rand.random(), 100*Rand.random(), 100*Rand.random()]),
	                    tValid = 0,
	                    sensorID = "sensor2"))
	measurements.append(Measurement(position = np.array([100*Rand.random(), 100*Rand.random(), 100*Rand.random()]),
	                    velocity = np.array([100*Rand.random(), 100*Rand.random(), 100*Rand.random()]),
	                    tValid = 0,
	                    sensorID = "sensor3"))
	measurements.append(Measurement(position = np.array([100*Rand.random(), 100*Rand.random(), 100*Rand.random()]),
	                    velocity = np.array([100*Rand.random(), 100*Rand.random(), 100*Rand.random()]),
	                    tValid = 0,
	                    sensorID = "sensor1"))
	measurements.append(Measurement(position = np.array([100*Rand.random(), 100*Rand.random(), 100*Rand.random()]),
	                    velocity = np.array([100*Rand.random(), 100*Rand.random(), 100*Rand.random()]),
	                    tValid = 1,
	                    sensorID = "sensor2"))
	measurements.append(Measurement(position = np.array([100*Rand.random(), 100*Rand.random(), 100*Rand.random()]),
	                    velocity = np.array([100*Rand.random(), 100*Rand.random(), 100*Rand.random()]),
	                    tValid = 2,
	                    sensorID = "sensor3"))
	measurements.append(Measurement(position = np.array([100*Rand.random(), 100*Rand.random(), 100*Rand.random()]),
	                    velocity = np.array([100*Rand.random(), 100*Rand.random(), 100*Rand.random()]),
	                    tValid = 3,
	                    sensorID = "sensor4"))
	measurements.append(Measurement(position = np.array([100*Rand.random(), 100*Rand.random(), 100*Rand.random()]),
	                    velocity = np.array([100*Rand.random(), 100*Rand.random(), 100*Rand.random()]),
	                    tValid = 4,
	                    sensorID = "sensor1"))
	measurements.append(Measurement(position = np.array([100*Rand.random(), 100*Rand.random(), 100*Rand.random()]),
	                    velocity = np.array([100*Rand.random(), 100*Rand.random(), 100*Rand.random()]),
	                    tValid = 5,
	                    sensorID = "sensor5"))
	## TODO: add tests for build_pos_cost_matrix_MT, build_pos_vel_cost_matrix_MT



