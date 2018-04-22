import numpy as np


def yaw(angle_rad):
	return np.array( [ [np.cos(angle_rad), -np.sin(angle_rad), 0],
			           [np.sin(angle_rad), np.cos(angle_rad),  0],
	                   [0 , 0, 1] ] )
