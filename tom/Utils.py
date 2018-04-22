# Utils.py
# Static methods for common operations in math, etc.

import numpy as np
import math


def yaw(angle_rad):
	return np.array( [ [np.cos(angle_rad), -np.sin(angle_rad), 0],
			           [np.sin(angle_rad), np.cos(angle_rad),  0],
	                   [0 , 0, 1] ] )

def pitch(angle_rad):
	return np.array( [ [np.cos(angle_rad), 0, np.sin(angle_rad)],
	                   [0 , 1,  0] ],
			           [-np.sin(angle_rad), 0,  np.cos(angle_rad)] )

## length of a vector (one-dimensional array)
def norm(v):
	return math.sqrt(np.dot(v,v))
	

def unitVector(v):
	d = norm(v)
	if d == 0:
		raise Exception("Cannot take unit vector of zero-length vector")
	return v/d

## Perform pitch rotation on the object scene
## Looking from positive Y to the origin, the rotation is counter-clockwise;  
## aka the positive x-axis rotates toward negative z for positive rotation angles
def pitch_scene(angle_rad, object_list):
	(r,c) = object_list.shape
	new_object_pos = np.zeros([r,c])
	raise Exception("not implemented yet")
	##TODO: FINISH!

#Make a class for 2D object scenes,  3D object scenes (or just one class?)
#Also need class for cost matrices;