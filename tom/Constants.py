"""
****************************************************
Module Name: Constants.py
Author(s): Larry Gariepy
Version: 1.0
Date: 12/04/17

    Module Description:
    	Physical and Mathematical Constants for TOM Correlation Algorithms
"""
import numpy as np

D2R				= np.pi/180;		# degrees to radians
R2D				= 180/np.pi;		# radians to degrees
D2S				= 24*60*60;			# days to seconds
S2D				= 1/D2S;			# seconds to days
KG2LB			= 2.20462262;		# kilograms to pounds

# WGS-84
# R_EARTH			= 6378137;
# FLATTEN			= 1/298.257223560;
# MU				= 3.986004418e14;
# W_EARTH			= 7.292115e-5;
J2				= 1.081874e-3;
#J2				= 1.082629989051944e-3;
#
R_EARTH			= 6378136.3;		# Radius of earth (m)
FLATTEN			= 1/298.257223563;  # WGS84 flattening term
MU				= 3.986004415e14;	# gravitational constant (m3/s2)
G_o				= 9.80665;          # gravity at earth surface (m/s2)
W_EARTH			= 7.2921151467e-5;	# earth rotation rate, (rad/sec)
SPEED_LIGHT		= 299792458;        # speed of light (m/s)
BOLTZMAN		= 1.380662e-23;     # Boltzmann (mks units)
PLANCK			= 6.6262e-34;       # Plank Constant (mks units)
H_TROPO			= 7400;             # Height of troposhpere (m)
ARIES			= 0;				# Default Point of Aries angle.

WX = np.array([ [0,-W_EARTH,0], [W_EARTH,0,0], [0, 0, 0]])   # Cross multiply earth rotation

RAD2SEC = 206264.806247096355;      
HR2RAD	 = 0.26179938779915;
T0 = 2451545.00000000;

CMAP = np.array( [[1,0,0],[0,0,1],[0,0.5,0],[.9,.6,0],[.4,.8,.7],[.1,.2,.1]] )  ## Colormap for plots
CMAP = np.vstack( [CMAP, CMAP, CMAP] )
