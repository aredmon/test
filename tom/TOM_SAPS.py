"""
****************************************************
Module Name: TOM_SAPS.py
Author(s): Larry Gariepy
Version: 1.0
Date: 12/04/17

    Module Description:
    	System Adjustable Parameters for TOM Correlation Algorithms
"""

## TOM Algorithm Parameters
BIG_L = 1e10
EIGENVALUE_DIFF_THRESHOLD = 0.01

## TOM Correlation Study Parameters
SIGMA_FACTOR = 1  # covariance multiplier
CARRIER_SENSOR = 0  # 1 = centralized, carrier has sensor;  0 = distributed, no carrier sensor

