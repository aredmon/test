"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*   Module Name: setConstants.py                                                                            *
*   Author(s): Brent McCoy                                                                                  *
*   Version: 1.0                                                                                            *
*   Date: 12/20/17                                                                                          *
*                                                                                                           *
*       Module Description:                                                                                 *
*           Python script used to set constant variables and store their outputs as json files. This script *
*           only needs to be run if you make a change to the constants that you want to use. The file could *
*           be used as an import but the json functionality provides a cleaner level of interaction as the  *
*           '.py' files are intended to be the functional algorithm pieces.                                 *
*************************************************************************************************************
"""
import os
import json
import numpy as np
currentDir = os.path.dirname(os.path.realpath(os.getcwd()))
jsonFiles={}
"""
===================================== PHYSICAL AND MATHEMATICAL CONSTANTS ===================================
"""
info={}
data={}
#
#--------------------------------- conversion factors ------------------------------------------
#
data.update({'D2R'          :       np.pi/180})
info.update({'D2R'          :       'degrees to radians'})
#
data.update({'R2D'          :       180/np.pi})		
info.update({'R2D'          :       'radians to degrees'})
#
data.update({'D2S'          :       24*60*60})			
info.update({'D2S'          :       'days to seconds'})
#
data.update({'S2D'          :       1/data["D2S"]})		
info.update({'S2D'          :       'seconds to days'})
#
data.update({'KG2LB'        :       2.20462262})		
info.update({'KG2LB'        :       'kilograms to pounds'})
#
data.update({'ft2m'         :       0.3048})
info.update({'ft2m'         :       'feet to meters'})
#
data.update({'mile2m'       :       1609.344})
info.update({'mile2m'       :       'land mile to meters'})
#
data.update({'nm2m'         :       1852})
info.update({'nm2m'         :       'nautical mile to meters'})
#
data.update({'mile2ft'      :       5280})
info.update({'mile2ft'      :       'land mile to feet'})
#
data.update({'mileph2mps'   :       0.44704})
info.update({'mileph2mps'   :       'miles per hour to meters per second'})
#
data.update({'mile2kmph'    :       1.609344})
info.update({'mile2kmph'    :       'miles per hour to kilometers per hour'})
#
data.update({'nmph2kmph'    :       1.852})
info.update({'nmph2kmph'    :       'nautical miles per hour to kilometers per hour'})
#
#------------------------------------ mathematical --------------------------------------------
#
data.update({'twopi'        :       2.0 * np.pi})
info.update({'twopi'        :       'shortcut for 2pi'})
#
data.update({'halfpi'       :       np.pi * 0.5})
info.update({'halfpi'       :       'shortcut for pi/2'})
#
#-------------------------------------- physical ----------------------------------------------
#
info.update({'J2000' :              'J2000.0 epoch in Julian days'})
data.update({'J2000' :              2451545.0})
#
data.update({'SPEED_LIGHT' :        299792458.0})        
info.update({'SPEED_LIGHT' :        'speed of light (m/s)'})
#
data.update({'BOLTZMAN'	:           1.380662e-23})    
info.update({'BOLTZMAN'	:           'Boltzmann (mks units)'})
#
data.update({'PLANCK' :             6.6262e-34})       
info.update({'PLANCK' :             'Plank Constant (mks units)'})
#
data.update({'au'           :       149597870.0})      
info.update({'au'           :       'astronomical unit (distance from earth to sun in km)'})      
#
data.update({'earth2moon'   :       384400.0}) 
info.update({'earth2moon'   :       'distance from earth to moon in km'}) 
#
data.update({'moonradius'   :       1738.0}) 
info.update({'moonradius'   :       'radius of the moon in km'}) 
#
data.update({'sunradius'    :       696000.0}) 
info.update({'sunradius'    :       'radius of the sun in km'}) 
#
data.update({'masssun'      :       1.9891e30})
info.update({'masssun'      :       'mass of the sun in kg'})
#
data.update({'massearth'    :       5.9742e24})
info.update({'massearth'    :       'mass of the earth in kg'})
#
data.update({'massmoon'     :       7.3483e22})
info.update({'massmoon'     :       'mass of the moon in kg'})
##
physMathConstants = {"info": info, "data": data}
jsonFiles.update({"physMathConstants": physMathConstants})

"""
============================================= EARTH PARAMETERS ==============================================
"""
info={}
data={}
#
#-------------------------------------- physical ----------------------------------------------
#
info.update({'R_EARTHa' :           'Earth semi-major axis [m]'})
data.update({'R_EARTHa' :           6378136.3})
#
info.update({'R_EARTHb' :           'Earth semi-minor axis [m] (b = a - f*a)'})
data.update({'R_EARTHb' :           6356752.3142})
#
info.update({'R_EARTH' :            'Earth mean radius [m] ( R = (2a+b)/3'})
data.update({'R_EARTH' :            6371008.7714})
#
info.update({'FLATTEN' :            'Earth flattening (a - b) / a : f = (a - b) / a'})
data.update({'FLATTEN' :            1 / 298.257223563})
#
info.update({'OMEGA_EARTH' :        'Earth angular rotation rate [rad/sec]'})
data.update({'OMEGA_EARTH' :        7.2921151467e-5})	
#
info.update({'MU' :                 'Gravitational constant (m^3/s^2)'})
data.update({'MU' :                 3.986004415e14})	
#
info.update({'J2' :                 'Dynamical form factor'})
data.update({'J2' :                 1.082626683553150e-3})
#
info.update({'GAMMA_EQ' :           'Normal gravity at equator [m/s^2]'})
data.update({'GAMMA_EQ' :           9.7803253359})
#
info.update({'GAMMA_PL' :           'Normal gravity at poles [m/s^2]'})
data.update({'GAMMA_PL' :           9.8321849378})
#
info.update({'GAMMA_MN' :           'Mean normal gravity on ellipsoid [m/s^2]'})
data.update({'GAMMA_MN' :           9.7976432222})
#
info.update({'G0' :                 'Gravity at earth surface (m/s2)'})
data.update({'G0' :                 9.80665})
##
#info.update({'tu' :                 'Canonical time unit [solar sec] (sqrt(a^3/mu))'})
#data.update({'tu' :                 8.068111238242922e+02})
#
info.update({'spd' :                'Canonical earth rotation unit [km / solar sec] (a / tu)'})
data.update({'spd' :                7.905367017459467e+03})
#                                   
info.update({'massearth' :          'Mass of the earth in kg'})
data.update({'massearth' :          5.9742e24})
#
#------------------------------ derived constants from physical ------------------------------
#
data.update({'eccearth' :	    np.sqrt(2.0*data["FLATTEN"] - data["FLATTEN"]**2)})
info.update({'eccearth' :	    'eccentricity of the earth'})
#
data.update({'eccearthsqrd' :	    data["eccearth"]**2})
info.update({'eccearthsqrd' :	    'square of the eccentricity of the earth'})
#
data.update({'renm' :	            data["R_EARTH"] / physMathConstants["data"]["nm2m"]})
info.update({'renm' :	            'radius of the earth in nautical miles'})
#                                   
data.update({'reft' :	            data["R_EARTH"] * 1000.0 / physMathConstants["data"]["ft2m"]})
info.update({'reft' :	            'radius of the earth in feet'})
#                                  
data.update({'tusec' :	            np.sqrt(data["R_EARTH"]**3/data["MU"])})
info.update({'tusec' :	            'canonical time unit [solar sec] (sqrt(r^3/mu))'})
#                                   
data.update({'tumin' :	            data["tusec"] / 60.0})
info.update({'tumin' :	            'canonical time unit [solar min] (sqrt(r^3/mu))'})
#                                   
data.update({'tuday' :	            data["tusec"] / 86400.0})
info.update({'tuday' :	            'canonical time unit [solar day] (sqrt(r^3/mu))'})
#
data.update({'omegaearthradptu' :   data["OMEGA_EARTH"] * data["tusec"]})
info.update({'omegaearthradptu' :   'number of radians the earth rotates through in 1 solar sec'})
#
data.update({'omegaearthradpmin' :  data["OMEGA_EARTH"] * 60.0})
info.update({'omegaearthradpmin' :  'number of radians the earth rotates through in 1 solar min'})
#
data.update({'velkmps' :            np.sqrt(data["MU"] / data["R_EARTH"])})
info.update({'velkmps' :            'earth escape velocity in kilometers per second'})
#                                   
data.update({'velftps' :            data["velkmps"] * 1000.0/physMathConstants["data"]["ft2m"]})
info.update({'velftps' :            'earth escape velocity in feet per second'})
#                                   
data.update({'velradpmin' :         data["velkmps"] * 60.0/data["R_EARTH"]})
info.update({'velradpmin' :         'earth escape velocity in rads/min'})
#                                   
data.update({'degpsec' :            physMathConstants["data"]["R2D"] / data["tusec"]})
info.update({'degpsec' :            'number of degrees the earth rotates through in 1 sec'})
#                                   
data.update({'radpday' :            physMathConstants["data"]["twopi"] * 1.0027379095079})
info.update({'radpday' :            'number of radians the earth rotates through in 1 day'})
#
data.update({'H_TROPO' :            7400})             
info.update({'H_TROPO' :            'Height of troposhpere (m)'})
#
data.update({'ARIES' :              0})				
info.update({'ARIES' :              'Default Point of Aries angle.'})
##
earthParams = {"info": info, "data": data}
jsonFiles.update({"earthParams": earthParams})

"""
================================================= TOM SAPS ==================================================
"""
info={}
data={}
#
data.update({'BIG_L'        :       1e10})
info.update({'BIG_L'        :       'algorithm parameter'})
#
data.update({'EIGENVALUE_DIFF_THRESHOLD'    :   0.01})
info.update({'EIGENVALUE_DIFF_THRESHOLD'    :   'algorithm parameter'})
#
data.update({'SIGMA_FACTOR' :       1 })
info.update({'SIGMA_FACTOR' :       'covariance multiplier'})
#
data.update({'CARRIER_SENSOR' :     0 })
info.update({'CARRIER_SENSOR' :     '1 = centralized, carrier has sensor;  0 = distributed, no carrier sensor'})
##
TOM_SAPS = {"info": info, "data": data}
jsonFiles.update({"TOM_SAPS": TOM_SAPS})

"""
=========================================== EXPORT TO JSON FILES ============================================
"""
for key in jsonFiles:
    filename = key
    with open(os.path.join(currentDir, 'generalUtilities' , key + '.json'), 'w') as outfile:
        json.dump(jsonFiles[key], outfile, encoding='utf8', indent=4)

"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""
