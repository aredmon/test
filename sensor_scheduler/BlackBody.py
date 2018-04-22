"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*   Module Name: utilities.py                                                                               *
*   Author(s): Brent McCoy                                                                                  *
*   Version: 1.0                                                                                            *
*   Date: 04/19/18                                                                                          *
*                                                                                                           *
*       Description:  Approximate signal-to-noise value calculation based on sensor                         *
*                     pass-band and object temperature                                                      *
*                                                                                                           *
*       Algorithm:    1. Compute the aspect angle from the sensor to the object                             *
*                     2. Find the object's blackbody radiance of the pass band                              *
*                     3. Determine the emissive area                                                        *
*                     4. Calculate the irradiance at the sensor aperture                                    *
*                     5. Compute the SNR and return                                                         *
*                                                                                                           *
*       Inputs:       slantRange - range from sensor to object, meters                                      *
*                     feature    - dictionary of object properties (geometry, temperature)                  *
*                     SAPs       - system adjustable parameters                                             *
*                     aspect     - aspect angle in radians (default is 45 deg or pi/4)                      *
*                                                                                                           *
*       Outputs:      snr -       signal-to-noise ratio (linear)                                            *
*                     irr_W_cm2 - irradiance at the sensor aperture, W/cm^2                                 *
*                                                                                                           *
*       Calls:        ShapeProjectedArea                                                                    *
*                                                                                                           *
*       OA:           M. A. Lambrecht                                                                       *
*                                                                                                           *
*       History:      MAL 29 Mar 2018:  Initial version                                                     *
*                                                                                                           *
*************************************************************************************************************
"""
import numpy as np
import importModules as mods

"""
----------------------------------------------------------------------------------------------
        Inputs:     wavelength  -   wavelength in [m]
                    temp        -   temperature in [deg-k]

        Outputs:    exitance    -   exitance in [W/m^3]
----------------------------------------------------------------------------------------------
"""
def PlanckBB(wavelength, temp):
    # return the exitance in watts/m^3 given wavelength in [m] and blackbody temp in Kelvin
    K = mods.physMath.BOLTZMAN
    c = mods.physMath.SPEED_LIGHT
    h = mods.physMath.PLANCK

    exitance = 2.0*np.pi*h*np.square(c) / (np.sqrt(wavelength)*np.expm1(h*c/(wavelength*K*temp)))

    return exitance

def calculateSNR(slantRange, feaure, aspect=np.pi/4):
    # constants
    T_atm = 1.0 # assuming exoatmospheric observation (no absorption)
    features = mods.SimpleNamespace(feature)

    # find the target blackbody radiance over sensor wavelength (W/m^2/sr)
    def getPoints(lower, upper, precision):
        return np.linspace(lower, upper+1, precision)

    mwirDomain = getPoints(mods.SAPs.SKR_BAND1[0], mods.SAPs.SKR_BAND1[1], 1000)
    lwirDomain = getPoints(mods.SAPs.SKR_BAND2[0], mods.SAPs.SKR_BAND2[1], 1000)

    mwirRange = PlanckBB(mwirDomain, features.temperature)
    lwirRange = PlanckBB(lwirDomain, features.temperature)

    mwirRadiance = np.trapz(mwirRange, mwirDomain)
    lwirRadiance = np.trapx(lwirRange, lwirDomain)

    maxRadiance = max(mwirRadiance, lwirRadiance)

    # find the area emissivity 
    crossSection = mods.ShapeProjectedArea(feature.shape, feature.diameter / 2, feature.length, aspect))
    emArea = crossSection * feature.emissivity

    # irradiance at aperture of sensor (W/cm^2)
    irradiance = (emArea * T_atm * maxRadiance / np.square(slantRange)) / 1e4

    # signal-to-noise ratio
    signalToNoise = irradiance / mods.SAPs.SKR_NEFD

    # release results
    return signalToNoise, irradiance

"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""
