"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*   Module Name: ProjectedArea                                                                              *
*   Author(s): Brent McCoy, Mark Lambrecht                                                                  *
*   Version: 1.0                                                                                            *
*   Date: 04/19/18                                                                                          *
*                                                                                                           *
*       Function:     ShapeProjectedArea                                                                    *
*                                                                                                           *
*       Description:  Calculated projected cross sectional area in m^2 given different                      *
*                     axially symmetric convex shape types of cylinder, sphere, or                          *
*                     cone with respect to aspect angle (0 = head on).  Inputs of base                      *
*                     radius, length in meters.                                                             *
*                                                                                                           *
*       Algorithm:    Various                                                                               *
*                                                                                                           *
*       Inputs:       ShapeType       - 'Sphere', 'Cylinder', 'Cone'                                        *
*                     Radius_m        - radius of object [m]                                                *
*                     Length_m        - length of object [m]                                                *
*                     AspectAngle_rad - aspect angle from observer to object [radians]                      *
*                                                                                                           *
*       Outputs:      A_m2 - area, [m^2]                                                                    *
*                                                                                                           *
*       Calls:        ShapeIntegrate (Local)                                                                *
*                     ProjectedDisc (Local)                                                                 *
*                                                                                                           *
*       OA:           M. A. Lambrecht                                                                       *
*                                                                                                           *
*       History:      MAL 29 Mar 2018:  Initial version                                                     *
*                                                                                                           *
*************************************************************************************************************
"""
import numpy as np
"""
----------------------------------------------------------------------------------------------
        Inputs:     radius      -   radius [m]
                    aspect      -   aspect angle [rads]

        Outputs:    crossArea   -   cross sectional area in [m^2]
----------------------------------------------------------------------------------------------
"""



"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""
