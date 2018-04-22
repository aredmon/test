"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*   Module Name: config.py                                                                                  *
*   Author(s): Brent McCoy                                                                                  *
*   Version: 1.0                                                                                            *
*   Date: 02/12/18                                                                                          *
*                                                                                                           *
*       jsonData: returns an object with the json keys mapped to fields                                     *
*                 to access the data use object.field syntax. For information                               *
*                 on the available fields you can use the                                                   *
*                                                                                                           *
*       jsonInfo: returns pull out descritions of each field.                                               *
*                                                                                                           *
*************************************************************************************************************
"""
import os, sys
import numpy as np
from generalUtilities.Classes import jsonData, jsonInfo
"""-------------------------------------------- PATH SETUP -----------------------------------------------"""
if not os.path.isdir('generalUtilities'):
    if os.getcwd().rsplit('/', 1)[-1] != 'generalUtilities':
        currentDir = os.path.dirname(os.path.realpath(os.getcwd()))
        filePrefix = os.path.join(currentDir, 'generalUtilities')
    else:
        filePrefix = ''
else:
    filePrefix = 'generalUtilities'
"""---------------------------------- ATMOSPHERIC DATA TABLE IMPORT --------------------------------------"""
if filePrefix == '':
    filename = 'dataTable.npz'
else:
    filename = os.path.join(filePrefix, 'dataTable.npz')

#load file
dataTable = np.load(filename)
density_data = dataTable['atmosData']
"""-------------------------------------- EARTH PARAMETERS IMPORT ----------------------------------------"""
if filePrefix == '':
    filename = 'earthParams.json'
else:
    filename = os.path.join(filePrefix, 'earthParams.json')

# read in json file
eParms = jsonData(filename)
"""-------------------------------------- PHYS/MATH CONSTANTS --------------------------------------------"""
if filePrefix == '':
    filename = 'physMathConstants.json'
else:
    filename = os.path.join(filePrefix, 'physMathConstants.json')

# read in json file
physMath = jsonData(filename)
"""----------------------------------------- TOM SAPS IMPORT ---------------------------------------------"""
if filePrefix == '':
    filename = 'TOM_SAPS.json'
else:
    filename = os.path.join(filePrefix, 'TOM_SAPS.json')

# read in json file
TOM_SAPS = jsonData(filename)
"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""
