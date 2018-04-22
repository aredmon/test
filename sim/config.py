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
if not os.path.isdir('sim'):
    if os.getcwd().rsplit('/', 1)[-1] != 'sim':
        currentDir = os.path.dirname(os.path.realpath(os.getcwd()))
        filePrefix = os.path.join(currentDir, 'sim')
    else:
        filePrefix = ''
else:
    filePrefix = 'sim'
"""-------------------------------------- GENERAL SAPS IMPORT --------------------------------------------"""
if filePrefix == '':
    filename = 'SAPs_Null.json'
else:
    filename = os.path.join(filePrefix, 'SAPs_Null.json')

# read in json file
SAPs = jsonData(filename)
"""---------------------------------------- DOCTRINE FILE IMPORT -----------------------------------------"""
if filePrefix == '':
    filename = 'Doctrine_Null.json'
else:
    filename = os.path.join(filePrefix, 'Doctrine_Null.json')

# read in json file
doctrine = jsonData(filename)
"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""
