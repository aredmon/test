"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*   Module Name: setConstants.py                                                                            *
*   Author(s): Brent McCoy                                                                                  *
*   Version: 1.0                                                                                            *
*   Date: 12/20/17                                                                                          *
*                                                                                                           *
*       Module Description:                                                                                 *
*           Python script used to set constant variables and store their outputs as json files. This is     *
*           script only needs to be run if you make a change to the constants that you want to use. The     *
*           file could potentially be used as an import but the json functionality provides a cleaner level *
*           of interaction as the .py files are intended to be the functional algorithm pieces.             *
*************************************************************************************************************
"""
import os, sys
import json
import numpy as np
currentDir = os.path.dirname(os.path.realpath(os.getcwd()))
jsonFiles={}
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
================================================ SCENARIO SAPS ==============================================
"""
info={'SAPs': 'SAPs for scenario setup',
        'MDL_PLOT_ON' :              'bool', 
        'MDL_TIMESTEP' :             'float', 
        'MDL_T_START' :              'float', 
        'MDL_T_DURATION' :           'float', 
        'MDL_OUTPUT_ON' :            'bool', 
        'THT_SPEED_MIN' :            'float', 
        'THT_SPEED_MAX' :            'float', 
        'THT_MIN_NUM_OBJS' :         'int', 
        'THT_MAX_NUM_OBJS' :         'int', 
        'THT_MIN_NUM_CLOUDS' :       'int', 
        'THT_MAX_NUM_CLOUDS' :       'int', 
        'THT_MIN_RADIUS' :           'float', 
        'THT_MAX_RADIUS' :           'float', 
        'THT_POS_DEV' :              'float', 
        'THT_VEL_DEV' :              'float', 
        'CV_SPEED_MIN' :             'float', 
        'CV_SPEED_MAX' :             'float', 
        'CV_CLOSE_ANGLE_MIN' :       'float', 
        'CV_CLOSE_ANGLE_MAX' :       'float', 
        'CV_KFACTOR' :               'float', 
        'CV_RADIUS' :                'decimal', 
        'CV_MAX_DIVERT' :            'float', 
        'CV_MAX_ACC' :               'float', 
        'CV_FUEL_RESERVE' :          'float', 
        'CV_FINAL_ASSGN_TGO' :       'float', 
        'CV_PROB_KILL' :             'float', 
        'CV_FRAC_TRACKS_DETECTED' :  'float', 
        'CV_TGO_SENSOR_ON' :         'float', 
        'CV_POS_BIAS' :              'float', 
        'CV_VEL_BIAS' :              'float', 
        'CV_POS_NOISE' :             'float', 
        'CV_VEL_NOISE' :             'float', 
        'KV_NUM_KVS' :               'int', 
        'KV_MAX_DIVERT' :            'float', 
        'KV_MAX_ACC' :               'float', 
        'KV_FUEL_RESERVE' :          'float', 
        'KV_BATTERY_LIFE' :          'float', 
        'KV_DISPENSE_DV' :           'float', 
        'KV_T_STELLARCAL' :          'float', 
        'KV_T_HANDOVER' :            'float', 
        'KV_PROB_KILL' :             'float', 
        'RDR_MAX_TRACKS_TOM' :       'int', 
        'RDR_FRAC_TRACKS_DETECTED' : 'decimal', 
        'RDR_POS_BIAS' :             'float', 
        'RDR_VEL_BIAS' :             'float', 
        'RDR_POS_NOISE' :            'float', 
        'RDR_VEL_NOISE' :            'float', 
        'RDR_KFACTOR' :              'float', 
        'RDR_TOM_TGO' :              'float', 
        'WTA_COV_RADIUS_FRAC' :      'float', 
        'WTA_VALUE_CUTOFF' :         'decimal', 
        'WTA_PROP_T_MAX' :           'float', 
        'WTA_AVE_KV_PK' :            'decimal' 
        }
data={}

#
##
SAPs_Null = {"info": info, "data": data}
jsonFiles.update({"SAPs_Null": SAPs_Null})
"""
================================================ DOCTRINE NULL ==============================================
"""
info = {'Doctine': 'null structure for firing doctrine'}
data = {
        'maxShotsThreat' : 2, 
        'maxShotsWeapon' : 1, 
        'valueCutoff' :    0.099, 
        'minPk' :          0.4, 
        'avePk' :          0.8 }
#
##
Doctrine_Null = {"info": info, "data": data}
jsonFiles.update({"Doctrine_Null": Doctrine_Null})
"""
=========================================== EXPORT TO JSON FILES ============================================
"""
for key in jsonFiles:
    filename = key
    with open(os.path.join(currentDir, 'sim' , key + '.json'), 'w') as outfile:
        json.dump(jsonFiles[key], outfile, encoding='utf8', indent=4)

"""
=========================================== BUILDING TABLE FILES ============================================
"""
density_data = np.zeros([150])
data = [1.2254383217,
        1.1119681155,
        1.0067621942,
        0.9093876508,
        0.8194252061,
        0.7364690562,
        0.6601267209,
        0.5900188921,
        0.5257792812,
        0.4670544677,
        0.4135037462,
        0.3647989745,
        0.3190251134,
        0.2726479791,
        0.2330242901,
        0.1991689362,
        0.1702407458,
        0.1455214194,
        0.1243975510,
        0.1063452847,
        0.0909172195,
        0.0765495002,
        0.0652202638,
        0.0556107735,
        0.0474535254,
        0.0405236170,
        0.0346318063,
        0.0296187276,
        0.0253500660,
        0.0217125282,
        0.0186104772,
        0.0159631162,
        0.0137021326,
        0.0115759463,
        0.0098899729,
        0.0084655966,
        0.0072598017,
        0.0062370348,
        0.0053678445,
        0.0046277809,
        0.0039965022,
        0.0034570494,
        0.0029952548,
        0.0025992588,
        0.0022591142,
        0.0019664622,
        0.0017142655,
        0.0014965881,
        0.0013174594,
        0.0011634453,
        0.0010274765,
        0.0009074339,
        0.0008014480,
        0.0007105213,
        0.0006315787,
        0.0005609424,
        0.0004977877,
        0.0004413669,
        0.0003910023,
        0.0003460806,
        0.0003060467,
        0.0002703987,
        0.0002392987,
        0.0002124983,
        0.0001883506,
        0.0001666281,
        0.0001471203,
        0.0001296315,
        0.0001139810,
        0.0001000015,
        0.0000875385,
        0.0000764496,
        0.0000666035,
        0.0000578796,
        0.0000501672,
        0.0000433647,
        0.0000373789,
        0.0000321250,
        0.0000275254,
        0.0000235093,
        0.0000200123,
        0.0000166430,
        0.0000138418,
        0.0000115128,
        0.0000095762,
        0.0000079658,
        0.0000066267,
        0.0000055130,
        0.0000045867,
        0.0000038163,
        0.0000031699,
        0.0000025986,
        0.0000021373,
        0.0000017634,
        0.0000014595,
        0.0000012115,
        0.0000010086,
        0.0000008420,
        0.0000007048,
        0.0000005916,
        0.0000004994,
        0.0000004177,
        0.0000003508,
        0.0000002958,
        0.0000002503,
        0.0000002126,
        0.0000001813,
        0.0000001550,
        0.0000001330,
        0.0000001145,
        0.0000000978,
        0.0000000832,
        0.0000000712,
        0.0000000612,
        0.0000000530,
        0.0000000460,
        0.0000000402,
        0.0000000352,
        0.0000000310,
        0.0000000274,
        0.0000000244,
        0.0000000211,
        0.0000000184,
        0.0000000162,
        0.0000000143,
        0.0000000128,
        0.0000000114,
        0.0000000102,
        0.0000000092,
        0.0000000084,
        0.0000000076,
        0.0000000069,
        0.0000000063,
        0.0000000058,
        0.0000000053,
        0.0000000049,
        0.0000000046,
        0.0000000042,
        0.0000000039,
        0.0000000036,
        0.0000000034,
        0.0000000032,
        0.0000000030,
        0.0000000028,
        0.0000000026,
        0.0000000025,
        0.0000000023,
        0.0000000022,
        0.0000000021,
        0.0000000019,
        0.0000000018]

for i in range(150):
    density_data[i] = data[i]

filename = os.path.join(currentDir, 'sim', 'dataTable.npz')

if not os.path.isfile(filename):
    np.savez(filename, atmosData=density_data)
else:
    overwrite = raw_input("file, {}, already exists, do you want to recreate it? (y/n)\n".format(
        filename.rsplit('/',1)[-1]))
    if overwrite == "y":
        print("overwriting {}".format(filename.rsplit('/',1)[-1]))
        os.remove(filename) 
        np.savez(filename, atmosData=density_data)
    else:
        pass
"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""
