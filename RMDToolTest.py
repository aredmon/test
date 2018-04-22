"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*
*                                                                                                           *
*************************************************************************************************************
"""
import os, json, sys
import numpy as np

def np2jsonDefault(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj.__dict__

def serializeJSON(jsonData, scenarioFile="rmdToolOutput.json"):
    try:
        print("cleaning up old scenario data file")
        os.remove(scenarioFile)
    except OSError:
        pass

    # output serialized json object
    with open(scenarioFile, 'w') as outfile:
        json.dump(jsonData, outfile, ensure_ascii=True, separators=(',',': '), encoding='utf8', 
                default=np2jsonDefault, indent=0)

if __name__== "__main__":
    inputDictionary={}
    inputDictionary['cvState'] = np.random.rand(7)
    inputDictionary['truthStates'] = np.random.rand(12,7)
    inputDictionary['threatStates'] = np.random.rand(10,7)
    inputDictionary['thtRadius'] = 0.5

    serializeJSON(inputDictionary)