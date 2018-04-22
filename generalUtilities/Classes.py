"""****************************** Classification:   Unclassified *****************************************"""
"""
*************************************************************************************************************
*   Module Name: Classes.py                                                                                 *
*   Author(s): Brent McCoy                                                                                  *
*   Version: 1.0                                                                                            *
*   Date: 12/20/17                                                                                          *
*                                                                                                           *
*       Module Description:                                                                                 *
*           Universal classes used by the different aspects of the rms_mokv algorithm. Broadly defined as   *
*           utility classes and function classes. Utility classes provide additional accessability options  *
*           while functional classes pertain to a specific functionality or operation that the algorithm    *
*           needs to perform.                                                                               *
*************************************************************************************************************
"""
import sys
import json
import numpy as np

"""
--------------------------------------------- JSON2NP CONVERTER ------------------------------------------
"""
def np2jsonDefault(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj.__dict__

"""
----------------------------------------------------------------------------------------------
    Inputs:     arrays      -   n >= 2 1 dimensional array (must all have the same length)

    Outputs:    new_object  -   concatenated array along associated dimension
    
    axis:       invert      -   flag that indicates vertical or horizontal zipping
                            -   0 = vertical (default)  array elements are treated as rows
                            -   1 = horizontal  array elements are treated as columns
----------------------------------------------------------------------------------------------
"""
def zipArray(arrays, axis=0):
    if isinstance(axis, int):
        try:
            shapes = [array.shape[0] for array in arrays]
            ndims  = [array.ndim for array in arrays]
            # zip will by default only combine up to the shortest array
            # currently this is unwanted so the ValueError exception is thrown
            if len(set(shapes)) > 1 or not np.array_equal(ndims, np.ones_like(ndims)):
                if not len(set(shapes)) > 1:
                    raise ValueError("all arrays must be 1 dimensional: {}".format(ndims))
                else:
                    raise ValueError("all arrays must have the same length: {}".format(shapes))
            if axis == 0:
                combinedArray = np.asarray( zip(*arrays) )
            else:
                combinedArray = np.squeeze( np.asarray( zip(arrays) ) )
            return combinedArray

        except IndexError:
            print("'arrays' argument must be a tuple or a list")
            return {"arrays": locals()["arrays"], "axis": locals()["axis"]}      
    else:
        raise TypeError("axis argument must be of type integer")
"""
--------------------------------------------- UTILITY CLASSES --------------------------------------------
"""
if sys.version_info[0] < 3:
    class SimpleNamespace(object):
        def __init__(self, kwargs):
            self.__dict__.update(kwargs)

        def __repr__(self):
            keys = sorted(self.__dict__)
            items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
            return "{}({})".format(type(self).__name__, ", ".join(items))

        def __eq__(self, other):
            return self.__dict__ == other.__dict__

        def fields(self):
            return sorted(self.__dict__)

        def toDict(self):
            return self.__dict__

else:
    from types import SimpleNamespace

class jsonData(SimpleNamespace):
    def __init__(self, jsonFile):
        with open(jsonFile, 'r') as f:
            dataDict = json.load(f, encoding='utf8')

        super(jsonData, self).__init__(dataDict['data'])

class jsonInfo(SimpleNamespace):
    def __init__(self, jsonFile):
        with open(jsonFile, 'r') as f:
            dataDict = json.load(f, encoding='utf8')

        super(jsonData, self).__init__(dataDict['info'])

"""
--------------------------------------------- DATA STORING CLASSES --------------------------------------------
"""
class CVObject(object):
    def __init__(self, state, fuel):
        self.state = state
        self.fuel = fuel

    def __len__(self):
        return len(vars(self))

class KVObjects(CVObject):
    def __init__(self, states, fuel):
        self.states = states
        self.fuel = fuel

    def __len__(self):
        return self.states.shape[0]

class TOMObject(object):
    def __init__(self, tomDat, tomCov, tomIds):
        self.tomDat = tomDat
        self.tomCov = tomCov
        self.tomIds = tomIds

    def __len__(self):
        return self.tomDat.shape[0]

class lethalityMatrix(object):
    def __init__(self, pLethal, confidence, ranking):
        self.pLethal = pLethal
        self.confidence = confidence
        self.ranking = ranking

    def getLethality(self, index=None):
        index = np.asarray(index)
        if index.ndim == 0:
            return self.pLethal, self.confidence, self.ranking
        else:
            try:
                return self.pLethal[index], self.confidence[index], self.ranking[index]
            except IndexError:
                print("IndexError: the index, {}, is out of bounds for an array of length {}".format(index, 
                    pLethal.size))

class cvState(object):
    def __init__(self, rCV, vCV, rTracks, time):
        self.rCurr = rCV
        self.vCurr = vCV
        self.CVup = rCV
        self.time = time
        tmpVec = np.mean(rTracks, axis=0) - self.rCurr
        if np.linalg.norm(tmpVec) != 0:
            self.CVpointing = tmpVec / np.linalg.norm(tmpVec)
        else:
            self.CVpointing = tmpVec

    def extract(self, JSON=False):
        if not JSON:
            return self.__dict__
        else:
            return json.dumps(self.__dict__,  ensure_ascii=True, separators=(',', ': '), 
                    default=np2jsonDefault, indent=2)

class trackStates(object):
    def __init__(self, rTracks, vTracks, lethalityObj, **kwargs):
        self.rCurr = rTracks
        self.vCurr = vTracks
        # check for existing Ids field
        if 'Ids' in kwargs:
            self.Ids = kwargs["Ids"]
        else:
            self.Ids = np.arange(rTracks.shape[0])
        # check for individual lethalityMatrix fields
        if lethalityObj != None:
            self.pLethal, self.scpl, self.snr = lethalityObj.getLethality()
        else:
            self.pLethal = kwargs.get('pLethal')
            self.scpl = kwargs.get('scpl')
            self.snr = kwargs.get('snr')
        # auto generated fields
        self.active = np.ones_like(self.Ids, dtype=bool)
        self.collection = np.zeros((rTracks.shape[0], 11))

        self.collection[:, 0:6] = np.hstack((self.rCurr, self.vCurr))
        self.collection[:, 6:] = zipArray((self.Ids, self.active, self.pLethal, self.scpl, self.snr))

    def getTracks(self, index=None):
        index = np.asarray(index)
        if index.ndim == 0:
            return self.collection
        else:
            try:
                rawMat = self.collection[index]
                selectedTracks = trackStates(rawMat[:,0:3], rawMat[:,3:6], None, 
                        Ids=rawMat[:,6], pLethal=rawMat[:,8], scpl=rawMat[:,9], snr=rawMat[:,10])
                return selectedTracks
            except IndexError:
                print("IndexError: the index, {}, is out of bounds for an array of length {}".format(index, 
                    collection.shape[0]))

    def getActiveStates(self):
        index = self.active
        # index is a boolean array so we can use it to directly parse the values we want
        redTrackState = trackStates(self.rCurr[index], self.vCurr[index], None, 
                pLethal=self.pLethal[index], scpl=self.scpl[index], snr=self.snr[index], Ids=self.Ids[index])
        return redTrackState

    def extract(self, JSON=False):
        if not JSON:
            return self.__dict__
        else:
            return json.dumps(self.__dict__,  ensure_ascii=True, separators=(',', ': '), 
                    default=np2jsonDefault)

    def __len__(self):
        return self.collection.shape[0]

class simState(object):
    def __init__(self, time, **kwargs): 
        self.cvState = kwargs.get("cvState")
        self.trackStates = kwargs.get("trackStates")
        self.time = time
        if self.cvState == None:
            self.cvState = cvState(kwargs["rCV"], kwargs["vCV"], kwargs["rTracks"], time)
        if self.trackStates == None:
            self.trackStates = trackStates(kwargs["rTracks"], kwargs["vTracks"], kwargs["lethalityObject"])

    def extract(self, JSON=False):
        result = {"cvstate": self.cvState.extract(), "trackStates": self.trackStates.extract(), 
                "time": self.time}
        if not JSON:
            return result
        else:
            return json.dumps(result,  ensure_ascii=True, separators=(',', ': '), 
                    default=np2jsonDefault)

            
class stateCollection(object):
    def __init__(self, timeArray, **kwargs):
        self.state = [None] * timeArray.size
        for count, step in enumerate(timeArray):
            self.state[count] = simState(step, **kwargs)
            
    def getState(self, time, unpack=False):
        stateIndex = [state.time for state in self.state].index(time)
        if not unpack:
            return self.state[stateIndex]
        else:
            return (self.state[stateIndex].cvState, self.state[stateIndex].trackStates, 
                    self.state[stateIndex].time)
        
"""
*************************************************************************************************************
********************************* Classification:   Unclassified ********************************************
"""
