import numpy as np
# Returns array of locations of the features in the original data
def get_more_features(JMdictionary, firstClassNum, secClassNum, numOfClasses, numfeatures):
    newArray = JMdictionary[firstClassNum][secClassNum::numOfClasses] # jumps every NumOfClasses in order to get the correct match of first and sev class num at the JM matrix
    a = np.argsort(newArray) 
    featureIndices = np.argsort(a)[-numfeatures:]

    return featureIndices

