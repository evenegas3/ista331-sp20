import numpy as np
from datetime import datetime

def euclidean_distance(fv1, fv2): 
    ''' fv = feature vector '''
    features = set(fv1.index) & set(fv2.index)
    return sum((fv1[f] - fv2[f])**2 for f in features)**0.5 if len(features) > 0 else np.nan
    
def scaled_feature_vector(fv, lower, upper):
    scaled = fv.copy()
    for label in fv.index:
        scaled[label] = (fv[label] - lower[label]) / (upper[label] - lower[label])
    return scaled
        
def scaled_euclidean_distance(fv1, fv2, lower, upper):
    dist = euclidean_distance(scaled_feature_vector(fv1, lower, upper), \
        scaled_feature_vector(fv2, lower, upper))
    n = len(set(fv1.index) & set(fv2.index))
    return dist, n
    
def similarity(fv1, fv2, lower, upper):
    dist, n = scaled_euclidean_distance(fv1, fv2, lower, upper)
    return 1 - dist / n**0.5




