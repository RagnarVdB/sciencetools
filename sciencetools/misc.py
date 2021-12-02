import numpy as np
import math

def substitute(x, index, array):
    """Vervangt één element in een numpy array"""
    new_array = np.array(array)
    new_array[index] = x
    return new_array

def intersect(x_array, y_array1, y_array2):
    """Geeft intersects van twee numpy arrays"""
    intersect_indices = np.argwhere(np.diff(np.sign(y_array1 - y_array2))).flatten()
    intersects = x_array[intersect_indices]
    return intersects

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

def find_zero_indices(array):
    diffs = (np.diff(np.sign(array)) != 0)
    indices = np.argwhere(diffs == True)[:,0]
    return indices

def filterDf(df, key, value):
    newdf = df[df[key] == value]
    # newdf.reset_index(inplace=True)
    return newdf.reset_index()

def diffDf(image, i1, i2):
    for x in images.columns:
        try:
            if images[x][i1] != images[x][i2]:
                print("{}:         {} | {}".format(x, images[x][i1], images[x][i2]))
        except ValueError:
            pass

def getMatrix(df, column):
    return np.array(list(df[column]))