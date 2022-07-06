from builtins import breakpoint
import numpy as np
from os.path import *
from scipy.misc import imread
from . import flow_utils 

def read_gen(file_name):
    ext = splitext(file_name)[-1]
    #print(ext) # ZS: for debugging ext==png

    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        im = imread(file_name)
        #print(im.shape[:]) #ZS
        return im
        if im.shape[2] > 3:
            return im[:,:,:3]
        else:
            return im
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        return flow_utils.readFlow(file_name).astype(np.float32)
    return []
