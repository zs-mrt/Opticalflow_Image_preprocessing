from utils.frame_utils import read_gen  # the path is depended on where you create this module
from utils.flow_utils import flow2img
import cv2
import cvbase as cvb 
import numpy as np
import matplotlib.pyplot as plt

def opencvShow(flow):
    img = cvb.flow2rgb(flow)
    '''
    cv2.imshow('flow', img)
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()
    '''
    plt.imshow(img, interpolation='nearest')
    plt.show()
    return 1

def cvbShow(flow):
    flow_img = cvb.optflow.visualize.flow2rgb(flow)
    cvb.optflow.visualize.show_flow(flow, 'flow', 5000)
    #flow = np.random.rand(100, 100, 2).astype(np.float32)

    return 1

def flownetShow(flow):
    img = flow2img(flow)
    plt.imshow(img, interpolation='nearest')
    plt.show()

    return 1

def JRshow(flo):
    #flo = result.data.cpu().numpy().transpose(1,2,0)
    flo = result.data.numpy().transpose(1,2,0)
    W = 320
    W_ = 320
    H = 240
    H_ = 256
    u_ = cv2.resize(flo[:,:,0],(W,H))
    v_ = cv2.resize(flo[:,:,1],(W,H))
    u_ *=W/float(W_)
    v_ *=H/float(H_)
    flo = np.dstack((u_,v_))
    cvbShow(flo)
    return 1
