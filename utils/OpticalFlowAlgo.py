from lib2to3.pgen2.token import OP
import cv2
from utils.frame_utils import read_gen
from utils.preprocessing import * 

def Brox(path1, path2):
    prev = cv2.imread(path1)
    next = cv2.imread(path2)

    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)


    brox = cv2.pythoncuda.gpuOpticalFlowBrox(prev, next, alpha=0.05, gamma=0.001, scale_factor=0.5,
                            inner_iterations=5, outer_iterations=4, solver_iterations=5)

    return brox


def farneback(path1, path2):

    prev = cv2.imread(path1)
    next = cv2.imread(path2)

    #prev = ROI_for_gas(path1)
    #next = ROI_for_gas(path2)

    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)

    prev = OpenCV_denoising(prev)
    next = OpenCV_denoising(next)

    #prev = stretching_CE(prev)
    #next = stretching_CE(next)

    print(str(prev.shape[:2])+" "+str(next.shape[:2]))



    flow = cv2.calcOpticalFlowFarneback(prev, next, flow = None, pyr_scale = 0.5, 
                                        levels = 3, winsize = 10, iterations = 3, poly_n = 5, poly_sigma = 1.2, flags = 0)

    #flow = cv2.pythoncuda.gpuOpticalFlowFarneback(prev, next, flow=None, pyr_scale=0.5, levels=5,
                                                    #winsize=10, iterations=10, poly_n=5, poly_sigma=1.1, flags=False)

    return flow



def flow_to_color(flow, hsv):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


