import torch
import numpy as np
import argparse

import cv2
import cvbase as cvb 
#ZS: for flo visualization "pip install cvbase"
import matplotlib.pyplot as plt

from models import FlowNet2  # the path is depended on where you create this module
from utils.frame_utils import read_gen  # the path is depended on where you create this module
from utils.flow_utils import flow2img
from math import ceil

from utils.preprocessing import ROI_for_gas, ROIinImg_for_gas, draw_rect
from utils.imageshow import flownetShow, opencvShow, cvbShow
from utils.OpticalFlowAlgo import Brox, farneback, flow_to_color


def ReadImg_Ori(path1, path2, IsGas):
    # load the image pair, you can find this operation in dataset.py


    pim1 = read_gen(path1)
    pim2 = read_gen(path2)

    if IsGas == 1:
        pim1 = np.float32(cv2.imread(path1, 1)) #-1 is last one "IMREAD_IGNORE_ORIENTATION", 1 is "IMREAD_GRAYSCALE"
        pim2 = np.float32(cv2.imread(path2, 1))

    print(pim1.shape)
    print(pim1.dtype)

    images = [pim1, pim2]
    print("before")
    #print(images.size())
    #print(images.dtype)
    images = np.array(images).transpose(3, 0, 1, 2)   
    #比如，你有一张 [公式] 的图片，你想把表示rgb的"3"那一维提到最前面，那就这样imshow：
    #img.transpose(2,0,1) # before: 0,1,2 after: 2,0,1

    print("after")

    #print(images.shape)
    #print(images.dtype)

    im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda() 

    print(im.dtype)
    print("im.type is" + str(im.type()))
    print("im.size is" + str(im.size()))
    # process the image pair to obtian the flow
    result = net(im).squeeze()

    return result

def ReadImg_JR(path1, path2, path3):
    # load the image pair, you can find this operation in dataset.py

    img1 = cv2.imread(path1, 1)
    img2 = cv2.imread(path2, 1)
    img3 = cv2.imread(path3, 1)

    diffimg1 = cv2.absdiff(img1, img2)
    diffimg2 = cv2.absdiff(img2, img3)

    #pim1 = np.float32(cv2.imread(path1, 1))
    #pim2 = np.float32(cv2.imread(path2, 1))

    #pim1 = cv2.fastNlMeansDenoising(img1, None, 10, 10, 7, 21)
    #pim2 = cv2.fastNlMeansDenoising(img2, None, 10, 10, 7, 21)

    pim1 = np.float32(diffimg1)
    pim2 = np.float32(diffimg2)
    

    #print(pim1.shape)
    #print(pim1.dtype)    H, W = pim1.shape[:2]
    #divisor = 64.
    #H_ = int(ceil(H/divisor) * divisor)
    #W_ = int(ceil(W/divisor) * divisor)  

    H, W = pim1.shape[:2]
    divisor = 64.
    H_ = int(ceil(H/divisor) * divisor)
    W_ = int(ceil(W/divisor) * divisor)
    print(W, W_)
    print(H, H_)    
    print("after")
    #print(pim1.shape)
    #print(pim1)


    pim1 = cv2.resize(pim1, (W_, H_))/255.
    pim2 = cv2.resize(pim1, (W_, H_))/255.
    print("before")
    #print(pim1.shape)
    #print(pim1)
    images = [pim1*1, pim2*1]
    print("after")
    #print(pim1.shape)
    #print(pim1)
    #images = np.array(images).reshape(6, H_, W_)

    images = np.array(images).transpose(3, 0, 1, 2) 
    #images = np.array(images).reshape(6, H_, W_)

    #print(images.shape)
    #print(images.dtype)

    im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()

    print(im.dtype)
    print("im.type is" + str(im.type()))
    print("im.size is" + str(im.size()))
    # process the image pair to obtian the flow
    result = (net(im)[0]*20.).squeeze()

    return result



if __name__ == '__main__':
    # obtain the necessary args for construct the flownet framework
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    
    args = parser.parse_args()

    # initial a Net
    net = FlowNet2(args).cuda()
    print("here")
    # load the state_dict
    dict = torch.load("/home/shen/Data/testFN2/FlowNet2_best_1000_e-5_ts2.pth.tar")
    #dict = torch.load("/home/shen/Data/testFN2/checkpoints/FlowNet2_checkpoint_og.pth.tar")
    net.load_state_dict(dict["state_dict"], strict=False)
    #net.load_state_dict(dict["state_dict"]) #ZS: changed



    path1_MPI = "/home/shen/Data/testFN2/0000007-img0.ppm"
    path2_MPI = "/home/shen/Data/testFN2/0000007-img1.ppm"
    
    path1_gas = "/home/shen/Data/imgs/A/frame_0070.png"
    path2_gas = "/home/shen/Data/imgs/A/frame_0073.png"
    path3_gas = "/home/shen/Data/imgs/A/frame_0076.png"

    path1_wkgas = "/home/shen/Data/testFN2/WKVideo/capture_image/1.jpg"
    path2_wkgas = "/home/shen/Data/testFN2/WKVideo/capture_image/2.jpg"
    path3_wkgas = "/home/shen/Data/testFN2/WKVideo/capture_image/3.jpg"


    #result = ReadImg_Ori(path1_MPI,path2_MPI, 0) #worked
    #result = ReadImg_Ori(path1_gas, path2_gas) #not worked, because the gas image has only one channel.
    #result = ReadImg_Ori(path1_gas, path2_gas, 1) #not worked
    #result = ReadImg_JR(path1_gas, path2_gas, path3_gas) # worked, but the parameter are not suitable.
    #result = ReadImg_JR(path1_wkgas, path2_wkgas) # worked, but the parameter are not suitable.

    #result = farneback(path1_MPI, path2_MPI) #worked
    #result = farneback(path1_wkgas, path2_wkgas) # worked, but the parameter are not suitable.
    #result = farneback(path1_gas, path2_gas)
    img = draw_rect(path2_gas)
    #roi = ROIinImg_for_gas(path1_gas)
    plt.imshow(img, interpolation='nearest')
    plt.show()

    #result = Brox(path1_gas, path2_gas)
    #opencvShow(result)

    #rgb = flow_to_color(result, hsv)
"""
    # save flow, I reference the code in scripts/run-flownet.py in flownet2-caffe project
    def writeFlow(name, flow):
        f = open(name, 'wb')
        f.write('PIEH'.encode('utf-8'))
        np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
        flow = flow.astype(np.float32)
        flow.tofile(f)
        f.flush()
        f.close()


    data = result.data.cpu().numpy().transpose(1, 2, 0)
    writeFlow("/home/shen/Data/testFN2/test1.flo", data)
    flow = cvb.optflow.io.read_flow('/home/shen/Data/testFN2/test1.flo')

    #opencvShow(flow)
    #cvbShow(flow)
    #flownetShow(flow)
    #JRshow(flow)


"""
    
