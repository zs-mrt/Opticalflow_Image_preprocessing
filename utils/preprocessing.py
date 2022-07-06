from networks.resample2d_package.resample2d import Resample2d
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mpimg


######Denoising Process#######
def OpenCV_denoising(img):
    
    blur = cv2.blur(img, (3,3))

    guassian = cv2.GaussianBlur(blur, (5,5), 1)
    

    return guassian




def Autoencoders(): #https://iq.opengenus.org/image-denoising-autoencoder-keras/
    return 0














######### Contrast Enhancement#####
#ZS: CE means constrast enhancement

def differential_CE(img1, img2): #only enhance the differential part between the input image pair.



    return res1, res2


def global_CE(img):

    return 0

def clahe_CE(img): #Contrast Limited Adaptive Histogram Equalization(CLAHE)

    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
    img_clahe = clahe.apply(img)

    return img_clahe


def stretching_CE(image): #Min-Max Contrast Stretching
    image_cs = np.zeros((image.shape[0],image.shape[1]),dtype = 'uint8')
 

    # Apply Min-Max Contrasting
    min = np.min(image)
    max = np.max(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image_cs[i,j] = 255*(image[i,j]-min)/(max-min)  
    return image_cs



#####draw rect###

def draw_rect(path):
    img = cv2.imread(path, 1)
    roi = img[20:170, 70:170] 
    cv2.rectangle(img,(70,20),(170,170),(0,255,0),2)
    
    return img


#########ROI###############

def ROI_for_gas(path):
    img = cv2.imread(path, 1)
    roi = img[20:170, 70:170]
    #backgroud = np.zeros((img.shape[0],img.shape[1]), np.uint8)
    #backgroud.fill(1)
    #res = backgroud + roi
    return roi

###############WTFFFFFFF!!!!!!! GAo bu DOng!!!!!#################
def ROIinImg_for_gas(path): #take the ROI then CE, put it back into the cropped img  
    img1 = cv2.imread(path, 1) #me
    img2 = cv2.imread(path, 1) #logo
    #print(img1.shape[:])

    roi = img1[20:170, 70:170] # the place for the processed image

    #processed_roi = clahe_CE(img2[20:170, 70:170])

    #print(roi.shape[:])
    #img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #location = img1[20:170, 70:170]
    
    ret, mask = cv2.threshold(img1, 10, 255, cv2.THRESH_BINARY) #mask is a white img with the original size
    mask = np.zeros(img1.shape[:2], dtype=np.uint8)
    #print(mask.dtype)
    mask_inv = cv2.bitwise_not(mask) # mask_inv total black
    #print(mask_inv.dtype)
    #print(mask_inv.shape[:])
    img1_bg = cv2.bitwise_and(roi, roi, mask = mask_inv)
    #img2_fg = cv2.bitwise_and(img, img, mask = mask)

    #dst = cv2.add(img1_bg, img2_fg)

    return img1_bg

#########Combination###########
#Global CE only after denoising  
'''
1. ROI -> diff CE(1,2,...) -> denoise(1,2,...) -> opticalflow
2. ROI -> denoise -> diff CE -> opticalflow
3. 


'''