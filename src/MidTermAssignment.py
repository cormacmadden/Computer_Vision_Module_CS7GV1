from sklearn.datasets import load_sample_images
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import pandas as pd
import math

def brighten_image(img,amt):
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h,s,v = cv.split(img)
    v=v.astype('float32')
    v=v+amt
    v = np.clip(v,0,255)
    v=v.astype('uint8')
    img = cv.merge([h,s,v])
    processedImage = cv.cvtColor(img, cv.COLOR_HSV2BGR)
    return processedImage

def saturate_image(img,amt):
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h,s,v = cv.split(img)
    s=s.astype('float32')
    s=s*amt
    s = np.clip(s,0,255)
    s=s.astype('uint8')
    img = cv.merge([h,s,v])
    processedImage = cv.cvtColor(img, cv.COLOR_HSV2BGR)
    return processedImage

def contrast_image(img,contrast, brightness):
    new_image = np.zeros(img.shape, img.dtype)
    #alpha = amt # Simple contrast control
    beta = 0    # Simple brightness control
    brightness += int(round(255*(1-contrast)/2))
    #adjusted = cv.convertScaleAbs(img, alpha=amt, beta=beta)
    #for y in range(img.shape[0]):
    #    for x in range(img.shape[1]):
    #        for c in range(img.shape[2]):
    #            new_image[y,x,c] = np.clip(alpha*img[y,x,c] + beta, 0, 255)
    brightness += int(round(255*(1-contrast)/2))
    return cv.addWeighted(img, contrast, img, 0, brightness)

def solarization(img, amt = 1):
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h,s,v = cv.split(img)
    v=v.astype('float32')
    v= v*amt
    v = np.clip(v,0,255)
    for i in range(len(v)):
        for j in range(len(v[i])):
            if(v[i][j] >= 255.0):
                v[i][j] = 0
    
    v=v.astype('uint8')
    img = cv.merge([h,s,v])
    processedImage = cv.cvtColor(img, cv.COLOR_HSV2BGR)
    return processedImage

def scale_image(img, percentage: float):
    width = int(img.shape[1] * percentage)
    height = int(img.shape[0] * percentage)
    dim = (width, height)
    return cv.resize(img,dim,cv.INTER_AREA)

def kernel_transform(img, amt):
    arr = np.zeros((amt,amt,3))
    arr += 0.1
    output = img * 0
    pad = (amt - 1) // 2
    output = cv.copyMakeBorder(img, pad, pad, pad, pad,
		                cv.BORDER_REPLICATE)
	#output = np.zeros((iH, iW), dtype="float32")
    
    for i in range(pad, img.shape[0]-pad,1):
        for j in range(pad, img.shape[1]-pad,1):
            #duplicate of base image, size of kernel
            roi = img[i-pad:i+pad+1,j-pad:j+pad+1]
            #base image section 
            k1 = (roi * arr)
            k2 = np.sum(k1, axis = 0)
            k3 = np.sum(k2, axis = 0)
            output[i][j]=k3
    return output

def create_kernel(dimemsion):
    arr = [0.1] * dimemsion
    return arr
