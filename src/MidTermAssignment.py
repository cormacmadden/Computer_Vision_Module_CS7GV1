from unittest import result
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

def sat_image(img,amt):
    
    #h,s,v = cv.split(img)
    processedImage=img[:,:].astype('float32')
    processedImage = processedImage[:,:]+amt
    #s=s*amt
    processedImage = np.clip(processedImage,0,255)
    processedImage=processedImage.astype('uint8')
    #img = cv.merge([h,s,v])
    #processedImage = cv.cvtColor(img, cv.COLOR_HSV2BGR)
    return processedImage

def contrast_image(img,contrast, brightness):
    new_image = np.zeros(img.shape, img.dtype)
    #alpha = amt # Simple contrast control
    beta = 0    # Simple brightness control
    brightness += int(round(255*(1-contrast)/2))
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

def image_temp(img, amt):
    
    b = img[:,:,0]
    g = img[:,:,1]
    r = img[:,:,2]
    b=b.astype('float32')
    r=r.astype('float32')
    
    b = b[:,:] - amt
    r = r[:,:] + amt
    
    b = np.clip(b,0,255)
    r = np.clip(r,0,255)
    
    b=b.astype('uint8')
    r=r.astype('uint8')

    output = np.zeros((img.shape[0],img.shape[1],3))
    output=output.astype('uint8')

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            output[i,j] = [b[i,j], g[i,j], r[i,j]]
    
    return output

def image_tint(img, amt):
    
    b = img[:,:,0]
    g = img[:,:,1]
    r = img[:,:,2]
    g=g.astype('float32')
    
    g = g[:,:] + amt
    
    g = np.clip(g,0,255)
    
    g=g.astype('uint8')

    output = np.zeros((img.shape[0],img.shape[1],3))
    output=output.astype('uint8')

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            output[i,j] = [b[i,j], g[i,j], r[i,j]]
    
    return output

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

# Python3 program change RGB Color
# Model to HSV Color Model
 
def rgb_to_hsv(r, g, b):
    
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    cmax = max(r, g, b)    # maximum of r, g, b
    cmin = min(r, g, b)    # minimum of r, g, b
    diff = cmax-cmin       # diff of cmax and cmin.
 
    # if cmax and cmax are equal then h = 0
    if cmax == cmin:
        h = 0
     
    # if cmax equal r then compute h
    elif cmax == r:
        h = (60 * ((g - b) / diff) + 360) % 360
 
    # if cmax equal g then compute h
    elif cmax == g:
        h = (60 * ((b - r) / diff) + 120) % 360
 
    # if cmax equal b then compute h
    elif cmax == b:
        h = (60 * ((r - g) / diff) + 240) % 360
 
    # if cmax equal zero
    if cmax == 0:
        s = 0
    else:
        s = (diff / cmax) * 100
 
    # compute v
    v = cmax * 100
    return h, s, v
 